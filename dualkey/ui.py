"""Dual-Key dashboard: live diagrams of discovery, the checking pipeline and detection.

  GET /            the dashboard page (polls /api/state every 3 seconds)
  GET /api/state   everything the page draws (see build_state)
  GET /?demo=1     a scripted demo, no cluster needed

Data sources:
  - Kubernetes: DualKeyCluster, ComparisonGroups, GPUHealthRecords, GPU nodes
  - Redis:      queue lengths, recent check results, the Detector's saved state
  - memory:     a score history sampled every 5 seconds (last hour)

Run: uvicorn dualkey.ui:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from .config import HASH_PAIRCOUNT, KEY_SCORER, STREAM_RESULTS, STREAM_VERIFY
from .vendor import VendorReader
from . import overhead as OH

log = logging.getLogger("dualkey.ui")
GROUP, VERSION = "dualkey.io", "v1alpha1"
NAMESPACE = os.environ.get("DK_NAMESPACE", "dualkey")
CLUSTER = os.environ.get("DK_CLUSTER", "")
REDIS_URL = os.environ.get("DK_REDIS_URL", f"redis://dualkey-redis.{NAMESPACE}.svc:6379/0")
GPU_RESOURCES = {"nvidia.com/gpu": "nvidia", "amd.com/gpu": "amd"}
HISTORY_POINTS = 720          # 1 hour at 5-second samples
RECENT_CHECKS = 400

HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        '</head><body>')


def page() -> str:
    return HEAD + resources.files("dualkey").joinpath("static/index.html").read_text() + "</body></html>"


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


# ------------------------------------------------------------------ pure helpers

def gpu_nodes(nodes: list[dict]) -> list[dict]:
    """Nodes that have at least one GPU, with the facts the diagram shows."""
    out = []
    for n in nodes:
        md, st = n.get("metadata") or {}, n.get("status") or {}
        labels = md.get("labels") or {}
        alloc = st.get("allocatable") or {}
        cap = st.get("capacity") or {}
        for res, vendor in GPU_RESOURCES.items():
            total = int(str(alloc.get(res, cap.get(res, "0")) or 0))
            if total > 0 or labels.get(f"{vendor}.com/gpu.present") == "true":
                product = (labels.get("nvidia.com/gpu.product") or labels.get("amd.com/gpu.product-name")
                           or labels.get("amd.com/gpu.product") or "")
                out.append({
                    "name": md.get("name", ""),
                    "short": md.get("name", "").split(".")[0],
                    "vendor": vendor,
                    "gpus": total,
                    "product": product,
                    "instanceType": labels.get("node.kubernetes.io/instance-type", ""),
                    "zone": labels.get("topology.kubernetes.io/zone", ""),
                    "tainted": any(t.get("key") == "dualkey.io/suspect" for t in (n.get("spec") or {}).get("taints") or []),
                })
                break
    return sorted(out, key=lambda x: x["name"])


def checks_summary(results: list[dict], now: float) -> dict:
    last = [r for r in results if now - r.get("ts", 0) <= 60]
    return {
        "lastMinute": len(last),
        "probeLastMinute": sum(1 for r in last if r.get("source") == "probe"),
        "gatewayLastMinute": sum(1 for r in last if r.get("source") == "gateway"),
        "last": results[0] if results else None,
    }


def zscore(gap: float, vendor_a: str, vendor_b: str, baselines: dict, floor: float = 0.01):
    key = "|".join(sorted([vendor_a or "", vendor_b or ""]))
    b = baselines.get(key)
    if not b or not b.get("frozen") or b.get("n", 0) < 2:
        return None
    import math
    std = max(math.sqrt(b["m2"] / (b["n"] - 1)), floor)
    return (gap - b["mean"]) / std


def scorer_view(raw: dict | None) -> dict:
    """Detector state per group: baselines (normal gap) and pair evidence."""
    import math
    out = {}
    for gname, g in ((raw or {}).get("groups") or {}).items():
        floor = (g.get("settings") or {}).get("std_floor", 0.01)
        bl = {}
        for k, b in (g.get("baselines") or {}).items():
            n = b.get("n", 0)
            std = max(math.sqrt(b["m2"] / (n - 1)), floor) if n > 1 else floor
            bl[k] = {"mean": b.get("mean", 0.0), "std": std, "samples": n, "ready": bool(b.get("frozen"))}
        out[gname] = {
            "settings": g.get("settings") or {},
            "baselines": bl,
            "rawBaselines": g.get("baselines") or {},
            "pairs": {k: {"z": round(v.get("ewma", 0.0), 3), "checks": v.get("n", 0)}
                      for k, v in (g.get("pairs") or {}).items()},
            "members": {k: {"state": v.get("state"), "score": v.get("score"), "checks": v.get("checks", 0),
                            "vendor": v.get("vendor")} for k, v in (g.get("gpus") or {}).items()},
        }
    return out


def annotate_recent(results: list[dict], scorer: dict, limit: int = 20) -> list[dict]:
    out = []
    for r in results[:limit]:
        g = scorer.get(r.get("group"), {})
        members = g.get("members", {})
        va = (members.get(r.get("primary")) or {}).get("vendor", "")
        vb = (members.get(r.get("checker")) or {}).get("vendor", "")
        z = zscore(float(r.get("gap", 0)), va, vb, g.get("rawBaselines", {}),
                   (g.get("settings") or {}).get("std_floor", 0.01))
        out.append({**{k: r.get(k) for k in ("ts", "group", "primary", "checker", "gap", "source", "tokens")},
                    "z": None if z is None else round(z, 2)})
    return out


def build_state(cluster: dict | None, groups: list[dict], records: list[dict], nodes: list[dict] | None = None,
                live: dict | None = None, history: dict | None = None, vendor: dict | None = None,
                incidents: list[dict] | None = None) -> dict:
    c = cluster or {}
    spec, status = c.get("spec") or {}, c.get("status") or {}
    live = live or {}
    scorer = live.get("scorer") or {}
    overhead = compute_overhead(spec, groups, nodes or [], live.get("results") or [])
    for g in scorer.values():
        g.pop("rawBaselines", None)
    return {
        "cluster": {
            "name": (c.get("metadata") or {}).get("name", ""),
            "mode": spec.get("mode") or ("manual" if spec.get("gpus") else "discover"),
            "phase": status.get("phase", "Discovering"),
            "message": status.get("message", ""),
            "discovery": status.get("discovery") or {},
            "events": status.get("events") or [],
            "detection": status.get("detection") or {},
            "referenceModel": (spec.get("model") or {}).get("name", "ibm-granite/granite-3.3-2b-instruct"),
            "thresholds": {"suspect": (spec.get("detector") or {}).get("suspectZ", 3),
                           "quarantine": (spec.get("detector") or {}).get("quarantineZ", 5)},
            "probeRate": (spec.get("probe") or {}).get("rate", 1.0),
            "sampleRate": (spec.get("sampling") or {}).get("rate", 0.05),
        },
        "groups": [{"name": g["metadata"]["name"], "spec": g.get("spec") or {}, "status": g.get("status") or {}}
                   for g in groups],
        "records": [{"name": r["metadata"]["name"], "spec": r.get("spec") or {}, "status": r.get("status") or {}}
                    for r in records],
        "nodes": gpu_nodes(nodes or []),
        "pipeline": live.get("pipeline") or {},
        "recent": live.get("recent") or [],
        "scorer": scorer,
        "history": history or {},
        "vendor": vendor or {"status": "not checked", "nodes": {}},
        "overhead": overhead,
        "incidents": [{"name": i["metadata"]["name"], "spec": i.get("spec") or {}, "status": i.get("status") or {},
                       "created": i["metadata"].get("creationTimestamp", "")} for i in incidents or []],
        "now": time.time(),
    }


def compute_overhead(spec: dict, groups: list[dict], nodes: list[dict], results: list[dict]) -> dict:
    member_nodes = {m.get("name"): m.get("node", "") for g in groups
                    for m in (g.get("status") or {}).get("members") or []}
    node_types = {(n.get("metadata") or {}).get("name"): ((n.get("metadata") or {}).get("labels") or {})
                  .get("node.kubernetes.io/instance-type", "") for n in nodes}
    return OH.summarize(results, time.time(), member_nodes, node_types,
                        (spec.get("cost") or {}).get("gpuHourUSD"),
                        float((spec.get("sampling") or {}).get("rate", 0.05)))


# ------------------------------------------------------------------ live readers

class Live:
    def __init__(self):
        self.redis = None
        self.history: dict[str, deque] = {}
        self.last_error = ""

    async def connect(self):
        if self.redis is None:
            import redis.asyncio as aioredis
            self.redis = aioredis.from_url(REDIS_URL, socket_timeout=2, socket_connect_timeout=2)
        return self.redis

    async def read(self) -> dict:
        try:
            r = await self.connect()
            raw = await r.get(KEY_SCORER)
            scorer = scorer_view(json.loads(_s(raw)) if raw else None)
            results = []
            for _id, fields in await r.xrevrange(STREAM_RESULTS, count=RECENT_CHECKS):
                try:
                    results.append(json.loads(_s(fields.get(b"data", fields.get("data")))))
                except Exception:
                    pass
            pending_v = await self._pending(STREAM_VERIFY, "verifiers")
            pending_r = await self._pending(STREAM_RESULTS, "detectors")
            pc = {_s(k): int(_s(v)) for k, v in (await r.hgetall(HASH_PAIRCOUNT)).items()}
            pipeline = {
                "verifyWaiting": pending_v, "resultsWaiting": pending_r,
                "verifyTotal": await r.xlen(STREAM_VERIFY), "checksTotal": sum(pc.values()),
                **checks_summary(results, time.time()),
            }
            recent = annotate_recent(results, scorer)
            self.last_error = ""
            return {"scorer": scorer, "pipeline": pipeline, "recent": recent, "results": results}
        except Exception as e:
            self.last_error = str(e)
            return {"pipeline": {"error": f"queue not reachable: {e}"}}

    async def _pending(self, stream: str, group: str) -> int:
        """Messages not yet picked up by the group plus picked up but unfinished."""
        try:
            groups = await self.redis.xinfo_groups(stream)
        except Exception:
            return 0
        for g in groups:
            name = _s(g.get("name") or g.get(b"name"))
            if name == group:
                lag = g.get("lag", g.get(b"lag")) or 0
                pending = g.get("pending", g.get(b"pending")) or 0
                return int(lag) + int(pending)
        return 0

    async def sample_history(self):
        while True:
            try:
                data = await self.read()
                now = time.time()
                for gname, g in (data.get("scorer") or {}).items():
                    for m, info in g.get("members", {}).items():
                        if info.get("score") is not None:
                            self.history.setdefault(m, deque(maxlen=HISTORY_POINTS)).append(
                                [round(now, 1), round(info["score"], 3)])
            except Exception as e:
                log.debug("history sample failed: %s", e)
            await asyncio.sleep(5)

    def history_json(self) -> dict:
        return {k: list(v) for k, v in self.history.items()}


def _read_k8s() -> tuple:
    from kubernetes import client, config as kcfg
    try:
        kcfg.load_incluster_config()
    except kcfg.ConfigException:
        kcfg.load_kube_config()
    api = client.CustomObjectsApi()
    clusters = api.list_namespaced_custom_object(GROUP, VERSION, NAMESPACE, "dualkeyclusters")["items"]
    cluster = next((c for c in clusters if not CLUSTER or c["metadata"]["name"] == CLUSTER),
                   clusters[0] if clusters else None)
    groups = api.list_namespaced_custom_object(GROUP, VERSION, NAMESPACE, "comparisongroups")["items"]
    records = api.list_namespaced_custom_object(GROUP, VERSION, NAMESPACE, "gpuhealthrecords")["items"]
    try:
        incidents = api.list_namespaced_custom_object(GROUP, VERSION, NAMESPACE, "gpuintegrityincidents")["items"]
    except Exception:
        incidents = []
    v1 = client.CoreV1Api()
    try:
        nodes = client.ApiClient().sanitize_for_serialization(v1.list_node().items)
    except Exception as e:  # dashboard still works without node details
        log.info("cannot list nodes: %s", e)
        nodes = []
    return cluster, groups, records, nodes, _ip_to_node(v1, nodes), incidents


def _ip_to_node(v1, nodes: list[dict]) -> dict:
    from .vendor import ip_to_node
    return ip_to_node(v1, nodes)


live = Live()
vendor_reader = VendorReader()


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(live.sample_history())
    yield
    task.cancel()


app = FastAPI(title="Dual-Key Dashboard", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    return page()


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/api/state")
async def state():
    try:
        cluster, groups, records, nodes, ip_map, incidents = await asyncio.to_thread(_read_k8s)
    except Exception as e:
        raise HTTPException(503, f"cannot read Dual-Key objects: {e}")
    data = await live.read()
    vendor = await vendor_reader.read(gpu_nodes(nodes), nodes, ip_map)
    return JSONResponse(build_state(cluster, groups, records, nodes, data, live.history_json(), vendor, incidents))


@app.get("/api/overhead")
async def overhead_report():
    """The overhead numbers on their own, for benchmark write-ups (curl it from a pod)."""
    try:
        cluster, groups, _records, nodes, _ip, _inc = await asyncio.to_thread(_read_k8s)
    except Exception as e:
        raise HTTPException(503, f"cannot read Dual-Key objects: {e}")
    data = await live.read()
    return JSONResponse(compute_overhead((cluster or {}).get("spec") or {}, groups, nodes, data.get("results") or []))
