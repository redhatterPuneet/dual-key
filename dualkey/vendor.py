"""What the GPU vendors' own monitoring says about each GPU node.

Sources, all read-only:
  - NVIDIA DCGM exporter metrics (installed by the NVIDIA GPU Operator)
  - AMD device metrics exporter metrics (installed by the AMD GPU Operator)
  - Node conditions set by health tools such as NVSentinel or Node Problem Detector

Metrics are read from OpenShift monitoring (Thanos querier) with the dashboard's
service account token; the account needs the cluster-monitoring-view role.
The dashboard shows this next to the Dual-Key verdict for the same GPU.
"""
from __future__ import annotations

import logging
import os
import time

log = logging.getLogger("dualkey.vendor")

PROM_URL = os.environ.get("DK_PROM_URL", "https://thanos-querier.openshift-monitoring.svc:9091")
SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
REFRESH_SECONDS = 15

# key -> (vendor, PromQL, label shown on the dashboard, is_error_counter)
QUERIES = {
    "xid":        ("nvidia", "max_over_time(DCGM_FI_DEV_XID_ERRORS[1h])", "XID error (last hour)", True),
    "eccDbe":     ("nvidia", "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", "ECC uncorrectable", True),
    "eccSbe":     ("nvidia", "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "ECC corrected", False),
    "remapFail":  ("nvidia", "DCGM_FI_DEV_ROW_REMAP_FAILURE", "Row remap failure", True),
    "remapUnc":   ("nvidia", "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS", "Rows retired (uncorrectable)", True),
    "pcieReplay": ("nvidia", "DCGM_FI_DEV_PCIE_REPLAY_COUNTER", "PCIe replays", False),
    "temp":       ("nvidia", "DCGM_FI_DEV_GPU_TEMP", "Temperature °C", False),
    "power":      ("nvidia", "DCGM_FI_DEV_POWER_USAGE", "Power W", False),
    "util":       ("nvidia", "DCGM_FI_DEV_GPU_UTIL", "Utilization %", False),
    "amdEccUnc":  ("amd", "gpu_ecc_uncorrect_total", "ECC uncorrectable", True),
    "amdEccCor":  ("amd", "gpu_ecc_correct_total", "ECC corrected", False),
    "amdTemp":    ("amd", "gpu_junction_temperature", "Temperature °C", False),
    "amdPower":   ("amd", "gpu_power_usage", "Power W", False),
    "amdUtil":    ("amd", "gpu_gfx_activity", "Utilization %", False),
}
NODE_LABELS = ("Hostname", "hostname", "node", "kubernetes_node", "exported_node", "nodename", "NodeName")
STANDARD_CONDITIONS = {"Ready", "MemoryPressure", "DiskPressure", "PIDPressure", "NetworkUnavailable"}
POSITIVE_WORDS = ("healthy", "ready", "ok", "available")


def node_conditions(node: dict) -> list[dict]:
    """Non-standard node conditions (set by NVSentinel, Node Problem Detector, vendor tools)."""
    out = []
    for c in (node.get("status") or {}).get("conditions") or []:
        t = c.get("type", "")
        if t in STANDARD_CONDITIONS or t.startswith("DualKey"):   # our own signal is not a vendor verdict
            continue
        positive = any(w in t.lower() for w in POSITIVE_WORDS)
        problem = (c.get("status") == "True") != positive
        out.append({"type": t, "status": c.get("status"), "reason": c.get("reason", ""),
                    "message": (c.get("message") or "")[:200], "problem": problem})
    return out


def match_node(labels: dict, nodes: list[dict], ip_to_node: dict) -> str | None:
    names = {n["name"]: n["name"] for n in nodes}
    names.update({n["short"]: n["name"] for n in nodes})
    for key in NODE_LABELS:
        v = labels.get(key)
        if v and (v in names or v.split(".")[0] in names):
            return names.get(v) or names[v.split(".")[0]]
    inst = (labels.get("instance") or "").rsplit(":", 1)[0]
    return ip_to_node.get(inst)


def summarize(results: dict, nodes: list[dict], ip_to_node: dict, raw_nodes: list[dict] | None = None) -> dict:
    """results: key -> list of Prometheus vector samples. Returns node name -> vendor view."""
    by_node: dict[str, dict] = {}
    for n in nodes:
        by_node[n["name"]] = {"source": "", "gpus": {}, "conditions": [], "verdict": "unknown", "problems": []}
    for key, samples in results.items():
        vendor, _q, label, is_error = QUERIES[key]
        for s in samples or []:
            labels = s.get("metric") or {}
            node = match_node(labels, nodes, ip_to_node)
            if not node or node not in by_node:
                continue
            try:
                val = float(s["value"][1])
            except (KeyError, ValueError, TypeError, IndexError):
                continue
            v = by_node[node]
            v["source"] = "NVIDIA DCGM exporter" if vendor == "nvidia" else "AMD device metrics exporter"
            gpu = labels.get("gpu") or labels.get("gpu_id") or labels.get("card_id") or "0"
            g = v["gpus"].setdefault(gpu, {"gpu": gpu, "model": labels.get("modelName") or labels.get("card_model", ""),
                                            "uuid": labels.get("UUID") or labels.get("serial_number", "")})
            g[key] = val
            if is_error and val > 0:
                v["problems"].append(f"GPU {gpu}: {label} = {val:g}")
    for rn in raw_nodes or []:
        name = (rn.get("metadata") or {}).get("name")
        if name in by_node:
            conds = node_conditions(rn)
            by_node[name]["conditions"] = conds
            by_node[name]["problems"] += [f"node condition {c['type']}={c['status']}" + (f" ({c['reason']})" if c["reason"] else "")
                                          for c in conds if c["problem"]]
    for v in by_node.values():
        v["gpus"] = list(v["gpus"].values())
        if v["problems"]:
            v["verdict"] = "error"
        elif v["gpus"] or v["conditions"]:
            v["verdict"] = "ok"
    return by_node


class VendorReader:
    def __init__(self):
        self.cache: dict = {}
        self.at = 0.0
        self.status = "not checked yet"

    def _headers(self) -> dict:
        try:
            with open(os.path.join(SA_DIR, "token")) as f:
                return {"Authorization": f"Bearer {f.read().strip()}"}
        except OSError:
            return {}

    def _verify(self):
        ca = os.path.join(SA_DIR, "service-ca.crt")
        return ca if os.path.exists(ca) else True

    async def query_all(self) -> dict:
        import httpx
        out, errors = {}, []
        async with httpx.AsyncClient(verify=self._verify(), headers=self._headers(), timeout=5) as c:
            for key, (_v, q, _l, _e) in QUERIES.items():
                try:
                    r = await c.get(f"{PROM_URL}/api/v1/query", params={"query": q})
                    if r.status_code in (401, 403):
                        raise PermissionError(f"HTTP {r.status_code}: grant cluster-monitoring-view to the dualkey-ui service account")
                    r.raise_for_status()
                    out[key] = r.json().get("data", {}).get("result", [])
                except PermissionError:
                    raise
                except Exception as e:  # one missing metric is fine
                    errors.append(f"{key}: {e}")
        if errors and not any(out.values()):
            raise RuntimeError("; ".join(errors[:2]))
        return out

    async def read(self, nodes: list[dict], raw_nodes: list[dict], ip_to_node: dict) -> dict:
        if time.time() - self.at >= REFRESH_SECONDS:
            try:
                results = await self.query_all()
                found = sum(len(v) for v in results.values())
                self.status = "connected" if found else "connected, but no NVIDIA or AMD GPU metrics found"
            except Exception as e:
                results = {}
                self.status = f"not connected: {e}"
                log.info("vendor metrics unavailable: %s", e)
            self.cache = results
            self.at = time.time()
        return {"status": self.status, "nodes": summarize(self.cache, nodes, ip_to_node, raw_nodes)}


def ip_to_node(v1, nodes: list[dict]) -> dict:
    """Metric 'instance' IPs -> node: node addresses plus vendor exporter pod IPs."""
    out = {}
    for n in nodes:
        for a in (n.get("status") or {}).get("addresses") or []:
            out[a.get("address")] = (n.get("metadata") or {}).get("name")
    for selector in ("app=nvidia-dcgm-exporter", "app.kubernetes.io/name=metrics-exporter", "app=metrics-exporter"):
        try:
            for p in v1.list_pod_for_all_namespaces(label_selector=selector).items:
                if p.status.pod_ip and p.spec.node_name:
                    out[p.status.pod_ip] = p.spec.node_name
        except Exception as e:
            log.debug("cannot list exporter pods (%s): %s", selector, e)
    return out


def snapshot(node_name: str, raw_nodes: list[dict], v1) -> dict:
    """One-off read of what the vendor tools say about one node (for incident evidence)."""
    import asyncio
    simple = [{"name": (n.get("metadata") or {}).get("name", ""),
               "short": (n.get("metadata") or {}).get("name", "").split(".")[0]} for n in raw_nodes]
    reader = VendorReader()
    try:
        out = asyncio.run(reader.read(simple, raw_nodes, ip_to_node(v1, raw_nodes)))
    except Exception as e:
        return {"status": f"not connected: {e}", "verdict": "unknown"}
    node = out["nodes"].get(node_name) or {}
    return {"status": out["status"], "takenAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **node}
