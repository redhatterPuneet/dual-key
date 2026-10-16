"""Dual-Key Detector (step 7 of the call flow).

Keeps a separate scorer for every comparison group (each model has its own
"normal"), a running score for every member, and moves members through
Learning -> Healthy -> Suspect -> Quarantined. It publishes states to Redis
(read by the Gateway, Prober and Verifier), to GPUHealthRecord / DualKeyCluster
status in OpenShift (read by the operator and the dashboard), and as metrics.

Run: python -m dualkey.detector
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time

from prometheus_client import Counter, Gauge, start_http_server

from . import config as dkconfig
from .config import HASH_STATE, KEY_SCORER, STREAM_RESULTS
from .scoring import FleetScorer, Settings, QUARANTINED

log = logging.getLogger("dualkey.detector")
GROUP = "detectors"

SCORE = Gauge("dualkey_gpu_score", "Running score (lowest partner evidence)", ["group", "gpu", "vendor"])
STATE = Gauge("dualkey_gpu_state", "1 for the member's current state", ["group", "gpu", "state"])
PARTNERS = Gauge("dualkey_gpu_partners", "Partners with enough checks", ["group", "gpu"])
SUSPECT = Gauge("dualkey_suspect", "1 when a member is Suspect or Quarantined", ["group", "gpu", "node"])
RESULTS = Counter("dualkey_detector_results_total", "Gap results processed", ["group"])
TRANSITIONS = Counter("dualkey_detector_transitions_total", "State changes", ["group", "gpu", "to"])

ALL_STATES = ["Learning", "Healthy", "Suspect", "Quarantined"]


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class NullStatus:
    """Used outside OpenShift (tests, local runs)."""
    def __init__(self):
        self.gpus: dict = {}
        self.cluster: dict = {}
        self.releases: list = []

    def update_gpu(self, name, status):
        self.gpus[name] = status

    def update_cluster(self, status):
        self.cluster = status

    def pending_releases(self):
        out, self.releases = self.releases, []
        return out

    def ack_release(self, name):
        pass


class K8sStatus:
    """Writes status to GPUHealthRecord / DualKeyCluster custom resources."""
    GROUP, VERSION = "dualkey.io", "v1alpha1"

    def __init__(self, namespace: str, cluster_name: str):
        from kubernetes import client, config as kcfg
        kcfg.load_incluster_config()
        self.api = client.CustomObjectsApi()
        self.ns = namespace
        self.cluster_name = cluster_name

    def update_gpu(self, name, status):
        try:
            self.api.patch_namespaced_custom_object_status(
                self.GROUP, self.VERSION, self.ns, "gpuhealthrecords", name, {"status": status})
        except Exception as e:
            log.debug("status patch for %s failed: %s", name, e)

    def update_cluster(self, status):
        try:
            self.api.patch_namespaced_custom_object_status(
                self.GROUP, self.VERSION, self.ns, "dualkeyclusters", self.cluster_name,
                {"status": {"detection": status}})
        except Exception as e:
            log.debug("cluster status patch failed: %s", e)

    def pending_releases(self):
        try:
            items = self.api.list_namespaced_custom_object(
                self.GROUP, self.VERSION, self.ns, "gpuhealthrecords")["items"]
        except Exception:
            return []
        return [i["metadata"]["name"] for i in items if i.get("spec", {}).get("release")]

    def ack_release(self, name):
        try:
            self.api.patch_namespaced_custom_object(
                self.GROUP, self.VERSION, self.ns, "gpuhealthrecords", name,
                {"spec": {"release": False}})
        except Exception as e:
            log.warning("could not reset release flag on %s: %s", name, e)


class Detector:
    def __init__(self, cfg: dkconfig.Config, redis, status=None, consumer: str | None = None):
        self.cfg = cfg
        self.redis = redis
        self.status = status or NullStatus()
        self.consumer = consumer or socket.gethostname()
        self.settings = Settings.from_dict(cfg.detector)
        self.scorers: dict[str, FleetScorer] = {}
        self.builds: dict[str, dict] = {}     # member -> build it was scored with
        self.rebuilt: list[str] = []          # members whose evidence was reset this run
        self._dirty = False

    def scorer(self, group: str) -> FleetScorer | None:
        return self.scorers.get(group)

    async def setup(self) -> None:
        saved = await self.redis.get(KEY_SCORER)
        if saved:
            data = json.loads(_s(saved))
            self.scorers = {g: FleetScorer.from_dict(d, settings=self.settings)
                            for g, d in data.get("groups", {}).items()}
            self.builds = data.get("builds", {})
            log.info("restored detector state for %d groups", len(self.scorers))
        # keep only current groups and members
        current = {g.name for g in self.cfg.groups}
        for name in [g for g in self.scorers if g not in current]:
            del self.scorers[name]
        for g in self.cfg.groups:
            sc = self.scorers.setdefault(g.name, FleetScorer(settings=self.settings))
            for m in g.members:
                sc.register_gpu(m.name, m.vendor)
                self._check_build(sc, m)
            for name in [n for n in sc.gpus if not g.member(n)]:
                del sc.gpus[name]
        try:
            await self.redis.xgroup_create(STREAM_RESULTS, GROUP, id="0", mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                raise
        await self.publish()

    def _check_build(self, sc: FleetScorer, m) -> None:
        """A member whose vLLM image, version or driver changed starts its evidence over:
        a software change also moves the numbers, and must not look like a GPU fault."""
        new = {k: v for k, v in (m.build or {}).items() if v}
        old = self.builds.get(m.name)
        if old is not None and new and old != new:
            changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            log.info("member %s build changed (%s); restarting its evidence", m.name, ", ".join(changed))
            if sc.gpus[m.name].state != QUARANTINED:
                sc.release(m.name)
            self.rebuilt.append(m.name)
            self._dirty = True
        if new:
            self.builds[m.name] = new

    def handle(self, result: dict):
        sc = self.scorers.get(result.get("group", "manual"))
        if sc is None or result["primary"] not in sc.gpus or result["checker"] not in sc.gpus:
            return []
        RESULTS.labels(result.get("group", "manual")).inc()
        self._dirty = True
        trans = sc.observe(result["primary"], result["checker"], float(result["gap"]))
        for t in trans:
            TRANSITIONS.labels(result.get("group", "manual"), t.gpu, t.new).inc()
            log.info("[%s] %s: %s -> %s (score %s)", result.get("group"), t.gpu, t.old, t.new,
                     None if t.score is None else round(t.score, 2))
        return trans

    def member_states(self) -> dict[str, str]:
        return {n: g.state for sc in self.scorers.values() for n, g in sc.gpus.items()}

    async def publish(self) -> None:
        states = self.member_states()
        if states:
            await self.redis.hset(HASH_STATE, mapping=states)
        groups_status = {}
        for gname, sc in self.scorers.items():
            group = self.cfg.group(gname)
            for name, g in sc.gpus.items():
                m = group.member(name) if group else None
                node = m.node if m else ""
                if g.score is not None:
                    SCORE.labels(gname, name, g.vendor).set(g.score)
                PARTNERS.labels(gname, name).set(g.partners)
                for st in ALL_STATES:
                    STATE.labels(gname, name, st).set(1 if g.state == st else 0)
                SUSPECT.labels(gname, name, node).set(1 if g.state in ("Suspect", QUARANTINED) else 0)
                self.status.update_gpu(name, {
                    "state": g.state, "group": gname,
                    "score": None if g.score is None else round(g.score, 3),
                    "partners": g.partners, "checks": g.checks, "lastUpdate": _now(),
                    "build": (m.build if m else {}) or {}})
            groups_status[gname] = {
                "phase": "Detecting" if sc.learning_complete() else "Learning",
                "baselines": sc.baseline_status()}
        self.status.update_cluster({"groups": groups_status, "lastUpdate": _now()})

    async def save(self) -> None:
        data = {"groups": {g: sc.to_dict() for g, sc in self.scorers.items()}, "builds": self.builds}
        await self.redis.set(KEY_SCORER, json.dumps(data))
        self._dirty = False

    async def process_releases(self) -> None:
        for name in self.status.pending_releases():
            for sc in self.scorers.values():
                if name in sc.gpus:
                    log.info("member %s released by admin", name)
                    sc.release(name)
                    self._dirty = True
            self.status.ack_release(name)

    async def run_once(self, block_ms: int = 2000, count: int = 200) -> int:
        resp = await self.redis.xreadgroup(GROUP, self.consumer, {STREAM_RESULTS: ">"},
                                           count=count, block=block_ms)
        n = 0
        for _stream, msgs in resp or []:
            for msg_id, fields in msgs:
                try:
                    self.handle(json.loads(_s(fields.get(b"data", fields.get("data")))))
                except Exception:
                    log.exception("bad result %s", msg_id)
                await self.redis.xack(STREAM_RESULTS, GROUP, msg_id)
                n += 1
        return n

    async def run_forever(self) -> None:
        await self.setup()
        log.info("detector started for %d groups", len(self.scorers))
        last = 0.0
        while True:
            await self.run_once()
            if time.time() - last > 5:
                await self.process_releases()
                await self.publish()
                if self._dirty:
                    await self.save()
                last = time.time()


async def _main() -> None:
    import redis.asyncio as aioredis
    cfg = dkconfig.load()
    start_http_server(int(os.environ.get("DK_METRICS_PORT", "9100")))
    status = K8sStatus(cfg.namespace, cfg.cluster_name) if os.environ.get("KUBERNETES_SERVICE_HOST") else NullStatus()
    await Detector(cfg, aioredis.from_url(cfg.redis_url), status).run_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(_main())
