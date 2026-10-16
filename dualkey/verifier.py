"""Dual-Key Verifier (steps 4-6 of the call flow).

4. Takes a sampled answer off the queue and picks ONE checker from the same
   comparison group, at random but following the rules: not the member that
   answered, not quarantined, weighted towards the other vendor (70% by default),
   favouring the least-checked pair. (Lab option check_all: every other member.)
5. Asks the checker to score the exact same tokens and computes the gap.
6. Sends the gap, tagged with group and both member names, to the Detector.

Run: python -m dualkey.verifier
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import socket
import time

import httpx
from prometheus_client import Counter, Histogram, start_http_server

from . import config as dkconfig
from .config import HASH_PAIRCOUNT, HASH_STATE, STREAM_RESULTS, STREAM_VERIFY
from .scoring import QUARANTINED, compute_gap, gpu_pair_key
from .vllm_client import VllmError, rescore

log = logging.getLogger("dualkey.verifier")
GROUP = "verifiers"

CHECKS = Counter("dualkey_verifier_checks_total", "Checks done", ["group", "primary", "checker"])
FAILS = Counter("dualkey_verifier_failures_total", "Checks that failed", ["group", "checker"])
RESCORE_S = Histogram("dualkey_verifier_rescore_seconds", "Time the checker GPU spent re-scoring one answer",
                      ["group", "checker"], buckets=(0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30))
QUEUE_S = Histogram("dualkey_verifier_queue_age_seconds", "How long an answer waited in the queue",
                    ["group"], buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 300))
GAP = Histogram("dualkey_verifier_gap", "Gap per check", ["group", "vendor_pair"],
                buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10))


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def pair_count_key(group: str, a: str, b: str) -> str:
    return f"{group}:{gpu_pair_key(a, b)}"


def choose_checkers(cfg: dkconfig.Config, group: dkconfig.Group, primary: str,
                    states: dict[str, str], paircounts: dict[str, int],
                    rng: random.Random) -> list[dkconfig.Member]:
    """Apply the checker-selection rules inside one group."""
    p = group.member(primary)
    cands = [m for m in group.members if m.name != primary and states.get(m.name) != QUARANTINED]
    if not cands or p is None:
        return []
    if cfg.check_all:
        return cands
    other = [m for m in cands if m.vendor != p.vendor]
    same = [m for m in cands if m.vendor == p.vendor]
    if other and same:
        pool = other if rng.random() < cfg.cross_vendor_weight else same
    else:
        pool = other or same
    counts = [paircounts.get(pair_count_key(group.name, primary, m.name), 0) for m in pool]
    low = min(counts)
    return [rng.choice([m for m, c in zip(pool, counts) if c == low])]


class Verifier:
    def __init__(self, cfg: dkconfig.Config, redis, http: httpx.AsyncClient,
                 consumer: str | None = None, seed: int | None = None):
        self.cfg = cfg
        self.redis = redis
        self.http = http
        self.consumer = consumer or socket.gethostname()
        self.rng = random.Random(seed)
        self.sem = asyncio.Semaphore(cfg.verifier_concurrency)

    async def setup(self) -> None:
        try:
            await self.redis.xgroup_create(STREAM_VERIFY, GROUP, id="0", mkstream=True)
        except Exception as e:  # group already exists
            if "BUSYGROUP" not in str(e):
                raise

    async def _snapshot(self) -> tuple[dict, dict]:
        st = {_s(k): _s(v) for k, v in (await self.redis.hgetall(HASH_STATE)).items()}
        pc = {_s(k): int(_s(v)) for k, v in (await self.redis.hgetall(HASH_PAIRCOUNT)).items()}
        return st, pc

    async def check_one(self, item: dict) -> int:
        group = self.cfg.group(item.get("group", "manual")) or self.cfg.default_group()
        if group is None or group.member(item["primary"]) is None:
            return 0  # group changed since the answer was sampled
        states, counts = await self._snapshot()
        p = group.member(item["primary"])
        queue_s = max(0.0, time.time() - float(item.get("ts") or time.time()))
        QUEUE_S.labels(group.name).observe(queue_s)
        done = 0
        for chk in choose_checkers(self.cfg, group, item["primary"], states, counts, self.rng):
            t0 = time.perf_counter()
            try:
                lps = await rescore(self.http, chk.endpoint, dkconfig.api_key_for(chk),
                                    chk.model, item["prompt_ids"], item["output_ids"])
                rescore_s = time.perf_counter() - t0
                RESCORE_S.labels(group.name, chk.name).observe(rescore_s)
            except (VllmError, httpx.HTTPError) as e:
                FAILS.labels(group.name, chk.name).inc()
                log.warning("check on %s failed: %s", chk.name, e)
                continue
            gap = compute_gap(item["primary_logprobs"], lps)
            if gap is None:
                continue
            result = {"request_id": item["request_id"], "group": group.name,
                      "primary": item["primary"], "checker": chk.name, "gap": gap,
                      "ts": time.time(), "tokens": len(item["output_ids"]),
                      "scored_tokens": len(item["prompt_ids"]) + len(item["output_ids"]),
                      "rescore_s": round(rescore_s, 4), "queue_s": round(queue_s, 3),
                      "answer_s": item.get("answer_s"),
                      "source": item.get("source", "")}
            await self.redis.xadd(STREAM_RESULTS, {"data": json.dumps(result)}, maxlen=200000)
            await self.redis.hincrby(HASH_PAIRCOUNT, pair_count_key(group.name, item["primary"], chk.name), 1)
            CHECKS.labels(group.name, item["primary"], chk.name).inc()
            GAP.labels(group.name, f"{p.vendor}|{chk.vendor}").observe(gap)
            done += 1
        return done

    async def _handle(self, msg_id, fields) -> None:
        async with self.sem:
            try:
                item = json.loads(_s(fields.get(b"data", fields.get("data"))))
                await self.check_one(item)
            except Exception:
                log.exception("failed to process %s", msg_id)
            finally:
                await self.redis.xack(STREAM_VERIFY, GROUP, msg_id)

    async def run_once(self, block_ms: int = 5000, count: int = 32) -> int:
        resp = await self.redis.xreadgroup(GROUP, self.consumer, {STREAM_VERIFY: ">"},
                                           count=count, block=block_ms)
        tasks = [asyncio.create_task(self._handle(mid, f))
                 for _stream, msgs in resp or [] for mid, f in msgs]
        if tasks:
            await asyncio.gather(*tasks)
        return len(tasks)

    async def run_forever(self) -> None:
        await self.setup()
        log.info("verifier %s started (%d groups, check_all=%s)",
                 self.consumer, len(self.cfg.groups), self.cfg.check_all)
        while True:
            await self.run_once()


async def _main() -> None:
    import redis.asyncio as aioredis
    cfg = dkconfig.load()
    start_http_server(int(os.environ.get("DK_METRICS_PORT", "9100")))
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as http:
        await Verifier(cfg, aioredis.from_url(cfg.redis_url), http).run_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(_main())
