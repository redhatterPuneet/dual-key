"""Dual-Key Prober: keeps every comparison group checked without touching app traffic.

For each group it sends a steady trickle of harmless test prompts (default one
per second per group), round robin across the group's members, using each
member's own model. Every probe answer goes to the Verifier, exactly like a
sampled live answer would.

Run: python -m dualkey.prober
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import os
import random
import time

import httpx
from prometheus_client import Counter, start_http_server

from . import config as dkconfig
from .config import HASH_STATE
from .loadgen import load_prompts
from .sampling import submit
from .scoring import QUARANTINED
from .vllm_client import VllmError, answer

log = logging.getLogger("dualkey.prober")
PROBES = Counter("dualkey_probes_total", "Probe prompts sent", ["group", "member", "result"])


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


class Prober:
    def __init__(self, cfg: dkconfig.Config, redis, http: httpx.AsyncClient,
                 prompts: list[str] | None = None, seed: int | None = None):
        self.cfg = cfg
        self.redis = redis
        self.http = http
        self.prompts = prompts or load_prompts()
        self.rng = random.Random(seed)
        self._rr = {g.name: itertools.cycle([m.name for m in g.members]) for g in cfg.groups}

    async def _states(self) -> dict[str, str]:
        return {_s(k): _s(v) for k, v in (await self.redis.hgetall(HASH_STATE)).items()}

    async def probe_once(self, group: dkconfig.Group, states: dict | None = None) -> bool:
        states = states if states is not None else await self._states()
        for _ in range(len(group.members)):
            member = group.member(next(self._rr[group.name]))
            if states.get(member.name) != QUARANTINED:
                break
        else:
            return False
        # greedy and seeded: the only randomness left is the hardware's arithmetic
        body = {"model": member.model, "prompt": self.rng.choice(self.prompts),
                "max_tokens": self.cfg.probe_max_tokens, "temperature": 0, "seed": 0}
        t0 = time.perf_counter()
        try:
            _data, rec = await answer(self.http, member.endpoint, dkconfig.api_key_for(member), body)
            rec["answer_s"] = round(time.perf_counter() - t0, 4)
        except (VllmError, httpx.HTTPError) as e:
            PROBES.labels(group.name, member.name, "error").inc()
            log.warning("probe to %s failed: %s", member.name, e)
            return False
        ok = await submit(self.redis, group.name, member.name, rec, "probe")
        PROBES.labels(group.name, member.name, "ok" if ok else "unusable").inc()
        return ok

    async def run_group(self, group: dkconfig.Group) -> None:
        interval = 1.0 / max(self.cfg.probe_rate, 0.01)
        while True:
            await self.probe_once(group)
            await asyncio.sleep(interval)

    async def run_forever(self) -> None:
        if not self.cfg.groups:
            log.info("no comparison groups yet; prober idle")
            while True:
                await asyncio.sleep(60)
        log.info("probing %d groups at %.2f prompts/s each", len(self.cfg.groups), self.cfg.probe_rate)
        await asyncio.gather(*(self.run_group(g) for g in self.cfg.groups if len(g.members) >= 2))


async def _main() -> None:
    import redis.asyncio as aioredis
    cfg = dkconfig.load()
    start_http_server(int(os.environ.get("DK_METRICS_PORT", "9100")))
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as http:
        await Prober(cfg, aioredis.from_url(cfg.redis_url), http).run_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(_main())
