"""Dual-Key Gateway (steps 1-3 of the call flow) for apps that opt in.

  POST /v1/completions                    -> the default group (lab / manual mode)
  POST /groups/{group}/v1/completions     -> a specific comparison group

1. Receives a completion request.
2. Picks a member of the group (round robin, skipping quarantined ones), gets the
   answer plus per-token confidence, and returns it immediately.
3. Samples a copy (5% by default, more for Suspect members) for the Verifier.

Run: uvicorn dualkey.gateway:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import itertools
import logging
import random
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

from . import config as dkconfig
from .config import HASH_STATE
from .sampling import submit
from .scoring import QUARANTINED, SUSPECT
from .vllm_client import VllmError, answer

log = logging.getLogger("dualkey.gateway")

REQUESTS = Counter("dualkey_gateway_requests_total", "Requests answered", ["group", "member"])
SAMPLED = Counter("dualkey_gateway_sampled_total", "Requests sampled for checking", ["group", "member"])
ERRORS = Counter("dualkey_gateway_errors_total", "Model server errors", ["group", "member"])


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


class Router:
    """Round robin over a group's members, skipping quarantined ones."""

    def __init__(self, cfg: dkconfig.Config, redis):
        self.cfg = cfg
        self.redis = redis
        self.states: dict[str, str] = {}
        self._rr = {g.name: itertools.cycle([m.name for m in g.members]) for g in cfg.groups}
        self._last = 0.0

    async def refresh(self, force: bool = False) -> None:
        if not force and time.time() - self._last < 2.0:
            return
        raw = await self.redis.hgetall(HASH_STATE)
        self.states = {_s(k): _s(v) for k, v in raw.items()}
        self._last = time.time()

    def pick(self, group: dkconfig.Group) -> dkconfig.Member:
        for _ in range(len(group.members)):
            name = next(self._rr[group.name])
            if self.states.get(name) != QUARANTINED:
                return group.member(name)
        raise HTTPException(503, f"all members of group {group.name} are quarantined")

    def sample_rate(self, member: str) -> float:
        if self.states.get(member) == SUSPECT:
            return self.cfg.suspect_sample_rate
        return self.cfg.sample_rate


def create_app(cfg: dkconfig.Config | None = None, redis=None,
               http: httpx.AsyncClient | None = None) -> FastAPI:
    state: dict = {}

    async def _startup():
        c = cfg or dkconfig.load()
        r = redis
        if r is None:
            import redis.asyncio as aioredis
            r = aioredis.from_url(c.redis_url)
        state.update(cfg=c, redis=r, router=Router(c, r),
                     http=http or httpx.AsyncClient(timeout=httpx.Timeout(300.0)))
        log.info("gateway ready with %d groups", len(c.groups))

    @asynccontextmanager
    async def lifespan(_app):
        await _startup()
        yield
        if http is None and state.get("http"):
            await state["http"].aclose()

    app = FastAPI(title="Dual-Key Gateway", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    async def _complete(group: dkconfig.Group | None, request: Request):
        if group is None:
            raise HTTPException(404, "unknown group")
        body = await request.json()
        if body.get("stream"):
            raise HTTPException(400, "streaming is not supported by the Dual-Key gateway")
        router: Router = state["router"]
        await router.refresh()
        member = router.pick(group)
        body["model"] = member.model or body.get("model")
        try:
            data, rec = await answer(state["http"], member.endpoint, dkconfig.api_key_for(member), body)
        except (VllmError, httpx.HTTPError) as e:
            ERRORS.labels(group.name, member.name).inc()
            raise HTTPException(502, f"model server {member.name} failed: {e}")
        REQUESTS.labels(group.name, member.name).inc()
        forced = request.headers.get("x-dualkey", "").lower() in ("required", "sample")
        if forced or random.random() < router.sample_rate(member.name):
            if await submit(state["redis"], group.name, member.name, rec, "gateway"):
                SAMPLED.labels(group.name, member.name).inc()
        return JSONResponse(data, headers={"x-dualkey-gpu": member.name, "x-dualkey-group": group.name})

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _complete(state["cfg"].default_group(), request)

    @app.post("/groups/{group}/v1/completions")
    async def group_completions(group: str, request: Request):
        return await _complete(state["cfg"].group(group), request)

    @app.get("/groups")
    async def groups():
        c = state["cfg"]
        return [{"name": g.name, "model_id": g.model_id, "source": g.source,
                 "members": [m.name for m in g.members]} for g in c.groups]

    return app


app = create_app()
