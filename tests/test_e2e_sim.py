"""End-to-end simulations: real Gateway, Prober, Verifier and Detector code, fake GPUs.

1. Gateway path, one group of 3: learning -> nv-b turns faulty -> nv-b quarantined.
2. Prober path, two groups of 3 running different models: a fault in one member of
   group B quarantines only that member; group A is unaffected.
"""
import asyncio
import random

import fakeredis
import httpx
import pytest

from dualkey import config as dkconfig
from dualkey.config import HASH_STATE, Group, Member
from dualkey.detector import Detector, NullStatus
from dualkey.gateway import create_app, Router
from dualkey.prober import Prober
from dualkey.verifier import Verifier, choose_checkers
from tests.fake_vllm import FakeGpu, transport


def lab_group(name="lab", prefix=""):
    return Group(name=name, model_id="granite", source="manual", members=[
        Member(f"{prefix}nv-a", "nvidia", f"http://{prefix}nv-a", model="granite"),
        Member(f"{prefix}nv-b", "nvidia", f"http://{prefix}nv-b", model="granite"),
        Member(f"{prefix}amd-1", "amd", f"http://{prefix}amd-1", model="granite")])


def make_cfg(check_all: bool, groups=None) -> dkconfig.Config:
    return dkconfig.Config(
        groups=groups or [lab_group()], sample_rate=1.0, check_all=check_all,
        detector={"baseline_samples": 150, "min_pair_checks": 15, "confirm_checks": 5})


async def drain(verifier, detector):
    while await verifier.run_once(block_ms=1):
        pass
    while await detector.run_once(block_ms=1):
        pass
    await detector.publish()


async def drive_gateway(client, verifier, detector, n, rng):
    words = "explain the sky water cycle interest compound gpu model token answer".split()
    for _ in range(n):
        prompt = " ".join(rng.choice(words) for _ in range(rng.randint(4, 12)))
        r = await client.post("/v1/completions", json={"prompt": prompt, "max_tokens": 16})
        assert r.status_code in (200, 503), r.text
    await drain(verifier, detector)


def states_of(detector, group):
    return {n: g.state for n, g in detector.scorer(group).gpus.items()}


@pytest.mark.parametrize("check_all", [True, False])
def test_gateway_path_faulty_gpu_is_quarantined(check_all):
    asyncio.run(_gateway_scenario(check_all))


async def _gateway_scenario(check_all):
    rng = random.Random(11)
    redis = fakeredis.FakeAsyncRedis()
    gpus = {"nv-a": FakeGpu("nv-a", "nvidia", 1), "nv-b": FakeGpu("nv-b", "nvidia", 2),
            "amd-1": FakeGpu("amd-1", "amd", 3)}
    http = httpx.AsyncClient(transport=transport(gpus))
    cfg = make_cfg(check_all)
    app = create_app(cfg, redis, http)
    async with app.router.lifespan_context(app):
        status = NullStatus()
        verifier = Verifier(cfg, redis, http, consumer="v1", seed=5)
        await verifier.setup()
        detector = Detector(cfg, redis, status, consumer="d1")
        await detector.setup()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
            # single-checker mode sends only ~30% of NVIDIA answers to the other NVIDIA,
            # so the nvidia|nvidia baseline needs more traffic to fill
            n1 = 300 if check_all else 1100
            await drive_gateway(client, verifier, detector, n1, rng)
            assert detector.scorer("lab").learning_complete()
            assert set(states_of(detector, "lab").values()) == {"Healthy"}

            gpus["nv-b"].faulty = True
            await drive_gateway(client, verifier, detector, n1, rng)
            st = states_of(detector, "lab")
            assert st == {"nv-a": "Healthy", "nv-b": "Quarantined", "amd-1": "Healthy"}, st
            assert (await redis.hgetall(HASH_STATE))[b"nv-b"] == b"Quarantined"
            assert status.gpus["nv-b"]["state"] == "Quarantined"
            assert status.cluster["groups"]["lab"]["phase"] == "Detecting"

            router = Router(cfg, redis)
            await router.refresh(force=True)
            assert "nv-b" not in {router.pick(cfg.groups[0]).name for _ in range(30)}
    await http.aclose()


def test_prober_two_groups_only_faulty_member_quarantined():
    asyncio.run(_prober_scenario())


async def _prober_scenario():
    redis = fakeredis.FakeAsyncRedis()
    gpus = {}
    for prefix, seed in (("a-", 10), ("b-", 20)):
        gpus[f"{prefix}nv-a"] = FakeGpu(f"{prefix}nv-a", "nvidia", seed + 1)
        gpus[f"{prefix}nv-b"] = FakeGpu(f"{prefix}nv-b", "nvidia", seed + 2)
        gpus[f"{prefix}amd-1"] = FakeGpu(f"{prefix}amd-1", "amd", seed + 3)
    http = httpx.AsyncClient(transport=transport(gpus))
    cfg = make_cfg(True, groups=[lab_group("group-a", "a-"), lab_group("group-b", "b-")])
    prober = Prober(cfg, redis, http, prompts=["what is the water cycle", "explain compound interest",
                                                 "name three planets", "how does a gpu work"], seed=1)
    verifier = Verifier(cfg, redis, http, consumer="v1", seed=2)
    await verifier.setup()
    detector = Detector(cfg, redis, NullStatus(), consumer="d1")
    await detector.setup()

    async def rounds(n):
        for _ in range(n):
            for g in cfg.groups:
                await prober.probe_once(g)
        await drain(verifier, detector)

    await rounds(260)   # ~1/3 of checks are nvidia|nvidia; baseline needs 150 of them
    for g in ("group-a", "group-b"):
        assert detector.scorer(g).learning_complete()
        assert set(states_of(detector, g).values()) == {"Healthy"}

    gpus["b-amd-1"].faulty = True          # this time the AMD member goes bad
    await rounds(200)
    assert set(states_of(detector, "group-a").values()) == {"Healthy"}
    assert states_of(detector, "group-b") == {"b-nv-a": "Healthy", "b-nv-b": "Healthy",
                                              "b-amd-1": "Quarantined"}
    await http.aclose()


def test_checker_selection_rules():
    cfg = make_cfg(check_all=False)
    g = cfg.groups[0]
    rng = random.Random(0)
    picks = [choose_checkers(cfg, g, "nv-a", {}, {}, rng)[0].name for _ in range(2000)]
    assert "nv-a" not in picks
    assert 0.65 < picks.count("amd-1") / len(picks) < 0.75          # ~70% other vendor
    counts = {"lab:amd-1|nv-a": 5, "lab:amd-1|nv-b": 0}            # least-checked pair wins
    assert choose_checkers(cfg, g, "amd-1", {}, counts, rng)[0].name == "nv-b"
    assert choose_checkers(cfg, g, "nv-a", {"amd-1": "Quarantined"}, {}, rng)[0].name == "nv-b"


def test_legacy_flat_gpu_config_becomes_one_group():
    cfg = dkconfig.from_dict({"served_model_name": "granite", "gpus": [
        {"name": "nv-a", "vendor": "nvidia", "endpoint": "http://x"}]})
    assert cfg.default_group().name == "manual"
    assert cfg.default_group().members[0].model == "granite"
