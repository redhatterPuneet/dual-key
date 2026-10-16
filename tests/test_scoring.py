import random

from dualkey.scoring import (FleetScorer, Settings, compute_gap, HEALTHY, LEARNING,
                             QUARANTINED, SUSPECT)


def make(names_vendors, **kw):
    s = Settings(baseline_samples=100, min_pair_checks=10, confirm_checks=5, **kw)
    fs = FleetScorer(settings=s)
    for n, v in names_vendors:
        fs.register_gpu(n, v)
    return fs


def healthy_gap(rng, va, vb):
    return abs(rng.gauss(0.03 if va != vb else 0.01, 0.005))


def run(fs, rng, n, faulty=None):
    names = list(fs.gpus)
    for _ in range(n):
        a, b = rng.sample(names, 2)
        g = healthy_gap(rng, fs.gpus[a].vendor, fs.gpus[b].vendor)
        if faulty in (a, b):
            g += 0.5
        fs.observe(a, b, g)


def test_compute_gap_basic():
    assert compute_gap([-0.1, -0.2], [-0.1, -0.4]) == (0 + 0.2) / 2
    assert compute_gap([None, -1.0], [-5.0, -1.5]) == 0.5
    assert compute_gap([], []) is None
    # missing checker value counts as very unlikely
    assert compute_gap([-0.1], [None]) > 10


def test_learning_then_healthy():
    rng = random.Random(1)
    fs = make([("nv-a", "nvidia"), ("nv-b", "nvidia"), ("amd-1", "amd")])
    run(fs, rng, 50)
    assert all(g.state == LEARNING for g in fs.gpus.values())
    run(fs, rng, 600)
    assert fs.learning_complete()
    assert all(g.state == HEALTHY for g in fs.gpus.values()), {n: g.state for n, g in fs.gpus.items()}


def test_faulty_nvidia_is_quarantined_and_others_stay_healthy():
    rng = random.Random(2)
    fs = make([("nv-a", "nvidia"), ("nv-b", "nvidia"), ("amd-1", "amd")])
    run(fs, rng, 600)
    run(fs, rng, 400, faulty="nv-b")
    assert fs.gpus["nv-b"].state == QUARANTINED
    assert fs.gpus["nv-a"].state == HEALTHY
    assert fs.gpus["amd-1"].state == HEALTHY


def test_faulty_amd_checker_is_caught_not_the_nvidia_it_checks():
    rng = random.Random(3)
    fs = make([("nv-a", "nvidia"), ("nv-b", "nvidia"), ("amd-1", "amd")])
    run(fs, rng, 600)
    run(fs, rng, 400, faulty="amd-1")
    assert fs.gpus["amd-1"].state == QUARANTINED
    assert fs.gpus["nv-a"].state == HEALTHY
    assert fs.gpus["nv-b"].state == HEALTHY


def test_release_resets_gpu():
    rng = random.Random(4)
    fs = make([("nv-a", "nvidia"), ("nv-b", "nvidia"), ("amd-1", "amd")])
    run(fs, rng, 600)
    run(fs, rng, 400, faulty="nv-b")
    assert fs.gpus["nv-b"].state == QUARANTINED
    fs.release("nv-b")
    assert fs.gpus["nv-b"].state == HEALTHY
    run(fs, rng, 400)
    assert fs.gpus["nv-b"].state == HEALTHY


def test_bigger_fleet_with_quantile():
    rng = random.Random(5)
    gpus = [(f"nv-{i}", "nvidia") for i in range(8)] + [(f"amd-{i}", "amd") for i in range(4)]
    fs = make(gpus, partner_quantile=0.2)
    run(fs, rng, 3000)
    run(fs, rng, 6000, faulty="nv-3")
    states = {n: g.state for n, g in fs.gpus.items()}
    assert states["nv-3"] == QUARANTINED
    assert all(s == HEALTHY for n, s in states.items() if n != "nv-3"), states


def test_roundtrip_persistence():
    rng = random.Random(6)
    fs = make([("nv-a", "nvidia"), ("nv-b", "nvidia"), ("amd-1", "amd")])
    run(fs, rng, 700)
    fs2 = FleetScorer.from_dict(fs.to_dict())
    assert fs2.learning_complete()
    assert fs2.gpus["nv-a"].state == fs.gpus["nv-a"].state
