from dualkey import ui
from dualkey.scoring import FleetScorer, Settings


def test_gpu_nodes_picks_gpu_nodes_only():
    nodes = [
        {"metadata": {"name": "gpu-1.x", "labels": {"nvidia.com/gpu.product": "Tesla-T4",
                                                    "node.kubernetes.io/instance-type": "g4dn.xlarge"}},
         "status": {"allocatable": {"nvidia.com/gpu": "1"}},
         "spec": {"taints": [{"key": "dualkey.io/suspect", "effect": "NoSchedule"}]}},
        {"metadata": {"name": "cpu-1", "labels": {}}, "status": {"allocatable": {"cpu": "4"}}},
        {"metadata": {"name": "amd-1", "labels": {}}, "status": {"allocatable": {"amd.com/gpu": "2"}}},
    ]
    out = ui.gpu_nodes(nodes)
    assert [n["name"] for n in out] == ["amd-1", "gpu-1.x"]
    t4 = out[1]
    assert t4["vendor"] == "nvidia" and t4["product"] == "Tesla-T4" and t4["short"] == "gpu-1" and t4["tainted"]
    assert out[0]["vendor"] == "amd" and out[0]["gpus"] == 2


def scorer_dump():
    fs = FleetScorer(Settings(baseline_samples=5, min_pair_checks=1))
    for g in ("a", "b", "c"):
        fs.register_gpu(g, "nvidia")
    for i in range(30):
        fs.observe("a", "b", 0.01 + (i % 3) * 0.001)
        fs.observe("b", "c", 0.01 + (i % 2) * 0.001)
        fs.observe("a", "c", 0.011)
    return {"groups": {"g1": fs.to_dict()}}


def test_scorer_view_and_annotations():
    view = ui.scorer_view(scorer_dump())
    g = view["g1"]
    assert g["baselines"]["nvidia|nvidia"]["ready"] and g["baselines"]["nvidia|nvidia"]["samples"] == 5
    assert set(g["pairs"]) == {"a|b", "b|c", "a|c"}
    assert g["members"]["a"]["vendor"] == "nvidia"
    recent = ui.annotate_recent([{"ts": 1, "group": "g1", "primary": "a", "checker": "b", "gap": 0.5}], view)
    assert recent[0]["z"] is not None and recent[0]["z"] > 5


def test_checks_summary_counts_last_minute():
    now = 1000.0
    res = [{"ts": 990, "source": "probe"}, {"ts": 980, "source": "gateway"}, {"ts": 900, "source": "probe"}]
    s = ui.checks_summary(res, now)
    assert s == {"lastMinute": 2, "probeLastMinute": 1, "gatewayLastMinute": 1, "last": res[0]}


def test_build_state_shape_and_no_raw_baselines():
    cluster = {"metadata": {"name": "fleet"}, "spec": {"detector": {"suspectZ": 3, "quarantineZ": 6}},
               "status": {"phase": "Checking"}}
    live = {"scorer": ui.scorer_view(scorer_dump()), "pipeline": {"lastMinute": 3}, "recent": []}
    s = ui.build_state(cluster, [], [], [], live, {"a": [[1, 0.2]]})
    assert s["cluster"]["thresholds"] == {"suspect": 3, "quarantine": 6}
    assert "rawBaselines" not in s["scorer"]["g1"]
    assert s["pipeline"]["lastMinute"] == 3 and s["history"]["a"] == [[1, 0.2]]


def test_page_has_three_views():
    html = ui.page()
    for text in ("What Dual-Key found", "What each part is doing", "What the Detector sees"):
        assert text in html
