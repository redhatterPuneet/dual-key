from dualkey import vendor as V

NODES = [{"name": "ip-10-0-1-116.us-east-2.compute.internal", "short": "ip-10-0-1-116"},
         {"name": "ip-10-0-15-24.us-east-2.compute.internal", "short": "ip-10-0-15-24"}]


def sample(value, **labels):
    return {"metric": labels, "value": [0, str(value)]}


def test_matches_by_hostname_and_instance_ip():
    results = {
        "xid": [sample(0, Hostname="ip-10-0-1-116", gpu="0", modelName="Tesla T4"),
                sample(79, instance="10.130.2.5:9400", gpu="0")],
        "temp": [sample(61, Hostname="ip-10-0-1-116.us-east-2.compute.internal", gpu="0")],
    }
    out = V.summarize(results, NODES, {"10.130.2.5": NODES[1]["name"]})
    t4, l4 = out[NODES[0]["name"]], out[NODES[1]["name"]]
    assert t4["verdict"] == "ok" and t4["gpus"][0]["temp"] == 61 and t4["source"] == "NVIDIA DCGM exporter"
    assert l4["verdict"] == "error" and "XID error" in l4["problems"][0]


def test_no_metrics_is_unknown_and_conditions_count():
    raw = [{"metadata": {"name": NODES[0]["name"]},
            "status": {"conditions": [{"type": "Ready", "status": "True"},
                                      {"type": "GpuXidError", "status": "True", "reason": "XID79"},
                                      {"type": "GpuHealthy", "status": "True"}]}}]
    out = V.summarize({}, NODES, {}, raw)
    assert out[NODES[1]["name"]]["verdict"] == "unknown"
    n = out[NODES[0]["name"]]
    assert [c["type"] for c in n["conditions"]] == ["GpuXidError", "GpuHealthy"]
    assert n["verdict"] == "error" and "GpuXidError" in n["problems"][0]


def test_healthy_positive_condition_is_not_a_problem():
    raw = [{"metadata": {"name": NODES[0]["name"]}, "status": {"conditions": [{"type": "GpuHealthy", "status": "True"}]}}]
    assert V.summarize({}, NODES, {}, raw)[NODES[0]["name"]]["verdict"] == "ok"
