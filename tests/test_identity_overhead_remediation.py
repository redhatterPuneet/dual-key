import asyncio
import json

from dualkey import discovery as D
from dualkey import overhead as OH
from dualkey import remediation as RM
from dualkey.config import Config, Group, Member
from dualkey.detector import Detector


# ------------------------------------------------------------------ identity

def server(key, tok="abc", vllm="0.11.0", rev="r1", ready=True, vendor="nvidia"):
    return D.Server(key=key, namespace="ns", pod=key, owner=key, node=f"n-{key}", vendor=vendor, gpus=1,
                    model_id="ibm/granite", served_name="granite", dtype="float16", quantization="none",
                    tensor_parallel=1, endpoint=f"http://{key}:8000", ready=ready, managed=False,
                    revision=rev, tokenizer_hash=tok, vllm_version=vllm)


def test_same_build_forms_one_group_with_identity():
    groups = D.form_groups([server("a"), server("b"), server("c")])
    assert len(groups) == 1 and groups[0].phase == "Ready"
    assert groups[0].identity["tokenizer"] == "abc" and groups[0].identity["vllm"] == "0.11.0"
    assert groups[0].split == []


def test_tokenizer_or_vllm_mismatch_splits_with_reason():
    groups = D.form_groups([server("a"), server("b"), server("c", tok="zzz"), server("d", vllm="0.10.1")])
    assert len(groups) == 3
    big = next(g for g in groups if len(g.members) == 2)
    assert any("tokenizer differs" in r for r in big.split) and any("vLLM version differs" in r for r in big.split)
    assert "kept apart" in big.message


def test_revision_mismatch_is_a_different_model():
    groups = D.form_groups([server("a"), server("b", rev="r2")])
    assert len(groups) == 2 and all(g.split == [] for g in groups)


def test_unprobed_servers_join_the_largest_variant():
    groups = D.form_groups([server("a"), server("b"), server("c", tok="", vllm="", ready=False)])
    assert len(groups) == 1 and len(groups[0].members) == 3


def test_identity_helpers():
    assert D.normalize_vllm_version("v0.9.2+rocm641") == "0.9.2"
    assert D.tokenizer_hash([1, 2, 3]) == D.tokenizer_hash([1, 2, 3]) != D.tokenizer_hash([1, 2, 4])
    node = {"metadata": {"labels": {"nvidia.com/cuda.driver.major": "580", "nvidia.com/cuda.driver.minor": "82",
                                    "nvidia.com/cuda.driver.rev": "07"}}}
    assert D.node_driver(node, "nvidia") == "580.82.07"
    pod = {"status": {"containerStatuses": [{"name": "vllm", "imageID": "docker.io/vllm/vllm-openai@sha256:abc"}]}}
    assert D.container_digest(pod, "vllm") == "sha256:abc"


def test_discovery_reads_revision_annotation_and_driver():
    pod = {"metadata": {"namespace": "apps", "name": "llm-1", "annotations": {D.REVISION_ANNOTATION: "deadbeef"},
                        "ownerReferences": [{"kind": "ReplicaSet", "name": "llm-5d9"}]},
           "spec": {"nodeName": "gpu-1", "containers": [{"name": "vllm", "image": "vllm/vllm-openai:v0.11.0",
                    "args": ["vllm", "serve", "ibm/granite", "--dtype", "float16"],
                    "resources": {"limits": {"nvidia.com/gpu": "1"}}}]},
           "status": {"phase": "Running", "podIP": "10.0.0.5", "conditions": [{"type": "Ready", "status": "True"}],
                      "containerStatuses": [{"name": "vllm", "imageID": "x@sha256:111"}]}}
    nodes = [{"metadata": {"name": "gpu-1", "labels": {"nvidia.com/cuda.driver-version.full": "580.82.07"}}}]
    servers, _ = D.discover_servers([pod], nodes=nodes)
    s = servers[0]
    assert s.revision == "deadbeef" and s.driver == "580.82.07" and s.image_digest == "sha256:111"
    D.apply_probes(servers, {s.key: {"ok": True, "version": "0.11.0+cu126", "canary_tokens": [5, 6]}})
    assert s.vllm_version == "0.11.0" and s.tokenizer_hash == D.tokenizer_hash([5, 6])


def test_detector_restarts_evidence_when_build_changes():
    class R:
        def __init__(self):
            self.kv = {}

        async def get(self, k):
            return self.kv.get(k)

        async def set(self, k, v):
            self.kv[k] = v

        async def hset(self, *a, **k):
            pass

        async def xgroup_create(self, *a, **k):
            pass

    def cfg(digest):
        ms = [Member(name=n, vendor="nvidia", endpoint="x", build={"imageDigest": digest if n == "a" else "d0"})
              for n in ("a", "b", "c")]
        return Config(groups=[Group(name="g", members=ms)], detector={"baseline_samples": 5, "min_pair_checks": 2})

    redis = R()
    d = Detector(cfg("d0"), redis)
    asyncio.run(d.setup())
    for _ in range(20):
        d.handle({"group": "g", "primary": "a", "checker": "b", "gap": 0.01})
        d.handle({"group": "g", "primary": "a", "checker": "c", "gap": 0.01})
        d.handle({"group": "g", "primary": "b", "checker": "c", "gap": 0.01})
    asyncio.run(d.save())
    assert any("a" in k for k in d.scorers["g"].pairs)
    d2 = Detector(cfg("d1"), redis)          # a's vLLM image changed
    asyncio.run(d2.setup())
    assert d2.rebuilt == ["a"]
    assert not any("a" in k.split("|") for k in d2.scorers["g"].pairs)
    assert "b|c" in d2.scorers["g"].pairs


# ------------------------------------------------------------------ overhead

def test_overhead_summary_and_cost():
    now = 1000.0
    results = []
    for i in range(10):
        results.append({"ts": now - i, "request_id": f"r{i}", "checker": "l4", "primary": "t4",
                        "rescore_s": 0.1, "queue_s": 0.5, "answer_s": 1.0, "scored_tokens": 100})
        results.append({"ts": now - i, "request_id": f"r{i}", "checker": "a10g", "primary": "t4",
                        "rescore_s": 0.2, "queue_s": 0.5, "answer_s": 1.0, "scored_tokens": 100})
    out = OH.summarize(results, now, {"l4": "node-l4", "a10g": "node-a10g"},
                       {"node-l4": "g6.xlarge", "node-a10g": "g5.2xlarge"}, None, 0.05)
    assert out["checks"] == 20 and out["checksPerSample"] == 2
    assert abs(out["rescoreP50"] - 0.15) < 1e-9 and abs(out["checkVsAnswer"] - 0.15) < 1e-9
    expected = (0.1 * 0.805 + 0.2 * 1.212) / 2 / 3600 * 1000
    assert abs(out["costPer1000Checks"] - round(expected, 4)) < 1e-4
    assert abs(out["costPer1000Requests"] - round(expected * 0.05 * 2, 4)) < 1e-4
    assert OH.price_for("x", "g6.xlarge", 2.5)[0] == 2.5
    assert OH.price_for("gpu-9.example", "", {"gpu-9": 3.0})[0] == 3.0
    assert OH.summarize([], now, {}, {})["checks"] == 0


# ------------------------------------------------------------------ remediation

def test_policy_defaults_and_back_compat():
    assert RM.policy({}, "discover")["action"] == "alert"
    assert RM.policy({}, "manual")["action"] == "cordon"
    assert RM.policy({"remediation": {"taintNodes": True}}, "discover")["action"] == "cordon"
    p = RM.policy({"remediation": {"action": "handoff"}}, "discover")
    assert p["action"] == "handoff" and p["autoReleaseAfterReboot"] is True
    p = RM.policy({"remediation": {"diagnostics": {"nvidia": {"command": ["dcgmi", "diag", "-r", "2", "-j"]}}}}, "discover")
    assert p["diagnostics"]["nvidia"]["command"][3] == "2" and p["diagnostics"]["nvidia"]["podSelectors"]


def test_pods_to_move_skips_own_system_and_daemonsets():
    def pod(ns, name, gpu=True, owner="ReplicaSet", node="n1"):
        res = {"limits": {"nvidia.com/gpu": "1"}} if gpu else {}
        return {"metadata": {"namespace": ns, "name": name, "ownerReferences": [{"kind": owner}]},
                "spec": {"nodeName": node, "containers": [{"resources": res}]}, "status": {"phase": "Running"}}
    pods = [pod("apps", "llm"), pod("apps", "web", gpu=False), pod("dualkey", "dk-model-ref"),
            pod("openshift-x", "sys"), pod("gpu-op", "ds", owner="DaemonSet"), pod("apps", "other", node="n2")]
    move, kept = RM.pods_to_move(pods, "n1", "dualkey", ["openshift-", "kube-"])
    assert move == [("apps", "llm")]
    assert {k[1] for k in kept} == {"dk-model-ref", "sys", "ds"}


def test_evidence_explains_the_quarantine():
    scorer = {"settings": {"std_floor": 0.001},
              "baselines": {"nvidia|nvidia": {"n": 200, "mean": 0.012, "m2": 199 * 0.004 ** 2, "frozen": True}},
              "pairs": {"l4|t4": {"ewma": 7.6, "n": 338}, "a10g|t4": {"ewma": 6.9, "n": 311}, "a10g|l4": {"ewma": 0.2, "n": 300}},
              "gpus": {"t4": {"vendor": "nvidia", "state": "Quarantined"}, "l4": {"vendor": "nvidia", "state": "Healthy"},
                       "a10g": {"vendor": "nvidia", "state": "Healthy"}}}
    results = [{"ts": 1, "primary": "t4", "checker": "l4", "gap": 0.052}, {"ts": 1, "primary": "l4", "checker": "a10g", "gap": 0.01}]
    ev = RM.build_evidence("t4", "nvidia", scorer, results, {"score": 6.9, "build": {"vllm": "0.11.0"}},
                           {"suspect": 3, "quarantine": 5}, {"modelId": "granite"})
    assert [p["partner"] for p in ev["partners"]] == ["l4", "a10g"]
    assert "2 of 2 partners" in ev["summary"]
    assert len(ev["recentChecks"]) == 1 and ev["recentChecks"][0]["z"] == 10.0
    assert ev["baselines"]["nvidia|nvidia"]["std"] == 0.004 and ev["identity"]["modelId"] == "granite"


def test_parse_vendor_diagnostics():
    ok = {"DCGM GPU Diagnostic": {"test_categories": [{"category": "Deployment", "tests": [
        {"name": "Denylist", "results": [{"gpu_id": "0", "status": "Pass"}]},
        {"name": "Persistence Mode", "results": [{"gpu_id": "0", "status": "Pass"}]}]}]}}
    assert RM.parse_dcgm(json.dumps(ok))[0] == "Pass"
    bad = json.loads(json.dumps(ok))
    bad["DCGM GPU Diagnostic"]["test_categories"][0]["tests"][1]["results"][0]["status"] = "Fail"
    res, summary = RM.parse_dcgm("Successfully ran diagnostic\n" + json.dumps(bad))
    assert res == "Fail" and "Persistence Mode" in summary
    assert RM.parse_dcgm("")[0] == "Unknown"
    assert RM.parse_amd(json.dumps([{"gpu": 0, "ecc": {"total_correctable_count": 3, "total_uncorrectable_count": 0}}]))[0] == "Pass"
    assert RM.parse_amd(json.dumps([{"gpu": 0, "ecc": {"total_uncorrectable_count": 2}}]))[0] == "Fail"


def test_validation_decision():
    args = dict(start_checks=100, partners_needed=2, checks_per_partner=20, suspect_z=3, started=0, timeout_minutes=30)
    assert RM.validation_decision({"state": "Healthy", "score": None, "checks": 110}, now=60, **args)[0] == "wait"
    assert RM.validation_decision({"state": "Healthy", "score": 0.4, "checks": 150}, now=600, **args)[0] == "passed"
    assert RM.validation_decision({"state": "Suspect", "score": 3.5, "checks": 150}, now=600, **args)[0] == "failed"
    assert RM.validation_decision({"state": "Healthy", "score": None, "checks": 120}, now=3600, **args)[0] == "timeout"


def test_incident_name_and_condition():
    assert RM.incident_name("ip-10-0-1-116.us-east-2.compute.internal", 0) == "ip-10-0-1-116-19700101-000000"
    c = RM.condition(True, "SilentDataCorruptionSuspected", "x")
    assert c["type"] == "DualKeyInferenceIntegrity" and c["status"] == "True"
