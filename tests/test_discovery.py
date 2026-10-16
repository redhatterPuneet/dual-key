from dualkey import discovery as D
from dualkey import operator as OP
from dualkey import ui


def pod(name, ns, node, gpu_res="nvidia.com/gpu", count="1", args=None, image="vllm/vllm-openai:v0.9",
        rs="llama-7d9f8", ip="10.0.0.1", ready=True, labels=None, annotations=None, command=None):
    return {
        "metadata": {"name": name, "namespace": ns, "labels": labels or {}, "annotations": annotations or {},
                     "ownerReferences": [{"kind": "ReplicaSet", "name": rs}]},
        "spec": {"nodeName": node, "containers": [{
            "name": "server", "image": image, "command": command or ["vllm", "serve"],
            "args": args if args is not None else ["meta-llama/Llama-3.1-8B-Instruct", "--dtype", "bfloat16"],
            "resources": {"limits": {gpu_res: count}} if gpu_res else {}}]},
        "status": {"phase": "Running", "podIP": ip,
                   "conditions": [{"type": "Ready", "status": "True" if ready else "False"}]},
    }


def node(name, res="nvidia.com/gpu", n="1", taints=None):
    return {"metadata": {"name": name}, "spec": {"taints": taints or []},
            "status": {"allocatable": {res: n, "cpu": "32"}}}


def cluster_pods():
    return [
        pod("llama-7d9f8-a", "chatbot", "gpu-1", ip="10.0.0.1"),
        pod("llama-7d9f8-b", "chatbot", "gpu-2", ip="10.0.0.2"),
        # KServe-style: model path is local, identity comes from the storage-uri annotation
        pod("granite-pred-x", "rag", "gpu-3", rs="granite-pred-55c",
            args=["--model=/mnt/models", "--served-model-name=granite", "--port=8080", "--dtype=bfloat16"],
            annotations={"internal.serving.kserve.io/storage-initializer-sourceuri": "hf://ibm-granite/granite-3.3-8b-instruct"}),
        # GPU pod that is not vLLM -> skipped with a reason
        pod("trainer-0", "ml", "gpu-4", command=["python", "train.py"], args=[], image="pytorch:2", rs="trainer"),
        # vLLM without a GPU -> ignored
        pod("cpu-vllm", "dev", "cpu-1", gpu_res=None, rs="cpu-vllm-1"),
        # platform namespaces are skipped
        pod("x", "openshift-monitoring", "gpu-5", rs="x-1"),
    ]


def test_discover_servers_and_identity():
    servers, skipped = D.discover_servers(cluster_pods())
    by = {s.pod: s for s in servers}
    assert set(by) == {"llama-7d9f8-a", "llama-7d9f8-b", "granite-pred-x"}
    assert by["llama-7d9f8-a"].model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert by["llama-7d9f8-a"].endpoint == "http://10.0.0.1:8000"
    assert by["llama-7d9f8-a"].key == "chatbot-llama-gpu-1"
    assert by["granite-pred-x"].model_id == "hf://ibm-granite/granite-3.3-8b-instruct"
    assert by["granite-pred-x"].endpoint.endswith(":8080")
    assert by["granite-pred-x"].served_name == "granite"
    assert [s.pod for s in skipped] == ["trainer-0"]
    assert "not a vLLM" in skipped[0].reason


def test_groups_too_small_then_ready():
    servers, _ = D.discover_servers(cluster_pods())
    groups = D.form_groups(servers)
    llama = next(g for g in groups if "Llama" in g.model_id)
    assert llama.phase == "TooSmall" and llama.message == "2 of 3 ready members needed"
    servers, _ = D.discover_servers(cluster_pods() + [
        pod("llama-7d9f8-c", "chatbot", "gpu-6", gpu_res="amd.com/gpu", ip="10.0.0.6")])
    llama = next(g for g in D.form_groups(servers) if "Llama" in g.model_id)
    assert llama.phase == "Ready" and llama.vendors == ["amd", "nvidia"]
    assert llama.source == "discovered"


def test_different_precision_is_a_different_group():
    pods = [pod(f"p{i}", "a", f"n{i}", rs=f"r{i}-x", ip=f"10.1.0.{i}") for i in range(3)]
    pods[2]["spec"]["containers"][0]["args"] = ["meta-llama/Llama-3.1-8B-Instruct", "--dtype", "float16"]
    groups = D.form_groups(D.discover_servers(pods)[0])
    assert sorted(len(g.members) for g in groups) == [1, 2]


def test_probe_failure_marks_member_not_ready():
    servers, _ = D.discover_servers(cluster_pods())
    D.apply_probes(servers, {"chatbot-llama-gpu-1": {"ok": False, "error": "needs an API key"}})
    s = next(s for s in servers if s.key == "chatbot-llama-gpu-1")
    assert not s.ready and s.reason == "needs an API key"


def test_probe_root_names_a_local_path_model():
    p = pod("p", "a", "n1", args=["/models/model", "--served-model-name", "m"], rs="r-x")
    servers, _ = D.discover_servers([p])
    D.apply_probes(servers, {servers[0].key: {"ok": True, "root": "Qwen/Qwen2.5-7B-Instruct", "models": ["m"]}})
    assert servers[0].model_id == "Qwen/Qwen2.5-7B-Instruct"


def test_reference_plan_mixes_vendors_and_is_sticky():
    pods = cluster_pods()
    nodes = [node("gpu-1"), node("gpu-2"), node("gpu-3"), node("gpu-4"),
             node("gpu-7"), node("gpu-8"), node("amd-9", "amd.com/gpu"),
             node("gpu-10", taints=[{"key": "dualkey.io/suspect", "effect": "NoSchedule"}])]
    groups = D.form_groups(D.discover_servers(pods)[0])
    free = D.free_gpu_nodes(nodes, pods)
    assert set(free) == {"gpu-7", "gpu-8", "amd-9"}          # used, tainted and non-GPU nodes excluded
    plan, reason = D.plan_reference(groups, free, [])
    assert len(plan) == 3 and {p["vendor"] for p in plan} == {"nvidia", "amd"}
    assert "deploying the reference model" in reason
    again, _ = D.plan_reference(groups, {}, plan)              # sticky: same placement next scan
    assert again == plan
    off, why = D.plan_reference(groups, free, [], enabled=False)
    assert off == [] and "disabled" in why


def test_reference_not_needed_when_discovered_group_ready():
    pods = cluster_pods() + [pod("llama-7d9f8-c", "chatbot", "gpu-6", ip="10.0.0.6")]
    groups = D.form_groups(D.discover_servers(pods)[0])
    plan, reason = D.plan_reference(groups, {"gpu-7": {"vendor": "nvidia", "free": 1}}, [])
    assert plan == [] and "no reference model needed" in reason


def test_not_enough_free_gpus():
    groups = D.form_groups(D.discover_servers(cluster_pods())[0])
    plan, reason = D.plan_reference(groups, {"gpu-7": {"vendor": "nvidia", "free": 1}}, [])
    assert plan == [] and "found 1, need 3" in reason


def test_reference_pods_form_reference_group():
    ref = [pod(f"dk-model-ref-n{i}-x", "dualkey", f"n{i}", rs=f"dk-model-ref-n{i}-55", ip=f"10.2.0.{i}",
               args=["/models/model", "--served-model-name", "granite", "--dtype", "bfloat16"],
               labels={"dualkey.io/cluster": "fleet", "dualkey.io/model-id": "ibm-granite.granite-3.3-2b-instruct"})
           for i in range(3)]
    groups = D.form_groups(D.discover_servers(ref)[0])
    assert len(groups) == 1 and groups[0].source == "reference" and groups[0].phase == "Ready"


def test_runtime_groups_and_timeline_messages():
    pods = cluster_pods() + [pod("llama-7d9f8-c", "chatbot", "gpu-6", ip="10.0.0.6")]
    groups = D.form_groups(D.discover_servers(pods)[0])
    rg = OP.runtime_groups(groups, has_discovery_key=False)
    assert len(rg) == 1 and len(rg[0]["members"]) == 3
    assert all(m["auth"] == "none" for m in rg[0]["members"])
    summary = {g.name: {"modelId": g.model_id, "source": g.source, "phase": g.phase,
                        "ready": len(g.ready_members()), "message": g.message,
                        "memberNames": [m["name"] for m in g.ready_members()]} for g in groups}
    msgs = OP.describe_changes({}, summary, {}, {"placements": [], "reason": "a discovered group is ready"},
                               4, None, first=True)
    assert msgs[0] == "First scan found 4 vLLM model server(s) on GPUs"
    assert any(m.startswith("Formed group") and "3 ready members" in m for m in msgs)
    # a member leaving drops the group back to too small
    prev = summary
    pods2 = cluster_pods()
    groups2 = D.form_groups(D.discover_servers(pods2)[0])
    now = {g.name: {"modelId": g.model_id, "source": g.source, "phase": g.phase, "ready": len(g.ready_members()),
                    "message": g.message, "memberNames": [m["name"] for m in g.ready_members()]} for g in groups2}
    msgs2 = OP.describe_changes(prev, now, {}, {}, 3, 4, first=False)
    assert "Model servers on GPUs: 4 -> 3" in msgs2
    assert any("paused" in m for m in msgs2) and any("left" in m for m in msgs2)


def test_dashboard_state_shape():
    st = ui.build_state(
        {"metadata": {"name": "fleet"}, "spec": {}, "status": {"phase": "Checking", "events": [{"time": "t", "message": "m"}]}},
        [{"metadata": {"name": "g1"}, "spec": {"source": "discovered"}, "status": {"phase": "Ready"}}],
        [{"metadata": {"name": "m1"}, "spec": {}, "status": {"state": "Healthy"}}])
    assert st["cluster"]["mode"] == "discover" and st["cluster"]["phase"] == "Checking"
    assert st["groups"][0]["name"] == "g1" and st["records"][0]["status"]["state"] == "Healthy"
    assert "<title>Dual-Key Fleet View</title>" in ui.page()
