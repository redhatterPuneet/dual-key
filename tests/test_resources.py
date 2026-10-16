import copy
import glob
import json

import yaml

from dualkey import resources as R

SAMPLE = "deploy/samples/dualkeycluster.yaml"


def sample_spec():
    with open(SAMPLE) as f:
        return yaml.safe_load(f)["spec"]


def test_all_deploy_yaml_parses():
    for path in glob.glob("deploy/**/*.yaml", recursive=True):
        if path.endswith("-values.yaml"):
            continue   # Helm values for another product, not a Kubernetes manifest
        with open(path) as f:
            docs = [d for d in yaml.safe_load_all(f) if d]
        assert docs, path
        for d in docs:
            assert "kind" in d, path


def test_objects_for_sample_cluster():
    spec = sample_spec()
    objs = R.all_objects(spec, "lab", "dualkey")
    kinds = [(o["kind"], o["metadata"]["name"]) for o in objs]
    for expected in [("ConfigMap", "dualkey-config"), ("Deployment", "dualkey-redis"),
                     ("Deployment", "dk-model-nv-a"), ("Deployment", "dk-model-nv-b"),
                     ("Deployment", "dk-model-amd-1"), ("Deployment", "dualkey-gateway"),
                     ("Route", "dualkey-gateway"), ("Deployment", "dualkey-verifier"),
                     ("Deployment", "dualkey-detector")]:
        assert expected in kinds, expected

    cfg = json.loads(objs[0]["data"]["config.json"])
    assert [m["endpoint"] for m in cfg["groups"][0]["members"]] == [
        "http://dk-model-nv-a:8000", "http://dk-model-nv-b:8000", "http://dk-model-amd-1:8000"]
    assert cfg["check_all"] is True
    assert cfg["detector"]["baseline_samples"] == 200


def test_model_servers_use_right_gpu_and_node():
    spec = sample_spec()
    by = {o["metadata"]["name"]: o for o in R.all_objects(spec, "lab", "dualkey") if o["kind"] == "Deployment"}
    nv = by["dk-model-nv-a"]["spec"]["template"]["spec"]
    amd = by["dk-model-amd-1"]["spec"]["template"]["spec"]
    assert nv["containers"][0]["resources"]["limits"] == {"nvidia.com/gpu": 1}
    assert amd["containers"][0]["resources"]["limits"] == {"amd.com/gpu": 1}
    assert nv["nodeSelector"] == {"kubernetes.io/hostname": "REPLACE-gpu-node-1"}
    assert "rocm" in amd["containers"][0]["image"]
    assert [c["name"] for c in nv["initContainers"]] == ["download"]


def test_fault_injection_adds_init_container_only_on_that_gpu():
    spec = copy.deepcopy(sample_spec())
    spec["gpus"][1]["faultInjection"]["enabled"] = True
    by = {o["metadata"]["name"]: o for o in R.all_objects(spec, "lab", "dualkey") if o["kind"] == "Deployment"}
    nvb = by["dk-model-nv-b"]["spec"]["template"]["spec"]["initContainers"]
    assert [c["name"] for c in nvb] == ["download", "fault-injection"]
    assert "--bit" in nvb[1]["command"] and "13" in nvb[1]["command"]
    nva = by["dk-model-nv-a"]["spec"]["template"]["spec"]["initContainers"]
    assert [c["name"] for c in nva] == ["download"]


def test_external_endpoint_skips_model_server():
    spec = copy.deepcopy(sample_spec())
    spec["gpus"][2] = {"name": "amd-1", "vendor": "amd", "endpoint": "https://amd-vm:8000"}
    names = [o["metadata"]["name"] for o in R.all_objects(spec, "lab", "dualkey")]
    assert "dk-model-amd-1" not in names
    cfg = json.loads(R.config_map(spec, "lab", "dualkey", R.manual_groups(spec))["data"]["config.json"])
    assert cfg["groups"][0]["members"][2]["endpoint"] == "https://amd-vm:8000"


def test_validation():
    assert any("at least 3" in e for e in R.validate({"gpus": [{"name": "a", "vendor": "nvidia", "node": "n"}]}))
    assert not [e for e in R.validate(sample_spec()) if not e.startswith("warning")]


def test_operator_module_imports():
    import dualkey.operator  # noqa: F401  (registers kopf handlers)


def test_discover_mode_objects():
    spec = {"model": {"name": "ibm-granite/granite-3.3-2b-instruct"}}
    assert R.mode(spec) == "discover" and R.validate(spec) == []
    objs = R.all_objects(spec, "fleet", "dualkey", [], [])
    names = {(o["kind"], o["metadata"]["name"]) for o in objs}
    assert ("Deployment", "dualkey-ui") in names and ("Route", "dualkey-dashboard") in names
    assert ("Deployment", "dualkey-prober") in names
    assert not any(n.startswith("dk-model") for _, n in names)
    placed = R.all_objects(spec, "fleet", "dualkey", [], [{"node": "gpu-7.example", "vendor": "nvidia"},
                                                           {"node": "amd-9", "vendor": "amd"}])
    deps = {o["metadata"]["name"]: o for o in placed if o["kind"] == "Deployment"}
    assert "dk-model-ref-gpu-7" in deps and "dk-model-ref-amd-9" in deps
    labels = deps["dk-model-ref-amd-9"]["spec"]["template"]["metadata"]["labels"]
    assert labels["dualkey.io/model-id"] == "ibm-granite.granite-3.3-2b-instruct"
    assert deps["dk-model-ref-amd-9"]["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"] == {"amd.com/gpu": 1}


def test_sample_discovery_cr_is_valid():
    with open("deploy/samples/dualkeycluster-discover.yaml") as f:
        spec = yaml.safe_load(f)["spec"]
    assert R.mode(spec) == "discover"
    assert R.validate(spec) == []


def test_aws_sample_uses_float16_and_fits_xlarge():
    with open("deploy/samples/dualkeycluster-aws.yaml") as f:
        spec = yaml.safe_load(f)["spec"]
    deps = [o for o in R.all_objects(spec, "fleet", "dualkey", [], [{"node": "n1", "vendor": "nvidia"}])
            if o["kind"] == "Deployment" and o["metadata"]["name"].startswith("dk-model-ref")]
    c = deps[0]["spec"]["template"]["spec"]["containers"][0]
    assert c["args"][c["args"].index("--dtype") + 1] == "float16"
    assert c["resources"]["requests"]["memory"] == "10Gi"
