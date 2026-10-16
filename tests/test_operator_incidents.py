"""Incident flow in the operator, against fake Kubernetes APIs."""
import types

import pytest

from dualkey import operator as OP


class FakeCustom:
    def __init__(self):
        self.objs = {}

    def create_namespaced_custom_object(self, g, v, ns, plural, body):
        self.objs[(plural, body["metadata"]["name"])] = {**body, "status": {},
                                                         "metadata": {**body["metadata"], "creationTimestamp": "t0"}}

    def get_namespaced_custom_object(self, g, v, ns, plural, name):
        if (plural, name) not in self.objs:
            raise OP.client.ApiException(status=404)
        return self.objs[(plural, name)]

    def patch_namespaced_custom_object_status(self, g, v, ns, plural, name, body):
        self.objs[(plural, name)]["status"].update(body["status"])

    def patch_namespaced_custom_object(self, g, v, ns, plural, name, body):
        self.objs[(plural, name)].setdefault("spec", {}).update(body["spec"])

    def list_namespaced_custom_object(self, g, v, ns, plural):
        return {"items": [o for (p, _), o in self.objs.items() if p == plural]}


class FakeCore:
    def __init__(self):
        self.taints, self.conditions, self.evicted = [], [], []

    def read_node(self, name):
        return types.SimpleNamespace(spec=types.SimpleNamespace(taints=list(self.taints)),
                                     status=types.SimpleNamespace(node_info=types.SimpleNamespace(boot_id="boot-1")))

    def patch_node(self, name, body):
        self.taints = [types.SimpleNamespace(key=t["key"], to_dict=lambda t=t: t) for t in body["spec"]["taints"]]

    def patch_node_status(self, name, body):
        self.conditions.append(body["status"]["conditions"][0])

    def list_pod_for_all_namespaces(self, field_selector=None, label_selector=None):
        return types.SimpleNamespace(items=[])

    def create_namespaced_pod_eviction(self, name, ns, body):
        self.evicted.append((ns, name))


@pytest.fixture
def fakes(monkeypatch):
    custom, core = FakeCustom(), FakeCore()
    custom.objs[("gpuhealthrecords", "t4")] = {"spec": {"gpu": "t4"}, "status": {"checks": 500, "state": "Quarantined"}}
    custom.objs[("comparisongroups", "g")] = {"spec": {"identity": {"modelId": "granite"}}, "status": {}}
    monkeypatch.setattr(OP, "_custom", lambda: custom)
    monkeypatch.setattr(OP.client, "CoreV1Api", lambda: core)
    monkeypatch.setattr(OP, "_cluster_obj", lambda ns: None)
    monkeypatch.setattr(OP, "read_scorer_and_results", lambda ns, g: ({
        "settings": {}, "baselines": {}, "gpus": {"t4": {"vendor": "nvidia"}, "l4": {"vendor": "nvidia"}},
        "pairs": {"l4|t4": {"ewma": 7.0, "n": 300}}}, []))
    monkeypatch.setattr(OP.client, "ApiClient", lambda: types.SimpleNamespace(sanitize_for_serialization=lambda x: []))
    return custom, core


def run(monkeypatch, fakes, action):
    custom, core = fakes
    monkeypatch.setattr(OP, "_cluster_spec", lambda ns: {"mode": "discover", "remediation": {"action": action}})
    name = OP.open_incident("dualkey", "t4", {"gpu": "t4", "node": "n1", "vendor": "nvidia", "group": "g"},
                            {"score": 7.0, "build": {}})
    return name, custom.objs[("gpuintegrityincidents", name)]


def test_cordon_opens_incident_with_evidence_and_taints(monkeypatch, fakes):
    name, inc = run(monkeypatch, fakes, "cordon")
    st = inc["status"]
    assert st["phase"] == "Contained" and st["policy"] == "cordon" and st["bootID"] == "boot-1"
    assert st["evidence"]["partners"][0]["partner"] == "l4" and st["evidence"]["identity"]["modelId"] == "granite"
    assert [a["action"] for a in st["actions"]][:3] == ["opened", "set node condition DualKeyInferenceIntegrity=True", "cordon"]
    _, core = fakes
    assert [t.key for t in core.taints] == ["dualkey.io/suspect"]
    assert core.conditions[-1]["status"] == "True"


def test_handoff_sets_condition_but_no_taint(monkeypatch, fakes):
    name, inc = run(monkeypatch, fakes, "handoff")
    _, core = fakes
    assert inc["status"]["phase"] == "HandedOff" and core.taints == [] and core.conditions[-1]["status"] == "True"


def test_release_then_validation(monkeypatch, fakes):
    custom, core = fakes
    name, inc = run(monkeypatch, fakes, "cordon")
    OP.release_incident("dualkey", name, inc["spec"], "test")
    st = custom.objs[("gpuintegrityincidents", name)]["status"]
    assert st["phase"] == "Validating" and st["validation"]["startChecks"] == 500
    assert core.taints == [] and core.conditions[-1]["status"] == "False"
    assert custom.objs[("gpuhealthrecords", "t4")]["spec"]["release"] is True
    # detector re-checks the member: healthy with enough new checks
    custom.objs[("gpuhealthrecords", "t4")]["status"] = {"state": "Healthy", "score": 0.3, "checks": 560}
    OP.incident_timer(spec=inc["spec"], status=st, name=name, namespace="dualkey")
    assert custom.objs[("gpuintegrityincidents", name)]["status"]["phase"] == "Resolved"


def test_quarantined_again_reuses_open_incident(monkeypatch, fakes):
    custom, _ = fakes
    name, inc = run(monkeypatch, fakes, "cordon")
    OP.release_incident("dualkey", name, inc["spec"], "test")
    again = OP.open_incident("dualkey", "t4", inc["spec"], {"score": 8.0})
    assert again == name
    assert custom.objs[("gpuintegrityincidents", name)]["status"]["phase"] == "Contained"
    assert any(a["action"] == "quarantined again" for a in custom.objs[("gpuintegrityincidents", name)]["status"]["actions"])


def test_handoff_records_vendor_cordon(monkeypatch, fakes):
    custom, core = fakes
    name, inc = run(monkeypatch, fakes, "handoff")
    orig = core.read_node

    def cordoned(n):
        node = orig(n)
        node.spec.unschedulable = True
        node.metadata = types.SimpleNamespace(labels={"dgxc.nvidia.com/nvsentinel-state": "quarantined"})
        return node
    core.read_node = cordoned
    OP.incident_timer(spec=inc["spec"], status=dict(inc["status"]), name=name, namespace="dualkey")
    st = custom.objs[("gpuintegrityincidents", name)]["status"]
    assert st["vendorAction"]["by"] == "NVSentinel"
    assert st["actions"][-1]["result"] == "node cordoned by NVSentinel (state: quarantined)"
