"""Builders for every Kubernetes object the operator manages.

Each function takes the DualKeyCluster spec and returns plain dicts, so the
manifests can be unit-tested without a cluster.
"""
from __future__ import annotations

import hashlib
import re
import json

GROUP = "dualkey.io"
VERSION = "v1alpha1"
DEFAULT_IMAGE = "image-registry.openshift-image-registry.svc:5000/dualkey/dualkey:latest"
GPU_RESOURCE = {"nvidia": "nvidia.com/gpu", "amd": "amd.com/gpu"}
DEFAULT_MODEL_IMAGE = {"nvidia": "docker.io/vllm/vllm-openai:latest",
                       "amd": "docker.io/rocm/vllm:latest"}
API_KEY_SECRET = "dualkey-api-key"
CONFIG_NAME = "dualkey-config"
SUSPECT_TAINT = "dualkey.io/suspect"

DETECTOR_KEYS = {  # CR camelCase -> scoring.Settings field
    "baselineSamples": "baseline_samples", "stdFloor": "std_floor", "ewmaAlpha": "ewma_alpha",
    "minPairChecks": "min_pair_checks", "minPartners": "min_partners",
    "partnerQuantile": "partner_quantile", "suspectZ": "suspect_z",
    "quarantineZ": "quarantine_z", "confirmChecks": "confirm_checks",
}


def _labels(cluster: str, component: str) -> dict:
    return {"app.kubernetes.io/name": "dualkey", "app.kubernetes.io/instance": cluster,
            "app.kubernetes.io/component": component, "dualkey.io/cluster": cluster}


def image(spec: dict) -> str:
    return spec.get("image") or DEFAULT_IMAGE


def model_service_name(gpu_name: str) -> str:
    return f"dk-model-{gpu_name}"


def gpu_endpoint(gpu: dict) -> str:
    return gpu.get("endpoint") or f"http://{model_service_name(gpu['name'])}:8000"


def manual_groups(spec: dict) -> list[dict]:
    """Manual mode: the GPUs listed in spec.gpus form one group called "manual"."""
    served = spec.get("model", {}).get("servedName", "granite")
    gpus = spec.get("gpus") or []
    if not gpus:
        return []
    return [{"name": "manual", "model_id": spec.get("model", {}).get("name", served),
             "source": "manual",
             "members": [{"name": g["name"], "vendor": g["vendor"].lower(),
                          "endpoint": gpu_endpoint(g), "node": g.get("node", ""),
                          "model": served, "auth": g.get("auth", "dualkey")} for g in gpus]}]


def runtime_config(spec: dict, name: str, namespace: str, groups: list[dict]) -> dict:
    sampling = spec.get("sampling", {})
    checks = spec.get("checks", {})
    probe = spec.get("probe", {})
    det = {DETECTOR_KEYS[k]: v for k, v in (spec.get("detector") or {}).items() if k in DETECTOR_KEYS}
    return {
        "groups": groups,
        "redis_url": f"redis://dualkey-redis.{namespace}.svc:6379/0",
        "sample_rate": float(sampling.get("rate", 0.05)),
        "suspect_sample_rate": float(sampling.get("suspectRate", 0.5)),
        "cross_vendor_weight": float(checks.get("crossVendorWeight", 0.7)),
        "check_all": bool(checks.get("checkAll", False)),
        "verifier_concurrency": int(checks.get("concurrency", 8)),
        "probe_rate": float(probe.get("rate", 1.0)),
        "probe_max_tokens": int(probe.get("maxTokens", 48)),
        "detector": det,
        "namespace": namespace,
        "cluster_name": name,
    }


def config_map(spec: dict, name: str, namespace: str, groups: list[dict]) -> dict:
    data = json.dumps(runtime_config(spec, name, namespace, groups), indent=2, sort_keys=True)
    return {"apiVersion": "v1", "kind": "ConfigMap",
            "metadata": {"name": CONFIG_NAME, "namespace": namespace,
                         "labels": _labels(name, "config")},
            "data": {"config.json": data}}


def config_hash(cm: dict) -> str:
    return hashlib.sha256(cm["data"]["config.json"].encode()).hexdigest()[:16]


def _api_key_env() -> dict:
    return {"name": "DK_API_KEY",
            "valueFrom": {"secretKeyRef": {"name": API_KEY_SECRET, "key": "DK_API_KEY"}}}


def _service(name: str, namespace: str, cluster: str, component: str, port: int,
             port_name: str = "http") -> dict:
    return {"apiVersion": "v1", "kind": "Service",
            "metadata": {"name": name, "namespace": namespace,
                         "labels": _labels(cluster, component)},
            "spec": {"selector": {"dualkey.io/cluster": cluster,
                                  "app.kubernetes.io/component": component},
                     "ports": [{"name": port_name, "port": port, "targetPort": port}]}}


def _deployment(name, namespace, cluster, component, containers, replicas=1,
                volumes=None, init=None, sa=None, recreate=False, annotations=None,
                node_selector=None, tolerations=None) -> dict:
    pod_spec = {"containers": containers}
    if volumes:
        pod_spec["volumes"] = volumes
    if init:
        pod_spec["initContainers"] = init
    if sa:
        pod_spec["serviceAccountName"] = sa
    if node_selector:
        pod_spec["nodeSelector"] = node_selector
    if tolerations:
        pod_spec["tolerations"] = tolerations
    dep = {"apiVersion": "apps/v1", "kind": "Deployment",
           "metadata": {"name": name, "namespace": namespace,
                        "labels": _labels(cluster, component)},
           "spec": {"replicas": replicas,
                    "selector": {"matchLabels": {"dualkey.io/cluster": cluster,
                                                 "app.kubernetes.io/component": component}},
                    "template": {"metadata": {"labels": _labels(cluster, component),
                                              "annotations": annotations or {}},
                                 "spec": pod_spec}}}
    if recreate:
        dep["spec"]["strategy"] = {"type": "Recreate"}
    return dep


def _config_volume():
    return [{"name": "config", "configMap": {"name": CONFIG_NAME}}]


def _app_container(spec, name, command, port=None, extra_env=None):
    c = {"name": name, "image": image(spec), "command": command,
         "env": [{"name": "DK_CONFIG", "value": "/etc/dualkey/config.json"},
                 _api_key_env()] + (extra_env or []),
         "volumeMounts": [{"name": "config", "mountPath": "/etc/dualkey"}],
         "resources": {"requests": {"cpu": "100m", "memory": "256Mi"},
                       "limits": {"memory": "1Gi"}}}
    if port:
        c["ports"] = [{"containerPort": port, "name": "http"}]
    return c


# ---------------------------------------------------------------- components

def redis(spec, name, ns) -> list[dict]:
    img = spec.get("redisImage", "docker.io/library/redis:7")
    container = {"name": "redis", "image": img,
                 "command": ["redis-server", "--save", "", "--appendonly", "no",
                             "--protected-mode", "no", "--dir", "/data"],
                 "ports": [{"containerPort": 6379, "name": "redis"}],
                 "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                 "resources": {"requests": {"cpu": "100m", "memory": "256Mi"},
                               "limits": {"memory": "2Gi"}}}
    return [_deployment("dualkey-redis", ns, name, "redis", [container],
                        volumes=[{"name": "data", "emptyDir": {}}], recreate=True),
            _service("dualkey-redis", ns, name, "redis", 6379, "redis")]


def gateway(spec, name, ns, cfg_hash) -> list[dict]:
    c = _app_container(spec, "gateway",
                       ["uvicorn", "dualkey.gateway:app", "--host", "0.0.0.0", "--port", "8080"],
                       port=8080)
    c["readinessProbe"] = {"httpGet": {"path": "/healthz", "port": 8080},
                           "periodSeconds": 5}
    objs = [_deployment("dualkey-gateway", ns, name, "gateway", [c],
                        replicas=int(spec.get("gateway", {}).get("replicas", 1)),
                        volumes=_config_volume(),
                        annotations={"dualkey.io/config-hash": cfg_hash}),
            _service("dualkey-gateway", ns, name, "gateway", 8080)]
    if spec.get("gateway", {}).get("route", True):
        objs.append({"apiVersion": "route.openshift.io/v1", "kind": "Route",
                     "metadata": {"name": "dualkey-gateway", "namespace": ns,
                                  "labels": _labels(name, "gateway")},
                     "spec": {"to": {"kind": "Service", "name": "dualkey-gateway"},
                              "port": {"targetPort": "http"},
                              "tls": {"termination": "edge",
                                      "insecureEdgeTerminationPolicy": "Redirect"}}})
    return objs


def verifier(spec, name, ns, cfg_hash) -> list[dict]:
    c = _app_container(spec, "verifier", ["python", "-m", "dualkey.verifier"])
    c["ports"] = [{"containerPort": 9100, "name": "metrics"}]
    return [_deployment("dualkey-verifier", ns, name, "verifier", [c],
                        replicas=int(spec.get("checks", {}).get("verifierReplicas", 1)),
                        volumes=_config_volume(),
                        annotations={"dualkey.io/config-hash": cfg_hash}),
            _service("dualkey-verifier-metrics", ns, name, "verifier", 9100, "metrics")]


def detector(spec, name, ns, cfg_hash) -> list[dict]:
    c = _app_container(spec, "detector", ["python", "-m", "dualkey.detector"])
    c["ports"] = [{"containerPort": 9100, "name": "metrics"}]
    return [_deployment("dualkey-detector", ns, name, "detector", [c], volumes=_config_volume(),
                        sa="dualkey-detector", recreate=True,
                        annotations={"dualkey.io/config-hash": cfg_hash}),
            _service("dualkey-detector-metrics", ns, name, "detector", 9100, "metrics")]


def service_monitor(spec, name, ns) -> list[dict]:
    if not spec.get("monitoring", {}).get("serviceMonitor", False):
        return []
    return [{"apiVersion": "monitoring.coreos.com/v1", "kind": "ServiceMonitor",
             "metadata": {"name": "dualkey", "namespace": ns, "labels": _labels(name, "monitoring")},
             "spec": {"selector": {"matchLabels": {"dualkey.io/cluster": name}},
                      "endpoints": [{"port": "metrics", "interval": "15s"},
                                    {"port": "http", "path": "/metrics", "interval": "15s"}]}}]


def model_server(spec, name, ns, gpu: dict) -> list[dict]:
    """vLLM server pinned to one GPU node, with optional lab fault injection."""
    if gpu.get("endpoint"):
        return []  # an existing server is used instead
    vendor = gpu["vendor"].lower()
    model = spec.get("model", {})
    servers = spec.get("modelServers", {})
    served = model.get("servedName", "granite")
    img = servers.get(f"{vendor}Image") or DEFAULT_MODEL_IMAGE[vendor]
    storage = servers.get("storageSize", "40Gi")
    res_name = GPU_RESOURCE[vendor]
    hf_env = [{"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {
        "name": model.get("hfTokenSecret", "dualkey-hf-token"), "key": "HF_TOKEN",
        "optional": True}}}]
    if model.get("revision"):
        hf_env.append({"name": "DK_MODEL_REVISION", "value": model["revision"]})
    mounts = [{"name": "models", "mountPath": "/models"}]

    init = [{"name": "download", "image": image(spec),
             "command": ["python", "-m", "dualkey.download",
                         "--model", model.get("name", "ibm-granite/granite-3.3-2b-instruct"),
                         "--dir", "/models/model"],
             "env": hf_env + [{"name": "HF_HOME", "value": "/models/.cache"}],
             "volumeMounts": mounts}]
    fault = gpu.get("faultInjection") or {}
    if fault.get("enabled"):
        cmd = ["python", "-m", "dualkey.faultinject", "--model-dir", "/models/model",
               "--bit", str(fault.get("bit", 13)), "--count", str(fault.get("count", 100)),
               "--seed", str(fault.get("seed", 7)),
               "--ground-truth", "/models/ground_truth.json"]
        if fault.get("tensor"):
            cmd += ["--tensor", fault["tensor"]]
        init.append({"name": "fault-injection", "image": image(spec), "command": cmd,
                     "env": [{"name": "DK_GPU_NAME", "value": gpu["name"]}],
                     "volumeMounts": mounts})

    args = ["/models/model", "--served-model-name", served, "--dtype", model.get("dtype", "bfloat16"),
            "--host", "0.0.0.0", "--port", "8000",
            "--max-model-len", str(model.get("maxModelLen", 4096)),
            "--api-key", "$(DK_API_KEY)"] + list(servers.get("extraArgs", []))
    container = {
        "name": "vllm", "image": img, "command": ["vllm", "serve"], "args": args,
        "env": [_api_key_env(),
                {"name": "HOME", "value": "/models"},
                {"name": "HF_HOME", "value": "/models/.cache"},
                {"name": "XDG_CACHE_HOME", "value": "/models/.cache"},
                {"name": "TRITON_CACHE_DIR", "value": "/models/.cache/triton"},
                {"name": "VLLM_NO_USAGE_STATS", "value": "1"}],
        "ports": [{"containerPort": 8000, "name": "http"}],
        "resources": {"limits": {res_name: 1},
                      "requests": {"cpu": servers.get("cpu", "2"),
                                   "memory": servers.get("memory", "16Gi")}},
        "volumeMounts": mounts + [{"name": "shm", "mountPath": "/dev/shm"}],
        "startupProbe": {"httpGet": {"path": "/health", "port": 8000},
                         "periodSeconds": 10, "failureThreshold": 90},
        "readinessProbe": {"httpGet": {"path": "/health", "port": 8000}, "periodSeconds": 10},
    }
    sel = {"kubernetes.io/hostname": gpu["node"]} if gpu.get("node") else None
    tol = [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
           {"key": "amd.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
           {"key": SUSPECT_TAINT, "operator": "Exists", "effect": "NoSchedule"},
           # Dual-Key's own server stays on a cordoned node (e.g. by NVSentinel) so it can be re-checked
           {"key": "node.kubernetes.io/unschedulable", "operator": "Exists", "effect": "NoSchedule"}]
    comp = f"model-{gpu['name']}"
    # the model-id label lets discovery recognise the model behind the local path
    labels_extra = {"dualkey.io/gpu": gpu["name"], "dualkey.io/vendor": vendor,
                    "dualkey.io/model-id": label_value(model.get("name", "ibm-granite/granite-3.3-2b-instruct"))}
    dep = _deployment(model_service_name(gpu["name"]), ns, name, comp, [container],
                      volumes=[{"name": "models", "emptyDir": {"sizeLimit": storage}},
                               {"name": "shm", "emptyDir": {"medium": "Memory",
                                                            "sizeLimit": servers.get("shmSize", "2Gi")}}],
                      init=init, recreate=True, node_selector=sel, tolerations=tol,
                      annotations={"dualkey.io/fault-injection": json.dumps(fault, sort_keys=True),
                                   "dualkey.io/model-revision": model.get("revision", "")})
    dep["metadata"]["labels"].update(labels_extra)
    dep["spec"]["template"]["metadata"]["labels"].update(labels_extra)
    svc = _service(model_service_name(gpu["name"]), ns, name, comp, 8000)
    return [dep, svc]


def label_value(text: str) -> str:
    """Kubernetes label values: <=63 chars of [A-Za-z0-9._-]; '/' becomes '.'."""
    v = re.sub(r"[^A-Za-z0-9._-]", ".", text.replace("/", "."))
    return v[:63].strip("._-") or "x"


def reference_servers(spec, name, ns, placements: list[dict]) -> list[dict]:
    """The prebuilt reference model, one server per chosen free GPU node."""
    objs = []
    for p in placements:
        gpu = {"name": reference_member_name(p["node"]), "vendor": p["vendor"], "node": p["node"]}
        objs += model_server(spec, name, ns, gpu)
    return objs


def reference_member_name(node: str) -> str:
    return "ref-" + re.sub(r"[^a-z0-9-]+", "-", node.split(".")[0].lower()).strip("-")[:50]


def gpu_health_record(name, ns, gpu: dict) -> dict:
    return {"apiVersion": f"{GROUP}/{VERSION}", "kind": "GPUHealthRecord",
            "metadata": {"name": gpu["name"], "namespace": ns,
                         "labels": _labels(name, "gpu-record")},
            "spec": {"gpu": gpu["name"], "vendor": gpu["vendor"].lower(),
                     "node": gpu.get("node", ""), "group": gpu.get("group", "")}}


def comparison_group(name, ns, group: dict) -> dict:
    """Spec part only; the operator writes the status separately."""
    return {"apiVersion": f"{GROUP}/{VERSION}", "kind": "ComparisonGroup",
            "metadata": {"name": group["name"], "namespace": ns,
                         "labels": _labels(name, "comparison-group")},
            "spec": {"modelId": group.get("model_id", ""), "dtype": group.get("dtype", ""),
                     "quantization": group.get("quantization", ""),
                     "source": group.get("source", "discovered"),
                     "identity": {k: str(v) for k, v in (group.get("identity") or {}).items()}}}


def prober(spec, name, ns, cfg_hash) -> list[dict]:
    c = _app_container(spec, "prober", ["python", "-m", "dualkey.prober"],
                       extra_env=[_discovery_key_env()])
    c["ports"] = [{"containerPort": 9100, "name": "metrics"}]
    return [_deployment("dualkey-prober", ns, name, "prober", [c], volumes=_config_volume(),
                        annotations={"dualkey.io/config-hash": cfg_hash}),
            _service("dualkey-prober-metrics", ns, name, "prober", 9100, "metrics")]


def ui(spec, name, ns) -> list[dict]:
    c = {"name": "ui", "image": image(spec),
         "command": ["uvicorn", "dualkey.ui:app", "--host", "0.0.0.0", "--port", "8080"],
         "env": [{"name": "DK_NAMESPACE", "value": ns}, {"name": "DK_CLUSTER", "value": name},
                 {"name": "DK_REDIS_URL", "value": f"redis://dualkey-redis.{ns}.svc:6379/0"}],
         "ports": [{"containerPort": 8080, "name": "http"}],
         "readinessProbe": {"httpGet": {"path": "/healthz", "port": 8080}, "periodSeconds": 10},
         "resources": {"requests": {"cpu": "50m", "memory": "128Mi"}, "limits": {"memory": "512Mi"}}}
    objs = [_deployment("dualkey-ui", ns, name, "ui", [c], sa="dualkey-ui"),
            _service("dualkey-ui", ns, name, "ui", 8080)]
    objs.append({"apiVersion": "route.openshift.io/v1", "kind": "Route",
                 "metadata": {"name": "dualkey-dashboard", "namespace": ns, "labels": _labels(name, "ui")},
                 "spec": {"to": {"kind": "Service", "name": "dualkey-ui"},
                          "port": {"targetPort": "http"},
                          "tls": {"termination": "edge", "insecureEdgeTerminationPolicy": "Redirect"}}})
    return objs


def _discovery_key_env() -> dict:
    return {"name": "DK_DISCOVERY_API_KEY", "valueFrom": {"secretKeyRef": {
        "name": "dualkey-discovery-api-key", "key": "DK_DISCOVERY_API_KEY", "optional": True}}}


def mode(spec: dict) -> str:
    return (spec.get("mode") or ("manual" if spec.get("gpus") else "discover")).lower()


def all_objects(spec: dict, name: str, ns: str, groups: list[dict] | None = None,
                placements: list[dict] | None = None) -> list[dict]:
    """Everything the operator applies for one DualKeyCluster (except Secrets and CR records).

    groups:     comparison groups to check (manual mode: built from spec.gpus).
    placements: where the reference model runs (discover mode fallback).
    """
    if groups is None:
        groups = manual_groups(spec) if mode(spec) == "manual" else []
    cm = config_map(spec, name, ns, groups)
    h = config_hash(cm)
    objs = [cm] + redis(spec, name, ns)
    if mode(spec) == "manual":
        for gpu in spec.get("gpus", []):
            objs += model_server(spec, name, ns, gpu)
    objs += reference_servers(spec, name, ns, placements or [])
    objs += gateway(spec, name, ns, h) + prober(spec, name, ns, h) + verifier(spec, name, ns, h)
    objs += detector(spec, name, ns, h) + ui(spec, name, ns) + service_monitor(spec, name, ns)
    for o in objs:   # every component can reach discovered servers that need their own key
        if o["kind"] == "Deployment" and o["metadata"]["name"] in ("dualkey-gateway", "dualkey-verifier"):
            o["spec"]["template"]["spec"]["containers"][0]["env"].append(_discovery_key_env())
    return objs


def validate(spec: dict) -> list[str]:
    if mode(spec) == "discover":
        return []
    errs = []
    gpus = spec.get("gpus", [])
    names = [g.get("name") for g in gpus]
    if len(gpus) < 3:
        errs.append("at least 3 GPUs are needed so a faulty one can be identified by majority")
    if len(set(names)) != len(names):
        errs.append("GPU names must be unique")
    vendors = {str(g.get("vendor", "")).lower() for g in gpus}
    if not vendors <= {"nvidia", "amd"}:
        errs.append("vendor must be nvidia or amd")
    if len(vendors) < 2:
        errs.append("warning: only one vendor; cross-vendor checks are not possible")
    for g in gpus:
        if not g.get("endpoint") and not g.get("node"):
            errs.append(f"GPU {g.get('name')}: set either node (to deploy a model server) or endpoint")
    return errs
