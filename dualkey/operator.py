"""Dual-Key Operator (kopf).

Two modes, set by DualKeyCluster.spec.mode:

  discover (default) - every few seconds:
      1. scan all pods for vLLM model servers that use GPUs
      2. ask each one which model it runs (/v1/models)
      3. group servers running the same model; a group needs >= 3 ready members
      4. if no discovered group is big enough, deploy the prebuilt reference model
         on free GPU nodes (it then shows up as its own group)
      5. write ComparisonGroup and GPUHealthRecord objects, the runtime config,
         and a human-readable event log the dashboard shows

  manual - deploy one model server per GPU listed in spec.gpus (the v0.1 lab mode).

Also handles incidents: a Quarantined member opens a GPUIntegrityIncident with the
evidence, the response policy acts on its node (alert / cordon / drain / evict /
handoff to the vendor's remediation), the vendor's own diagnostic runs on that
node, and releasing the incident re-checks the member before it is resolved.
See remediation.py.

Run in-cluster: kopf run --standalone --namespace <ns> -m dualkey.operator
"""
from __future__ import annotations

import asyncio
import base64
import logging
import secrets
import time

import httpx
import kopf
from kubernetes import client, config as kcfg
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError

from . import discovery as D
from . import remediation as RM
from . import resources as R

log = logging.getLogger("dualkey.operator")
FIELD_MANAGER = "dualkey-operator"
MAX_EVENTS = 50
DEFAULT_EXCLUDE = ["openshift-", "kube-"]
_dyn: DynamicClient | None = None
_last_scan: dict[str, float] = {}


# ------------------------------------------------------------------ k8s helpers

def dyn() -> DynamicClient:
    global _dyn
    if _dyn is None:
        try:
            kcfg.load_incluster_config()
        except kcfg.ConfigException:
            kcfg.load_kube_config()
        _dyn = DynamicClient(client.ApiClient())
    return _dyn


def apply(obj: dict) -> None:
    api = dyn().resources.get(api_version=obj["apiVersion"], kind=obj["kind"])
    api.server_side_apply(body=obj, name=obj["metadata"]["name"],
                          namespace=obj["metadata"].get("namespace"),
                          field_manager=FIELD_MANAGER, force_conflicts=True)


def apply_all(objs: list[dict], owner: dict) -> list[str]:
    skipped = []
    for obj in objs:
        kopf.adopt(obj, owner=owner)
        try:
            apply(obj)
        except ResourceNotFoundError:
            skipped.append(obj["kind"])   # e.g. ServiceMonitor CRD not installed
    return sorted(set(skipped))


def _crd_api(kind: str):
    return dyn().resources.get(api_version=f"{R.GROUP}/{R.VERSION}", kind=kind)


def set_cr_status(kind_plural: str, ns: str, name: str, status: dict) -> None:
    try:
        client.CustomObjectsApi().patch_namespaced_custom_object_status(
            R.GROUP, R.VERSION, ns, kind_plural, name, {"status": status})
    except client.ApiException as e:
        log.debug("status patch %s/%s failed: %s", kind_plural, name, e)


def delete_stale(kind: str, ns: str, cluster: str, keep: set[str]) -> list[str]:
    api = _crd_api(kind)
    gone = []
    for item in api.get(namespace=ns, label_selector=f"dualkey.io/cluster={cluster}").items:
        if item.metadata.name not in keep:
            api.delete(name=item.metadata.name, namespace=ns)
            gone.append(item.metadata.name)
    return gone


def delete_stale_reference(ns: str, cluster: str, keep: set[str]) -> None:
    apps, core = client.AppsV1Api(), client.CoreV1Api()
    for d in apps.list_namespaced_deployment(ns, label_selector=f"dualkey.io/cluster={cluster}").items:
        if d.metadata.name.startswith("dk-model-ref-") and d.metadata.name not in keep:
            apps.delete_namespaced_deployment(d.metadata.name, ns)
            try:
                core.delete_namespaced_service(d.metadata.name, ns)
            except client.ApiException:
                pass


def read_secret(ns: str, name: str, key: str) -> str:
    try:
        s = client.CoreV1Api().read_namespaced_secret(name, ns)
        return base64.b64decode((s.data or {}).get(key, "")).decode()
    except client.ApiException:
        return ""


def ensure_api_key(ns: str, owner: dict) -> None:
    v1 = client.CoreV1Api()
    try:
        v1.read_namespaced_secret(R.API_KEY_SECRET, ns)
        return
    except client.ApiException as e:
        if e.status != 404:
            raise
    body = {"apiVersion": "v1", "kind": "Secret",
            "metadata": {"name": R.API_KEY_SECRET, "namespace": ns}, "type": "Opaque",
            "data": {"DK_API_KEY": base64.b64encode(secrets.token_urlsafe(32).encode()).decode()}}
    kopf.adopt(body, owner=owner)
    v1.create_namespaced_secret(ns, body)


def list_pods_and_nodes(namespaces: list[str]) -> tuple[list[dict], list[dict]]:
    pods_api = dyn().resources.get(api_version="v1", kind="Pod")
    nodes_api = dyn().resources.get(api_version="v1", kind="Node")
    if namespaces:
        pods = [p.to_dict() for ns in namespaces for p in pods_api.get(namespace=ns).items]
    else:
        pods = [p.to_dict() for p in pods_api.get().items]
    nodes = [n.to_dict() for n in nodes_api.get().items]
    return pods, nodes


# ------------------------------------------------------------------ discovery

async def probe_servers(servers: list[D.Server], keys: dict[str, str]) -> dict[str, dict]:
    """GET /v1/models on every ready server to confirm it answers and learn its model."""
    async def one(http: httpx.AsyncClient, s: D.Server):
        key = keys["dualkey"] if s.managed else keys.get("discovery", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            r = await http.get(f"{s.endpoint}/v1/models", headers=headers)
        except httpx.HTTPError:
            return s.key, {"ok": False, "error": "not reachable from the dualkey namespace (check NetworkPolicy)"}
        if r.status_code in (401, 403):
            return s.key, {"ok": False, "error": "needs an API key: create secret dualkey-discovery-api-key"}
        if r.status_code != 200:
            return s.key, {"ok": False, "error": f"/v1/models returned {r.status_code}"}
        data = r.json().get("data") or []
        out = {"ok": True, "root": (data[0].get("root") if data else "") or "",
               "models": [d.get("id") for d in data]}
        # identity checks: vLLM release and how this server tokenizes a fixed text
        try:
            v = await http.get(f"{s.endpoint}/version", headers=headers)
            if v.status_code == 200:
                out["version"] = v.json().get("version", "")
        except (httpx.HTTPError, ValueError):
            pass
        try:
            model = s.served_name if s.served_name in out["models"] else (out["models"] or [None])[0]
            t = await http.post(f"{s.endpoint}/tokenize", headers=headers,
                                json={"model": model, "prompt": D.TOKENIZER_CANARY, "add_special_tokens": False})
            if t.status_code == 200:
                out["canary_tokens"] = t.json().get("tokens") or []
        except (httpx.HTTPError, ValueError):
            pass
        return s.key, out

    targets = [s for s in servers if s.ready and s.endpoint]
    async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as http:
        results = await asyncio.gather(*(one(http, s) for s in targets))
    return dict(results)


_revisions: dict[str, str] = {}


def pin_reference_revision(spec: dict) -> dict:
    """Resolve the reference model to an exact commit once, so every copy is identical."""
    model = dict(spec.get("model") or {})
    if model.get("revision"):
        return spec
    name = model.get("name", "ibm-granite/granite-3.3-2b-instruct")
    if name not in _revisions:
        try:
            from huggingface_hub import HfApi
            _revisions[name] = HfApi().model_info(name).sha or ""
        except Exception as e:
            log.info("could not resolve the revision of %s: %s", name, e)
            _revisions[name] = ""
    if _revisions[name]:
        model["revision"] = _revisions[name]
        return {**spec, "model": model}
    return spec


def runtime_groups(groups: list[D.GroupView], has_discovery_key: bool) -> list[dict]:
    out = []
    for g in groups:
        if g.phase != "Ready":
            continue
        out.append({"name": g.name, "model_id": g.model_id, "source": g.source,
                    "identity": g.identity,
                    "members": [{"name": m["name"], "vendor": m["vendor"], "endpoint": m["endpoint"],
                                 "node": m["node"], "model": m["model"], "build": m.get("build") or {},
                                 "auth": "dualkey" if m["managed"] else ("discovery" if has_discovery_key else "none")}
                                for m in g.ready_members()]})
    return out


def describe_changes(prev: dict, now: dict, prev_ref: dict, ref: dict, n_servers: int,
                     prev_servers: int | None, first: bool) -> list[str]:
    """Human-readable lines for the dashboard's discovery timeline."""
    msgs = []
    if first:
        msgs.append(f"First scan found {n_servers} vLLM model server(s) on GPUs")
    elif prev_servers is not None and n_servers != prev_servers:
        msgs.append(f"Model servers on GPUs: {prev_servers} -> {n_servers}")
    for name, g in now.items():
        members = ", ".join(g["memberNames"]) or "none"
        old = prev.get(name)
        label = "reference group" if g["source"] == "reference" else "group"
        if old is None:
            if g["phase"] == "Ready":
                msgs.append(f"Formed {label} {name} for {g['modelId']} with {g['ready']} ready members: {members}")
            else:
                msgs.append(f"Found {label} {name} for {g['modelId']}: {g['message']}")
            continue
        if old["phase"] != g["phase"]:
            if g["phase"] == "Ready":
                msgs.append(f"{label.capitalize()} {name} is ready: {g['ready']} members ({members}); checking starts")
            else:
                msgs.append(f"{label.capitalize()} {name} paused: {g['message']}")
        joined = set(g["memberNames"]) - set(old["memberNames"])
        left = set(old["memberNames"]) - set(g["memberNames"])
        if joined:
            msgs.append(f"{', '.join(sorted(joined))} joined {name}")
        if left:
            msgs.append(f"{', '.join(sorted(left))} left {name}")
    for name in set(prev) - set(now):
        msgs.append(f"Group {name} no longer exists")
    old_nodes = sorted(p["node"] for p in prev_ref.get("placements", []))
    new_nodes = sorted(p["node"] for p in ref.get("placements", []))
    if new_nodes != old_nodes or (first and ref.get("reason")):
        msgs.append(ref.get("reason") or ("Reference model removed" if not new_nodes else ""))
    elif ref.get("reason") != prev_ref.get("reason") and not new_nodes:
        msgs.append(ref["reason"])
    return [m for m in msgs if m]


def overall_phase(groups: list[D.GroupView], placements: list[dict], n_servers: int) -> str:
    if any(g.phase == "Ready" for g in groups):
        return "Checking"
    if placements:
        return "DeployingReference"
    return "Discovering" if n_servers == 0 else "Waiting"


async def run_discovery(spec: dict, name: str, ns: str, status: dict, body: dict) -> dict:
    dspec = spec.get("discovery") or {}
    rspec = spec.get("reference") or {}
    min_members = int(dspec.get("minMembers", 3))
    exclude = [e for e in (dspec.get("excludeNamespaces") or DEFAULT_EXCLUDE) if e]

    pods, nodes = await asyncio.to_thread(list_pods_and_nodes, dspec.get("namespaces") or [])
    servers, skipped = D.discover_servers(pods, tuple(exclude), nodes)
    keys = {"dualkey": await asyncio.to_thread(read_secret, ns, R.API_KEY_SECRET, "DK_API_KEY"),
            "discovery": await asyncio.to_thread(read_secret, ns, "dualkey-discovery-api-key",
                                                 "DK_DISCOVERY_API_KEY")}
    D.apply_probes(servers, await probe_servers(servers, keys))
    groups = D.form_groups(servers, min_members)

    node_names = {n["metadata"]["name"] for n in nodes}
    prev_disc = status.get("discovery") or {}
    current = [p for p in (prev_disc.get("reference") or {}).get("placements", []) if p["node"] in node_names]
    free = D.free_gpu_nodes(nodes, pods)
    placements, reason = D.plan_reference(groups, free, current,
                                          enabled=rspec.get("enabled", True),
                                          want=int(rspec.get("gpus", 3)))
    rgroups = runtime_groups(groups, bool(keys["discovery"]))
    if placements:
        spec = await asyncio.to_thread(pin_reference_revision, spec)

    # ---- apply everything
    objs = R.all_objects(spec, name, ns, rgroups, placements)
    for g in groups:
        objs.append(R.comparison_group(name, ns, g.to_dict()))
    records = []
    for rg in rgroups:
        for m in rg["members"]:
            records.append(R.gpu_health_record(name, ns, {**m, "group": rg["name"]}))
    objs += records
    skipped_kinds = await asyncio.to_thread(apply_all, objs, body)
    for g in groups:
        await asyncio.to_thread(set_cr_status, "comparisongroups", ns, g.name, {
            "phase": g.phase, "vendors": g.vendors, "message": g.message,
            "readyMembers": len(g.ready_members()), "members": g.members,
            "identity": g.identity, "split": g.split,
            "lastScan": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    await asyncio.to_thread(delete_stale, "ComparisonGroup", ns, name, {g.name for g in groups})
    await asyncio.to_thread(delete_stale, "GPUHealthRecord", ns, name, {r["metadata"]["name"] for r in records})
    await asyncio.to_thread(delete_stale_reference, ns, name,
                            {R.model_service_name(R.reference_member_name(p["node"])) for p in placements})

    # ---- timeline
    summary = {g.name: {"modelId": g.model_id, "source": g.source, "phase": g.phase,
                        "ready": len(g.ready_members()), "message": g.message,
                        "memberNames": [m["name"] for m in g.ready_members()]} for g in groups}
    ref = {"placements": placements, "reason": reason}
    prev_groups = {g["name"]: g for g in prev_disc.get("groups", [])}
    msgs = describe_changes(prev_groups, summary, prev_disc.get("reference") or {}, ref, len(servers),
                            prev_disc.get("servers"), first=not prev_disc)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    events = (status.get("events") or [])
    events = events + [{"time": now, "message": m} for m in msgs]
    for m in msgs:
        kopf.info(body, reason="Discovery", message=m[:1000])
    if skipped_kinds:
        log.info("APIs not available, skipped: %s", skipped_kinds)
    return {
        "phase": overall_phase(groups, placements, len(servers)),
        "discovery": {
            "lastScan": now, "servers": len(servers),
            "groups": [{"name": n, **v} for n, v in summary.items()],
            "skipped": [{"namespace": s.namespace, "pod": s.pod, "node": s.node, "reason": s.reason}
                        for s in skipped][:25],
            "unready": [{"name": s.key, "pod": s.pod, "namespace": s.namespace, "reason": s.reason}
                        for s in servers if not s.ready][:25],
            "reference": ref,
        },
        "events": events[-MAX_EVENTS:],
        "message": msgs[-1] if msgs else (status.get("message") or ""),
    }


# ------------------------------------------------------------------ handlers

@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_):
    settings.persistence.finalizer = "dualkey.io/finalizer"
    settings.posting.level = logging.INFO


@kopf.on.create(R.GROUP, R.VERSION, "dualkeyclusters")
@kopf.on.update(R.GROUP, R.VERSION, "dualkeyclusters")
@kopf.on.resume(R.GROUP, R.VERSION, "dualkeyclusters")
def reconcile(spec, name, namespace, body, patch, **_):
    spec = dict(spec)
    problems = R.validate(spec)
    hard = [p for p in problems if not p.startswith("warning")]
    for p in problems:
        kopf.warn(body, reason="Validation", message=p)
    if hard:
        patch.status["phase"] = "Invalid"
        patch.status["message"] = "; ".join(hard)
        raise kopf.PermanentError("; ".join(hard))
    ensure_api_key(namespace, body)

    if R.mode(spec) == "discover":
        _last_scan[f"{namespace}/{name}"] = 0.0   # scan right away on the next timer tick
        patch.status["message"] = "discovery mode: scanning the cluster for model servers"
        return

    groups = R.manual_groups(spec)
    objs = R.all_objects(spec, name, namespace, groups)
    objs.append(R.comparison_group(name, namespace, {**groups[0], "source": "manual"}))
    objs += [R.gpu_health_record(name, namespace, {**g, "group": "manual"}) for g in spec.get("gpus", [])]
    skipped = apply_all(objs, body)
    set_cr_status("comparisongroups", namespace, "manual", {
        "phase": "Ready", "message": f"{len(groups[0]['members'])} GPUs listed in spec.gpus",
        "readyMembers": len(groups[0]["members"]),
        "vendors": sorted({m["vendor"] for m in groups[0]["members"]}),
        "members": [{**m, "ready": True} for m in groups[0]["members"]]})
    delete_stale("GPUHealthRecord", namespace, name, {g["name"] for g in spec.get("gpus", [])})
    if skipped:
        kopf.warn(body, reason="Skipped", message=f"API not available for: {skipped}")
    patch.status["phase"] = "Checking"
    patch.status["message"] = f"manual mode: {len(spec.get('gpus', []))} GPUs"


@kopf.timer(R.GROUP, R.VERSION, "dualkeyclusters", interval=10, initial_delay=5)
async def discovery_timer(spec, name, namespace, status, body, patch, **_):
    spec = dict(spec)
    if R.mode(spec) != "discover":
        return
    key = f"{namespace}/{name}"
    every = int((spec.get("discovery") or {}).get("intervalSeconds", 30))
    if time.time() - _last_scan.get(key, 0.0) < every:
        return
    _last_scan[key] = time.time()
    try:
        result = await run_discovery(spec, name, namespace, dict(status or {}), body)
    except Exception as e:
        log.exception("discovery failed")
        patch.status["message"] = f"discovery failed: {e}"
        return
    for k, v in result.items():
        patch.status[k] = v


def _cluster_spec(namespace: str) -> dict:
    items = _crd_api("DualKeyCluster").get(namespace=namespace).items
    return items[0].spec.to_dict() if items else {}


def _set_taint(node: str, member: str, present: bool) -> None:
    v1 = client.CoreV1Api()
    n = v1.read_node(node)
    taints = [t for t in (n.spec.taints or []) if t.key != R.SUSPECT_TAINT]
    if present:
        taints.append(client.V1Taint(key=R.SUSPECT_TAINT, value=member[:63], effect="NoSchedule"))
    v1.patch_node(node, {"spec": {"taints": [t.to_dict() for t in taints]}})


# ------------------------------------------------------------------ incidents

INCIDENTS = "gpuintegrityincidents"


def _custom():
    dyn()
    return client.CustomObjectsApi()


def list_incidents(ns: str, member: str | None = None) -> list[dict]:
    items = _custom().list_namespaced_custom_object(R.GROUP, R.VERSION, ns, INCIDENTS)["items"]
    return [i for i in items if member is None or (i.get("spec") or {}).get("gpu") == member]


def open_incident_for(ns: str, member: str) -> dict | None:
    live = [i for i in list_incidents(ns, member) if (i.get("status") or {}).get("phase") in RM.OPEN_PHASES]
    return sorted(live, key=lambda i: i["metadata"]["creationTimestamp"])[-1] if live else None


def patch_incident_status(ns: str, name: str, status: dict, action: str | None = None, result: str = "") -> None:
    api = _custom()
    if action:
        cur = api.get_namespaced_custom_object(R.GROUP, R.VERSION, ns, INCIDENTS, name)
        actions = ((cur.get("status") or {}).get("actions") or []) + [
            {"time": RM.now_iso(), "action": action, "result": result[:500]}]
        status = {**status, "actions": actions[-40:]}
    api.patch_namespaced_custom_object_status(R.GROUP, R.VERSION, ns, INCIDENTS, name, {"status": status})


def set_node_condition(node: str, active: bool, reason: str, message: str) -> None:
    client.CoreV1Api().patch_node_status(node, {"status": {"conditions": [RM.condition(active, reason, message)]}})


def read_scorer_and_results(ns: str, group: str) -> tuple[dict, list[dict]]:
    """The Detector's saved state and the latest check results, straight from the queue."""
    import json as _json
    import redis as _redis
    r = _redis.Redis.from_url(f"redis://dualkey-redis.{ns}.svc:6379/0", socket_timeout=3)
    raw = r.get("dk:scorer")
    scorer = (_json.loads(raw).get("groups") or {}).get(group, {}) if raw else {}
    results = []
    for _id, fields in r.xrevrange("dk:results", count=400):
        try:
            results.append(_json.loads(fields.get(b"data") or fields.get("data")))
        except Exception:
            pass
    return scorer, results


def move_pods(node: str, ns: str, pol: dict, evict: bool) -> str:
    """drain: Eviction API (respects PodDisruptionBudgets). evict: delete right away."""
    v1 = client.CoreV1Api()
    pods = client.ApiClient().sanitize_for_serialization(
        v1.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node}").items)
    move, kept = RM.pods_to_move(pods, node, ns, pol["skipNamespaces"])
    done, blocked = [], []
    for pns, name in move:
        try:
            if evict:
                v1.delete_namespaced_pod(name, pns, grace_period_seconds=pol["gracePeriodSeconds"])
            else:
                body = client.V1Eviction(metadata=client.V1ObjectMeta(name=name, namespace=pns),
                                         delete_options=client.V1DeleteOptions(
                                             grace_period_seconds=pol["gracePeriodSeconds"]))
                v1.create_namespaced_pod_eviction(name, pns, body)
            done.append(f"{pns}/{name}")
        except client.ApiException as e:
            blocked.append(f"{pns}/{name} ({'blocked by PodDisruptionBudget' if e.status == 429 else e.status})")
    parts = [f"{'deleted' if evict else 'evicted'} {len(done)} GPU pod(s)" + (f": {', '.join(done)}" if done else "")]
    if blocked:
        parts.append(f"not moved: {', '.join(blocked)}")
    if kept:
        parts.append("kept: " + ", ".join(f"{a}/{b} ({why})" for a, b, why in kept))
    return "; ".join(parts)


def contain(ns: str, inc: str, member: str, node: str, pol: dict, summary: str) -> str:
    """Carry out the response policy on the node. Returns the incident phase."""
    action = pol["action"]
    if node and (pol["nodeCondition"] or action == "handoff"):
        try:
            set_node_condition(node, True, "SilentDataCorruptionSuspected",
                               f"Dual-Key incident {inc}: {member} {summary}")
            patch_incident_status(ns, inc, {}, f"set node condition {RM.CONDITION}=True", "ok")
        except client.ApiException as e:
            patch_incident_status(ns, inc, {}, f"set node condition {RM.CONDITION}=True", f"failed: {e.status}")
    if not node or action in ("alert", "handoff"):
        phase = "HandedOff" if action == "handoff" else "Open"
        note = ("waiting for the vendor's remediation (NVSentinel / AMD GPU Operator) to act on the node condition"
                if action == "handoff" else "alert only: no change to the node")
        patch_incident_status(ns, inc, {"phase": phase}, action, note)
        return phase
    try:
        _set_taint(node, member, True)
        patch_incident_status(ns, inc, {}, "cordon", f"tainted {node} {R.SUSPECT_TAINT}:NoSchedule (running pods stay)")
    except client.ApiException as e:
        patch_incident_status(ns, inc, {}, "cordon", f"failed: {e.status} {e.reason}")
    if action in ("drain", "evict"):
        try:
            patch_incident_status(ns, inc, {}, action, move_pods(node, ns, pol, evict=action == "evict"))
        except client.ApiException as e:
            patch_incident_status(ns, inc, {}, action, f"failed: {e.status} {e.reason}")
    patch_incident_status(ns, inc, {"phase": "Contained"})
    return "Contained"


def _cluster_obj(ns: str) -> dict | None:
    items = _crd_api("DualKeyCluster").get(namespace=ns).items
    return items[0].to_dict() if items else None


def open_incident(ns: str, member: str, spec: dict, ghr_status: dict) -> str:
    cspec = _cluster_spec(ns)
    pol = RM.policy(cspec, R.mode(cspec))
    node, vendor, group = spec.get("node", ""), spec.get("vendor", ""), spec.get("group", "")
    det = cspec.get("detector") or {}
    thresholds = {"suspect": det.get("suspectZ", 3), "quarantine": det.get("quarantineZ", 5)}

    existing = open_incident_for(ns, member)
    if existing:   # quarantined again while an incident is open (e.g. validation failed)
        name = existing["metadata"]["name"]
        patch_incident_status(ns, name, {"phase": "ValidationFailed"} if existing["status"].get("phase") == "Validating" else {},
                              "quarantined again", f"score {ghr_status.get('score')}")
        contain(ns, name, member, node, pol, "disagreed with its partners again")
        return name

    try:
        scorer, results = read_scorer_and_results(ns, group)
    except Exception as e:
        log.warning("could not read evidence from the queue: %s", e)
        scorer, results = {}, []
    identity = {}
    try:
        cg = _custom().get_namespaced_custom_object(R.GROUP, R.VERSION, ns, "comparisongroups", group)
        identity = (cg.get("spec") or {}).get("identity") or {}
    except Exception:
        pass
    evidence = RM.build_evidence(member, vendor, scorer, results, ghr_status, thresholds, identity)
    boot_id = ""
    if node:
        try:
            boot_id = client.CoreV1Api().read_node(node).status.node_info.boot_id
        except Exception:
            pass
    name = RM.incident_name(node or member)
    body = {"apiVersion": f"{R.GROUP}/{R.VERSION}", "kind": "GPUIntegrityIncident",
            "metadata": {"name": name, "namespace": ns,
                         "labels": {"dualkey.io/gpu": D.slug(member), "dualkey.io/node": D.slug(node or "none")}},
            "spec": {"gpu": member, "node": node, "vendor": vendor, "group": group, "release": False}}
    owner = _cluster_obj(ns)   # incidents outlive the member's record; they go with the cluster
    if owner:
        kopf.adopt(body, owner=owner)
    _custom().create_namespaced_custom_object(R.GROUP, R.VERSION, ns, INCIDENTS, body)
    patch_incident_status(ns, name, {"phase": "Open", "openedAt": RM.now_iso(), "policy": pol["action"],
                                     "evidence": evidence, "bootID": boot_id,
                                     "message": f"{member} {evidence['summary']}"},
                          "opened", evidence["summary"])
    contain(ns, name, member, node, pol, evidence["summary"])
    return name


@kopf.on.field(R.GROUP, R.VERSION, "gpuhealthrecords", field="status.state")
def on_state(old, new, spec, status, namespace, body, **_):
    if new == old or not new:
        return
    member = spec.get("gpu")
    if new == "Quarantined":
        kopf.warn(body, reason="Quarantined",
                  message=f"{member} disagrees with all its partners; Dual-Key stops sending it work")
        try:
            inc = open_incident(namespace, member, dict(spec), dict(status or {}))
            kopf.warn(body, reason="Incident", message=f"incident {inc} opened for {member}")
        except Exception as e:
            log.exception("could not open incident for %s", member)
            kopf.warn(body, reason="IncidentFailed", message=str(e)[:500])
    elif new == "Suspect":
        kopf.warn(body, reason="Suspect", message=f"{member} score is rising; confirming")


@kopf.on.field(R.GROUP, R.VERSION, "gpuhealthrecords", field="spec.release")
def on_release(new, spec, body, **_):
    if not new:
        return
    node, member = spec.get("node"), spec.get("gpu")
    if node:
        try:
            _set_taint(node, member, False)
        except client.ApiException as e:
            log.warning("could not remove taint from %s: %s", node, e)
    kopf.info(body, reason="Released", message=f"{member} released; scoring restarts")


def release_incident(ns: str, name: str, spec: dict, why: str) -> None:
    member, node = spec.get("gpu"), spec.get("node")
    if node:
        try:
            _set_taint(node, member, False)
        except client.ApiException as e:
            log.warning("could not remove taint from %s: %s", node, e)
        try:
            set_node_condition(node, False, "ReleasedForValidation", f"incident {name} released: {why}")
        except client.ApiException as e:
            log.warning("could not clear node condition on %s: %s", node, e)
    start_checks = 0
    try:
        ghr = _custom().get_namespaced_custom_object(R.GROUP, R.VERSION, ns, "gpuhealthrecords", member)
        start_checks = int((ghr.get("status") or {}).get("checks") or 0)
        _custom().patch_namespaced_custom_object(R.GROUP, R.VERSION, ns, "gpuhealthrecords", member,
                                                 {"spec": {"release": True}})
    except client.ApiException as e:
        log.warning("could not release record %s: %s", member, e)
    patch_incident_status(ns, name, {"phase": "Validating",
                                     "validation": {"startedAt": RM.now_iso(), "startedTs": time.time(),
                                                    "startChecks": start_checks, "progress": "starting"}},
                          "released", f"{why}; taint removed, re-checking {member} against its partners")


@kopf.on.field(R.GROUP, R.VERSION, INCIDENTS, field="spec.release")
def on_incident_release(new, spec, name, namespace, status, **_):
    if not new or (status or {}).get("phase") not in ("Open", "Contained", "HandedOff", "ValidationFailed",
                                                       "ValidationTimedOut"):
        return
    release_incident(namespace, name, dict(spec), "released by an administrator")


@kopf.on.create(R.GROUP, R.VERSION, INCIDENTS)
async def on_incident_created(spec, name, namespace, **_):
    """Gather the vendor's view of the node: telemetry snapshot and its own diagnostic."""
    node, vendor = spec.get("node"), spec.get("vendor", "nvidia")
    if not node:
        return
    cspec = await asyncio.to_thread(_cluster_spec, namespace)
    pol = RM.policy(cspec, R.mode(cspec))
    await asyncio.to_thread(collect_telemetry, namespace, name, node)
    dcfg = pol["diagnostics"].get(vendor) or {}
    if dcfg.get("enabled", True):
        await asyncio.to_thread(run_diagnostic, namespace, name, node, vendor, dcfg)


def collect_telemetry(ns: str, name: str, node: str) -> None:
    from . import vendor as V
    v1 = client.CoreV1Api()
    raw = client.ApiClient().sanitize_for_serialization(v1.list_node().items)
    snap = V.snapshot(node, raw, v1)
    verdict = snap.get("verdict", "unknown")
    patch_incident_status(ns, name, {"telemetry": snap}, "vendor telemetry snapshot",
                          f"{snap.get('source') or 'vendor metrics'}: " +
                          ("errors reported: " + "; ".join(snap.get("problems") or []) if verdict == "error"
                           else "no errors reported" if verdict == "ok" else snap.get("status", "no data")))


def run_diagnostic(ns: str, name: str, node: str, vendor: str, dcfg: dict) -> None:
    """Run the vendor's diagnostic inside the vendor's own pod on that node."""
    from kubernetes.stream import stream
    v1 = client.CoreV1Api()
    pod = None
    for sel in dcfg.get("podSelectors") or []:
        items = v1.list_pod_for_all_namespaces(label_selector=sel, field_selector=f"spec.nodeName={node}").items
        pod = next((p for p in items if p.status.phase == "Running"), None)
        if pod:
            break
    base = {"tool": dcfg.get("tool"), "command": " ".join(dcfg.get("command") or []), "startedAt": RM.now_iso()}
    if not pod:
        patch_incident_status(ns, name, {"diagnostics": {**base, "result": "Unavailable",
                                                         "summary": "no vendor diagnostic pod found on the node"}},
                              "vendor diagnostic", "unavailable: no vendor pod on the node")
        return
    container = dcfg.get("container") or pod.spec.containers[0].name
    try:
        out = stream(v1.connect_get_namespaced_pod_exec, pod.metadata.name, pod.metadata.namespace,
                     command=dcfg["command"], container=container, stderr=True, stdout=True, stdin=False,
                     tty=False, _request_timeout=int(dcfg.get("timeoutSeconds", 300)))
    except Exception as e:
        patch_incident_status(ns, name, {"diagnostics": {**base, "result": "Error", "summary": str(e)[:300]}},
                              "vendor diagnostic", f"could not run: {str(e)[:200]}")
        return
    result, summary = RM.parse_diagnostic(vendor, out or "")
    patch_incident_status(ns, name, {"diagnostics": {**base, "result": result, "summary": summary,
                                                     "ranIn": f"{pod.metadata.namespace}/{pod.metadata.name}",
                                                     "finishedAt": RM.now_iso(), "output": (out or "")[-4000:]}},
                          "vendor diagnostic", f"{result}: {summary}")


@kopf.timer(R.GROUP, R.VERSION, INCIDENTS, interval=30, initial_delay=15)
def incident_timer(spec, status, name, namespace, **_):
    """Follow up on open incidents: validation after release, auto-release after a vendor repair."""
    status = dict(status or {})
    phase = status.get("phase")
    member, node = spec.get("gpu"), spec.get("node")
    cspec = _cluster_spec(namespace)
    pol = RM.policy(cspec, R.mode(cspec))
    if phase == "HandedOff" and node and not status.get("vendorAction"):
        try:   # record when the vendor's remediation (e.g. NVSentinel) acts on the node
            n = client.CoreV1Api().read_node(node)
        except client.ApiException:
            n = None
        if n is not None and n.spec.unschedulable:
            labels = n.metadata.labels or {}
            who = ("NVSentinel" if any(k.startswith("dgxc.nvidia.com/nvsentinel") for k in labels)
                   else "AMD GPU Operator" if any("amd" in t.key for t in (n.spec.taints or []))
                   else "the vendor's remediation")
            state = labels.get("dgxc.nvidia.com/nvsentinel-state", "")
            patch_incident_status(namespace, name, {"vendorAction": {"cordonedAt": RM.now_iso(), "by": who,
                                                                     "state": state}},
                                  "vendor remediation", f"node cordoned by {who}" + (f" (state: {state})" if state else ""))
    if phase in ("Contained", "HandedOff") and pol["autoReleaseAfterReboot"] and node and status.get("bootID"):
        try:
            boot = client.CoreV1Api().read_node(node).status.node_info.boot_id
        except client.ApiException:
            return
        if boot and boot != status["bootID"]:
            release_incident(namespace, name, dict(spec), "node was rebooted by the vendor's remediation")
        return
    if phase != "Validating":
        return
    val = status.get("validation") or {}
    try:
        ghr = _custom().get_namespaced_custom_object(R.GROUP, R.VERSION, namespace, "gpuhealthrecords", member)
    except client.ApiException:
        return
    det = cspec.get("detector") or {}
    decision, why = RM.validation_decision(
        ghr.get("status") or {}, val.get("startChecks", 0), int(det.get("minPartners", 2)),
        pol["validation"]["checksPerPartner"], float(det.get("suspectZ", 3)),
        float(val.get("startedTs") or time.time()), time.time(), pol["validation"]["timeoutMinutes"])
    if decision == "wait":
        patch_incident_status(namespace, name, {"validation": {**val, "progress": why}})
    elif decision == "passed":
        patch_incident_status(namespace, name, {"phase": "Resolved", "resolvedAt": RM.now_iso(),
                                                "validation": {**val, "progress": why, "result": "Passed"}},
                              "validation passed", why)
    elif decision == "timeout":
        patch_incident_status(namespace, name, {"phase": "ValidationTimedOut",
                                                "validation": {**val, "progress": why, "result": "TimedOut"}},
                              "validation timed out", why)
    # "failed": the record goes back to Quarantined and on_state re-contains the node
