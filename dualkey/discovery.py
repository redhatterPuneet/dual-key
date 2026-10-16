"""Discovery: find the model servers already running on GPUs and group them.

Pure functions over raw Kubernetes JSON (camelCase dicts), so they can be unit
tested without a cluster. The operator feeds them the cluster's pods and nodes.

Steps:
  1. discover_servers(): every running pod that uses a GPU is looked at.
     vLLM pods become candidates; others are listed as skipped, with a reason.
  2. probe results (from the operator calling /v1/models) mark which servers
     are reachable.
  3. form_groups(): servers running the same model build form one comparison
     group. "Same build" means the same model id, revision, weights checksum (when
     known), precision, quantization, tensor-parallel size, tokenizer and vLLM
     version. Servers that share a model id but differ in any of these are kept in
     separate groups, with the reason. A group needs >= 3 ready members.
  4. plan_reference(): if no group is big enough, pick free GPU nodes to run
     the prebuilt reference model on.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

GPU_RESOURCES = {"nvidia.com/gpu": "nvidia", "amd.com/gpu": "amd"}
MODEL_ID_LABEL = "dualkey.io/model-id"
REVISION_ANNOTATION = "dualkey.io/model-revision"
CHECKSUM_ANNOTATION = "dualkey.io/model-sha256"
TOKENIZER_CANARY = "Dual-Key tokenizer check: The quick brown fox jumps over 13 lazy dogs. ¿Qué tal? 日本語 ✓"
DRIVER_LABELS = {"nvidia": ("nvidia.com/cuda.driver-version.full", "nvidia.com/cuda.driver.major"),
                 "amd": ("amd.com/gpu.driver-version", "beta.amd.com/gpu.driver-version")}
MANAGED_LABEL = "dualkey.io/cluster"
KSERVE_SOURCE_ANNOTATIONS = ("internal.serving.kserve.io/storage-initializer-sourceuri",
                             "serving.kserve.io/storage-initializer-sourceuri")
LOCAL_PATH_PREFIXES = ("/mnt/models", "/models", "/data", "/opt", "/tmp", "/root", "/home")


def slug(text: str, maxlen: int = 63) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return (s[:maxlen].strip("-")) or "x"


def short_hash(text: str, n: int = 6) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


@dataclass
class Server:
    key: str                 # stable member name
    namespace: str
    pod: str
    owner: str
    node: str
    vendor: str
    gpus: int
    model_id: str
    served_name: str
    dtype: str
    quantization: str
    tensor_parallel: int
    endpoint: str
    ready: bool
    managed: bool            # deployed by Dual-Key (reference model)
    reason: str = ""
    revision: str = ""       # model revision (git commit / tag), when known
    model_sha: str = ""      # weights checksum, when the deployment declares one
    tokenizer_hash: str = "" # hash of how the server tokenizes a fixed canary text
    vllm_version: str = ""   # from /version, local build suffix removed
    image: str = ""          # vLLM container image
    image_digest: str = ""   # sha256 of the image actually running
    driver: str = ""         # GPU driver version on the node

    def identity(self) -> dict:
        return {"modelId": self.model_id, "revision": self.revision, "modelSha": self.model_sha,
                "dtype": self.dtype, "quantization": self.quantization, "tp": self.tensor_parallel,
                "tokenizer": self.tokenizer_hash, "vllm": self.vllm_version}

    def build(self) -> dict:
        return {"vllm": self.vllm_version, "image": self.image, "imageDigest": self.image_digest,
                "driver": self.driver}


@dataclass
class Skipped:
    namespace: str
    pod: str
    node: str
    reason: str


@dataclass
class GroupView:
    name: str
    model_id: str
    dtype: str
    quantization: str
    source: str              # discovered | reference
    phase: str               # Ready | TooSmall
    members: list = field(default_factory=list)     # list of dicts
    vendors: list = field(default_factory=list)
    message: str = ""
    identity: dict = field(default_factory=dict)    # what every member shares
    split: list = field(default_factory=list)       # why look-alike servers were kept apart

    def ready_members(self) -> list[dict]:
        return [m for m in self.members if m.get("ready")]

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ helpers

def _tokens(container: dict) -> list[str]:
    toks = []
    for part in (container.get("command") or []) + (container.get("args") or []):
        # args may be a single shell string, e.g. ["-c", "vllm serve x --port 8000"]
        toks.extend(str(part).split())
    return toks


def is_vllm(container: dict) -> bool:
    toks = _tokens(container)
    joined = " ".join(toks)
    if "vllm.entrypoints" in joined:
        return True
    for i, t in enumerate(toks):
        if t.endswith("vllm") and i + 1 < len(toks) and toks[i + 1] == "serve":
            return True
    image = str(container.get("image", "")).lower()
    return "vllm" in image and ("--model" in joined or " serve " in f" {joined} ")


def _flag(toks: list[str], name: str) -> Optional[str]:
    for i, t in enumerate(toks):
        if t == name and i + 1 < len(toks):
            return toks[i + 1]
        if t.startswith(name + "="):
            return t.split("=", 1)[1]
    return None


def parse_vllm_args(container: dict) -> dict:
    toks = _tokens(container)
    model = _flag(toks, "--model")
    if not model:
        for i, t in enumerate(toks):
            if t == "serve" and i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                model = toks[i + 1]
                break
    port = _flag(toks, "--port")
    if not port:
        ports = container.get("ports") or []
        port = str(ports[0]["containerPort"]) if ports else "8000"
    return {
        "model": model or "",
        "served_name": _flag(toks, "--served-model-name") or (model or ""),
        "dtype": (_flag(toks, "--dtype") or "auto").lower(),
        "quantization": (_flag(toks, "--quantization") or "none").lower(),
        "tensor_parallel": int(_flag(toks, "--tensor-parallel-size") or _flag(toks, "-tp") or 1),
        "revision": _flag(toks, "--revision") or "",
        "seed": _flag(toks, "--seed") or "",
        "port": int(port),
    }


def pod_gpus(pod: dict) -> tuple[str, int]:
    vendor, total = "", 0
    for c in pod.get("spec", {}).get("containers", []):
        res = c.get("resources") or {}
        for section in ("limits", "requests"):
            for k, v in (res.get(section) or {}).items():
                ven = GPU_RESOURCES.get(k) or ("nvidia" if k.startswith("nvidia.com/mig-") else "")
                if ven:
                    vendor = ven
                    total = max(total, int(str(v)))
    return vendor, total


def owner_name(pod: dict) -> str:
    md = pod.get("metadata", {})
    refs = md.get("ownerReferences") or []
    if refs:
        ref = refs[0]
        if ref.get("kind") == "ReplicaSet":
            # deployment-name-<hash>
            return ref["name"].rsplit("-", 1)[0]
        return ref["name"]
    return md.get("name", "")


def _is_local(path: str) -> bool:
    return not path or path.startswith(LOCAL_PATH_PREFIXES) or path.startswith("./")


def model_identity(pod: dict, args: dict, probe_root: str = "") -> str:
    md = pod.get("metadata", {})
    labels = md.get("labels") or {}
    if labels.get(MODEL_ID_LABEL):
        return labels[MODEL_ID_LABEL]
    ann = md.get("annotations") or {}
    for a in KSERVE_SOURCE_ANNOTATIONS:
        if ann.get(a):
            return ann[a]
    for c in pod.get("spec", {}).get("containers", []):
        for e in c.get("env") or []:
            if e.get("name") == "STORAGE_URI" and e.get("value"):
                return e["value"]
    if args.get("model") and not _is_local(args["model"]):
        return args["model"]
    if probe_root and not _is_local(probe_root):
        return probe_root
    return args.get("model") or probe_root or "unknown"


def normalize_vllm_version(v: str) -> str:
    """'0.9.2+rocm641' and '0.9.2+cu126' are the same vLLM release built for different GPUs."""
    v = (v or "").strip().lstrip("v")
    return v.split("+", 1)[0]


def tokenizer_hash(tokens: list) -> str:
    return hashlib.sha256(json.dumps(list(tokens or [])).encode()).hexdigest()[:12] if tokens else ""


def node_driver(node: dict | None, vendor: str) -> str:
    labels = ((node or {}).get("metadata") or {}).get("labels") or {}
    if vendor == "nvidia":
        if labels.get("nvidia.com/cuda.driver-version.full"):
            return labels["nvidia.com/cuda.driver-version.full"]
        parts = [labels.get(f"nvidia.com/cuda.driver.{k}") for k in ("major", "minor", "rev")]
        return ".".join(p for p in parts if p) if parts[0] else ""
    for key in DRIVER_LABELS.get(vendor, ()):
        if labels.get(key):
            return labels[key]
    return ""


def container_digest(pod: dict, container_name: str) -> str:
    for cs in (pod.get("status") or {}).get("containerStatuses") or []:
        if cs.get("name") == container_name:
            image_id = cs.get("imageID") or ""
            return image_id.split("@", 1)[1] if "@" in image_id else image_id
    return ""


def _pod_ready(pod: dict) -> bool:
    for cond in pod.get("status", {}).get("conditions") or []:
        if cond.get("type") == "Ready":
            return cond.get("status") == "True"
    return False


# ------------------------------------------------------------------ discovery

def discover_servers(pods: list[dict], exclude_namespaces=("openshift-", "kube-"),
                     nodes: list[dict] | None = None) -> tuple[list[Server], list[Skipped]]:
    servers, skipped, used = [], [], set()
    by_node = {(n.get("metadata") or {}).get("name"): n for n in nodes or []}
    for pod in pods:
        md, spec, status = pod.get("metadata", {}), pod.get("spec", {}), pod.get("status", {})
        ns, name = md.get("namespace", ""), md.get("name", "")
        if any(ns.startswith(p) for p in exclude_namespaces):
            continue
        vendor, count = pod_gpus(pod)
        if not vendor or count < 1:
            continue
        node = spec.get("nodeName", "")
        if status.get("phase") not in ("Running", "Pending"):
            continue
        vllm = next((c for c in spec.get("containers", []) if is_vllm(c)), None)
        if vllm is None:
            skipped.append(Skipped(ns, name, node, "uses a GPU but is not a vLLM server"))
            continue
        args = parse_vllm_args(vllm)
        ip = status.get("podIP", "")
        owner = owner_name(pod)
        base = slug(f"{ns}-{owner}-{node.split('.')[0]}", 56)
        key = base if base not in used else slug(f"{base}-{short_hash(name)}", 63)
        used.add(key)
        ready = bool(ip) and _pod_ready(pod)
        ann = md.get("annotations") or {}
        servers.append(Server(
            key=key, namespace=ns, pod=name, owner=owner, node=node, vendor=vendor, gpus=count,
            model_id=model_identity(pod, args), served_name=args["served_name"],
            dtype=args["dtype"], quantization=args["quantization"],
            tensor_parallel=args["tensor_parallel"],
            endpoint=f"http://{ip}:{args['port']}" if ip else "",
            ready=ready, managed=bool((md.get("labels") or {}).get(MANAGED_LABEL)),
            reason="" if ready else "pod not ready yet",
            revision=ann.get(REVISION_ANNOTATION) or args["revision"],
            model_sha=ann.get(CHECKSUM_ANNOTATION, ""),
            image=vllm.get("image", ""), image_digest=container_digest(pod, vllm.get("name", "")),
            driver=node_driver(by_node.get(node), vendor)))
    return servers, skipped


def apply_probes(servers: list[Server], probes: dict[str, dict]) -> None:
    """probes: server key -> {"ok", "root", "models", "error", "version", "canary_tokens"}"""
    for s in servers:
        p = probes.get(s.key)
        if p is None:
            continue
        if not p.get("ok"):
            s.ready = False
            s.reason = p.get("error") or "not reachable"
            continue
        if s.model_id in ("", "unknown") or _is_local(s.model_id):
            root = p.get("root", "")
            if root and not _is_local(root):
                s.model_id = root
        if p.get("models") and s.served_name not in p["models"]:
            s.served_name = p["models"][0]
        s.vllm_version = normalize_vllm_version(p.get("version", ""))
        s.tokenizer_hash = tokenizer_hash(p.get("canary_tokens") or [])


def base_fingerprint(s: Server) -> str:
    """What must match before two servers are even candidates: the declared model."""
    return f"{s.model_id}|{s.revision}|{s.model_sha}|{s.dtype}|{s.quantization}|tp{s.tensor_parallel}"


def fingerprint(s: Server) -> str:
    """Full identity: the declared model plus how the running server actually behaves."""
    return f"{base_fingerprint(s)}|tok:{s.tokenizer_hash}|vllm:{s.vllm_version}"


def group_name(model_id: str, dtype: str, fp: str) -> str:
    base = slug(model_id.rstrip("/").split("/")[-1], 40)
    return slug(f"{base}-{dtype}-{short_hash(fp)}", 63)


def _split_reasons(variants: dict[str, list[Server]]) -> list[str]:
    """Explain why servers of one model ended up in different groups."""
    reasons = []
    toks = {v[0].tokenizer_hash for v in variants.values()}
    vers = {v[0].vllm_version for v in variants.values()}
    if len(toks) > 1:
        reasons.append("tokenizer differs (" + ", ".join(sorted(t or "?" for t in toks)) + ")")
    if len(vers) > 1:
        reasons.append("vLLM version differs (" + ", ".join(sorted(v or "?" for v in vers)) + ")")
    return reasons


def form_groups(servers: list[Server], min_members: int = 3) -> list[GroupView]:
    # 1. candidates: same declared model
    by_base: dict[str, list[Server]] = {}
    for s in servers:
        by_base.setdefault(base_fingerprint(s), []).append(s)
    # 2. inside each, split by behaviour (tokenizer, vLLM version). Servers that could
    #    not be probed yet join the largest known variant, so they show up while starting.
    by_fp: dict[str, list[Server]] = {}
    splits: dict[str, list[str]] = {}
    for base, members in by_base.items():
        known = [m for m in members if m.tokenizer_hash or m.vllm_version]
        variants: dict[str, list[Server]] = {}
        for m in known:
            variants.setdefault(fingerprint(m), []).append(m)
        unknown = [m for m in members if m not in known]
        if variants:
            biggest = max(variants, key=lambda k: (len(variants[k]), k))
            variants[biggest] += unknown
        else:
            variants[f"{base}|tok:|vllm:"] = unknown
        reasons = _split_reasons(variants) if len(variants) > 1 else []
        for fp, ms in variants.items():
            by_fp[fp] = ms
            splits[fp] = reasons
    groups = []
    for fp, members in sorted(by_fp.items()):
        first = next((m for m in members if m.tokenizer_hash or m.vllm_version), members[0])
        ready = [m for m in members if m.ready]
        vendors = sorted({m.vendor for m in ready})
        phase = "Ready" if len(ready) >= min_members else "TooSmall"
        if phase == "Ready":
            msg = f"{len(ready)} ready members" + ("" if len(vendors) > 1 else " (single vendor)")
        else:
            msg = f"{len(ready)} of {min_members} ready members needed"
        if splits.get(fp):
            msg += "; kept apart from other servers of this model: " + "; ".join(splits[fp])
        groups.append(GroupView(
            name=group_name(first.model_id, first.dtype, fp), model_id=first.model_id,
            dtype=first.dtype, quantization=first.quantization,
            source="reference" if all(m.managed for m in members) else "discovered",
            phase=phase, vendors=vendors, message=msg, identity=first.identity(),
            split=splits.get(fp, []),
            members=[{"name": m.key, "vendor": m.vendor, "node": m.node, "namespace": m.namespace,
                      "pod": m.pod, "endpoint": m.endpoint, "model": m.served_name,
                      "ready": m.ready, "reason": m.reason, "managed": m.managed,
                      "build": m.build()}
                     for m in sorted(members, key=lambda x: x.key)]))
    return groups


# ------------------------------------------------------------------ fallback

def free_gpu_nodes(nodes: list[dict], pods: list[dict]) -> dict[str, dict]:
    """node -> {"vendor": .., "free": n} for nodes with at least one unallocated GPU."""
    used: dict[tuple, int] = {}
    for pod in pods:
        if pod.get("status", {}).get("phase") in ("Succeeded", "Failed"):
            continue
        node = pod.get("spec", {}).get("nodeName")
        if not node:
            continue
        for c in pod.get("spec", {}).get("containers", []):
            res = (c.get("resources") or {}).get("limits") or {}
            for k, v in res.items():
                if k in GPU_RESOURCES:
                    used[(node, k)] = used.get((node, k), 0) + int(str(v))
    out = {}
    for n in nodes:
        name = n.get("metadata", {}).get("name", "")
        spec = n.get("spec", {})
        if spec.get("unschedulable"):
            continue
        if any(t.get("key") == "dualkey.io/suspect" for t in spec.get("taints") or []):
            continue
        alloc = n.get("status", {}).get("allocatable") or {}
        for res, vendor in GPU_RESOURCES.items():
            total = int(str(alloc.get(res, "0")))
            free = total - used.get((name, res), 0)
            if free > 0:
                out[name] = {"vendor": vendor, "free": free}
    return out


def plan_reference(groups: list[GroupView], free: dict[str, dict], current: list[dict],
                   enabled: bool = True, want: int = 3) -> tuple[list[dict], str]:
    """Decide where the reference model should run.

    current: reference placements already made (kept, so the plan does not flap).
    Returns (placements [{"node", "vendor"}], human-readable reason).
    """
    if not enabled:
        return [], "reference model disabled"
    if any(g.phase == "Ready" and g.source == "discovered" for g in groups):
        if current:
            return current, "discovered group is ready; keeping existing reference model"
        return [], "a discovered group is ready; no reference model needed"
    if len(current) >= want:
        return current, "reference model placed"
    chosen = list(current)
    taken = {p["node"] for p in chosen}
    by_vendor: dict[str, list[str]] = {}
    for node, info in sorted(free.items()):
        if node not in taken:
            by_vendor.setdefault(info["vendor"], []).append(node)
    # round-robin across vendors so the reference group mixes NVIDIA and AMD when possible
    while len(chosen) < want and any(by_vendor.values()):
        for vendor in sorted(by_vendor, key=lambda v: (len([c for c in chosen if c["vendor"] == v]), v)):
            if by_vendor[vendor] and len(chosen) < want:
                chosen.append({"node": by_vendor[vendor].pop(0), "vendor": vendor})
    if len(chosen) < want:
        return current, f"not enough free GPUs for the reference model (found {len(chosen)}, need {want})"
    vendors = sorted({c["vendor"] for c in chosen})
    return chosen, (f"no discovered group has {want} ready members; deploying the reference model on "
                    f"{', '.join(c['node'] for c in chosen)} ({'/'.join(vendors)})")
