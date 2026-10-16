"""Incidents and the response to them.

When a member is Quarantined the operator opens a GPUIntegrityIncident:

  Open -> Contained (or HandedOff) -> Validating -> Resolved
                                         |-> ValidationFailed (re-contained)
                                         |-> ValidationTimedOut

The incident records the evidence (pair scores, recent checks, learned normal,
model identity and build), the vendor telemetry at that moment, the result of the
vendor's own diagnostic on that node, and every action taken.

Response policy (DualKeyCluster spec.remediation.action), all acting on the NODE,
because that is what OpenShift schedules:
  alert    record the incident and set the node condition only
  cordon   taint the node dualkey.io/suspect:NoSchedule (no new pods; running ones stay)
  drain    cordon, then evict GPU pods through the Eviction API (respects PodDisruptionBudgets)
  evict    cordon, then delete GPU pods immediately (ignores PodDisruptionBudgets)
  handoff  set the node condition and let the vendor's remediation act on it
           (NVIDIA NVSentinel, AMD GPU Operator auto-remediation); see deploy/integrations/

This module holds the decisions as pure functions; operator.py does the API calls.
"""
from __future__ import annotations

import json
import re
import time

CONDITION = "DualKeyInferenceIntegrity"
ACTIONS = ("alert", "cordon", "drain", "evict", "handoff")
OPEN_PHASES = ("Open", "Contained", "HandedOff", "Validating", "ValidationFailed")

DEFAULT_DIAGNOSTICS = {
    # Runs inside the vendor's own pod on that node (no new privileged pod needed).
    "nvidia": {"enabled": True, "tool": "NVIDIA DCGM diagnostics (dcgmi diag)",
               "podSelectors": ["app=nvidia-dcgm", "app=nvidia-dcgm-exporter"],
               "command": ["dcgmi", "diag", "-r", "1", "-j"], "timeoutSeconds": 300},
    "amd": {"enabled": True, "tool": "AMD SMI ECC / RAS query",
            "podSelectors": ["app.kubernetes.io/name=metrics-exporter", "app=metrics-exporter"],
            "command": ["amd-smi", "metric", "--ecc", "--json"], "timeoutSeconds": 120},
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def policy(cspec: dict, mode: str) -> dict:
    rem = cspec.get("remediation") or {}
    action = rem.get("action")
    if action not in ACTIONS:
        if rem.get("taintNodes") is True:
            action = "cordon"
        elif rem.get("taintNodes") is False:
            action = "alert"
        else:   # never act on customer nodes unless asked
            action = "cordon" if mode == "manual" else "alert"
    val = rem.get("validation") or {}
    diag = {v: {**DEFAULT_DIAGNOSTICS[v], **((rem.get("diagnostics") or {}).get(v) or {})}
            for v in DEFAULT_DIAGNOSTICS}
    return {
        "action": action,
        "nodeCondition": bool(rem.get("nodeCondition", True)),
        "gracePeriodSeconds": int(rem.get("gracePeriodSeconds", 30)),
        "skipNamespaces": list(rem.get("skipNamespaces") or ["openshift-", "kube-"]),
        "autoReleaseAfterReboot": bool(rem.get("autoReleaseAfterReboot", action == "handoff")),
        "validation": {"checksPerPartner": int(val.get("checksPerPartner", 20)),
                       "timeoutMinutes": int(val.get("timeoutMinutes", 30))},
        "diagnostics": diag,
    }


def incident_name(node: str, ts: float | None = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time() if ts is None else ts))
    short = re.sub(r"[^a-z0-9-]+", "-", node.split(".")[0].lower()).strip("-")[:40] or "gpu"
    return f"{short}-{stamp}"


def condition(active: bool, reason: str, message: str) -> dict:
    t = now_iso()
    return {"type": CONDITION, "status": "True" if active else "False", "reason": reason,
            "message": message[:1000], "lastHeartbeatTime": t, "lastTransitionTime": t}


def _gpu_pod(pod: dict) -> bool:
    for c in (pod.get("spec") or {}).get("containers") or []:
        for section in ("limits", "requests"):
            for k in ((c.get("resources") or {}).get(section) or {}):
                if k in ("nvidia.com/gpu", "amd.com/gpu") or k.startswith("nvidia.com/mig-"):
                    return True
    return False


def pods_to_move(pods: list[dict], node: str, own_namespace: str, skip_prefixes: list[str]) -> tuple[list, list]:
    """GPU pods on the node that a drain/evict should move. Returns (move, kept-with-reason)."""
    move, kept = [], []
    for p in pods:
        md, spec = p.get("metadata") or {}, p.get("spec") or {}
        if spec.get("nodeName") != node or (p.get("status") or {}).get("phase") in ("Succeeded", "Failed"):
            continue
        if not _gpu_pod(p):
            continue
        ns, name = md.get("namespace", ""), md.get("name", "")
        owners = [o.get("kind") for o in md.get("ownerReferences") or []]
        if ns == own_namespace:
            kept.append((ns, name, "Dual-Key's own model server (kept for validation)"))
        elif any(ns.startswith(x) for x in skip_prefixes):
            kept.append((ns, name, "system namespace"))
        elif "DaemonSet" in owners:
            kept.append((ns, name, "DaemonSet pod"))
        else:
            move.append((ns, name))
    return move, kept


def build_evidence(member: str, vendor: str, scorer_group: dict | None, results: list[dict],
                   ghr_status: dict, thresholds: dict, identity: dict | None = None) -> dict:
    """What made Dual-Key quarantine this member, in a form a person can check."""
    g = scorer_group or {}
    settings = g.get("settings") or {}
    gpus = g.get("gpus") or {}
    floor = settings.get("std_floor", 0.01)
    pairs = []
    for key, ev in (g.get("pairs") or {}).items():
        a, b = key.split("|")
        if member not in (a, b):
            continue
        partner = b if a == member else a
        pairs.append({"partner": partner, "partnerVendor": (gpus.get(partner) or {}).get("vendor", ""),
                      "partnerState": (gpus.get(partner) or {}).get("state", ""),
                      "z": round(float(ev.get("ewma", 0.0)), 2), "checks": int(ev.get("n", 0))})
    pairs.sort(key=lambda x: -x["z"])
    baselines = {}
    for key, b in (g.get("baselines") or {}).items():
        if vendor in key.split("|"):
            n = b.get("n", 0)
            std = max((b.get("m2", 0.0) / (n - 1)) ** 0.5, floor) if n > 1 else floor
            baselines[key] = {"mean": round(b.get("mean", 0.0), 5), "std": round(std, 5), "samples": n}
    checks = []
    for r in results:
        if member not in (r.get("primary"), r.get("checker")):
            continue
        key = "|".join(sorted([vendor, (gpus.get(r["checker"] if r.get("primary") == member else r.get("primary")) or {}).get("vendor", vendor)]))
        b = baselines.get(key)
        z = None if not b else round((float(r.get("gap", 0)) - b["mean"]) / b["std"], 2)
        checks.append({"ts": r.get("ts"), "primary": r.get("primary"), "checker": r.get("checker"),
                       "gap": round(float(r.get("gap", 0)), 5), "z": z})
        if len(checks) >= 20:
            break
    above = [p for p in pairs if p["z"] >= thresholds.get("quarantine", 5)]
    summary = (f"disagreed with {len(above)} of {len(pairs)} partners beyond the quarantine line "
               f"({thresholds.get('quarantine', 5)}σ): " +
               ", ".join(f"{p['partner']} +{p['z']}σ over {p['checks']} checks" for p in pairs))
    return {"score": ghr_status.get("score"), "thresholds": thresholds, "partners": pairs,
            "baselines": baselines, "recentChecks": checks, "identity": identity or {},
            "build": ghr_status.get("build") or {}, "summary": summary,
            "measure": "token likelihood divergence: mean |Δ log-probability| per token when a second "
                       "GPU re-scores the same token IDs; z = how far above the learned normal for that vendor pair"}


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def parse_dcgm(output: str) -> tuple[str, str]:
    """dcgmi diag -j output -> (Pass | Fail | Warn | Unknown, summary)."""
    try:
        data = json.loads(output[output.index("{"):]) if "{" in output else None
    except ValueError:
        data = None
    if data is None:
        low = output.lower()
        if "fail" in low:
            return "Fail", output.strip()[:300]
        return "Unknown", output.strip()[:300] or "no output"
    fails, warns, passes = [], [], 0

    def visit(obj, context: str) -> None:
        nonlocal passes
        if isinstance(obj, dict):
            name = str(obj.get("name") or obj.get("test_name") or obj.get("category") or context)
            st = str(obj.get("status") or obj.get("result") or "").lower()
            if st == "fail":
                fails.append(name or "test")
            elif st == "warn":
                warns.append(name or "test")
            elif st == "pass":
                passes += 1
            for v in obj.values():
                visit(v, name)
        elif isinstance(obj, list):
            for v in obj:
                visit(v, context)

    visit(data, "")
    if fails:
        return "Fail", "failed: " + ", ".join(sorted(set(fails)))
    if warns:
        return "Warn", "warnings: " + ", ".join(sorted(set(warns)))
    return ("Pass", f"{passes} checks passed") if passes else ("Unknown", "no test results in output")


def parse_amd(output: str) -> tuple[str, str]:
    """amd-smi JSON -> Fail if any uncorrectable error count is above zero."""
    try:
        data = json.loads(output[min(i for i in (output.find("["), output.find("{")) if i >= 0):])
    except (ValueError, TypeError):
        return ("Unknown", output.strip()[:300] or "no output")
    bad = []
    for d in _walk(data):
        for k, v in d.items():
            if "uncorrect" in str(k).lower():
                try:
                    if float(v) > 0:
                        bad.append(f"{k}={v}")
                except (TypeError, ValueError):
                    pass
    return ("Fail", "uncorrectable errors: " + ", ".join(bad)) if bad else ("Pass", "no uncorrectable errors reported")


def parse_diagnostic(vendor: str, output: str) -> tuple[str, str]:
    return parse_dcgm(output) if vendor == "nvidia" else parse_amd(output)


def validation_decision(ghr_status: dict, start_checks: int, partners_needed: int,
                        checks_per_partner: int, suspect_z: float, started: float, now: float,
                        timeout_minutes: int) -> tuple[str, str]:
    """After release: has the member proven healthy again? -> (wait|passed|failed|timeout, why)."""
    state = ghr_status.get("state")
    score = ghr_status.get("score")
    new_checks = int(ghr_status.get("checks") or 0) - int(start_checks or 0)
    need = max(1, partners_needed) * checks_per_partner
    if state in ("Suspect", "Quarantined"):
        return "failed", f"disagreed with its partners again (state {state}, score {score})"
    if new_checks >= need and score is not None and float(score) < suspect_z and state == "Healthy":
        return "passed", f"{new_checks} checks since release, score {float(score):.2f} (below {suspect_z})"
    if now - started > timeout_minutes * 60:
        return "timeout", (f"only {new_checks} of {need} checks in {timeout_minutes} minutes"
                           if new_checks < need else f"no score yet after {new_checks} checks")
    return "wait", f"{max(0, new_checks)} of {need} checks since release"
