"""What Dual-Key's checking costs, measured from real check results.

Every check result carries:
  rescore_s   wall time of the checker GPU's re-scoring call (an upper bound on its GPU time;
              vLLM skips its prefix cache for this kind of scoring, so each check is a full
              prefill of prompt + answer)
  queue_s     how long the sampled answer waited before a verifier picked it up
  answer_s    how long the primary GPU took to produce the answer (probe prompts)

From those this module reports latency percentiles, how busy Dual-Key keeps each
checker GPU, and the cost per 1,000 checks and per 1,000 application requests.
"""
from __future__ import annotations

import math

# Approximate AWS on-demand list prices (USD per hour, us-east). Override with
# DualKeyCluster spec.cost.gpuHourUSD: a single number, or a map keyed by
# instance type or node name.
LIST_PRICE_PER_HOUR = {
    "g4dn.xlarge": 0.526, "g4dn.2xlarge": 0.752, "g5.xlarge": 1.006, "g5.2xlarge": 1.212,
    "g6.xlarge": 0.805, "g6.2xlarge": 0.978, "g6e.xlarge": 1.861,
}


def percentile(values: list[float], q: float):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return vals[lo] if lo == hi else vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def price_for(node: str, instance_type: str, cost_spec) -> tuple[float | None, str]:
    """Returns (USD per GPU-hour, where the number came from)."""
    if isinstance(cost_spec, (int, float)):
        return float(cost_spec), "spec.cost.gpuHourUSD"
    if isinstance(cost_spec, dict):
        for key in (node, node.split(".")[0], instance_type):
            if key and key in cost_spec:
                return float(cost_spec[key]), "spec.cost.gpuHourUSD"
    if instance_type in LIST_PRICE_PER_HOUR:
        return LIST_PRICE_PER_HOUR[instance_type], "AWS on-demand list price (approx.)"
    return None, "unknown: set spec.cost.gpuHourUSD"


def summarize(results: list[dict], now: float, member_nodes: dict[str, str],
              node_types: dict[str, str], cost_spec=None, sample_rate: float = 0.05,
              window_s: float = 300.0) -> dict:
    recent = [r for r in results if now - float(r.get("ts", 0)) <= window_s and r.get("rescore_s") is not None]
    if not recent:
        return {"windowSeconds": window_s, "checks": 0}
    span = max(1.0, min(window_s, now - min(float(r["ts"]) for r in recent)))
    rescore = [float(r["rescore_s"]) for r in recent]
    queue = [float(r["queue_s"]) for r in recent if r.get("queue_s") is not None]
    answer = [float(r["answer_s"]) for r in recent if r.get("answer_s")]
    tokens = [int(r.get("scored_tokens") or 0) for r in recent]

    # checks per sampled answer (1 normally, members-1 in check-all lab mode)
    per_req: dict[str, int] = {}
    for r in recent:
        per_req[r.get("request_id", id(r))] = per_req.get(r.get("request_id", id(r)), 0) + 1
    checks_per_sample = sum(per_req.values()) / len(per_req)

    # per checker GPU: share of wall time spent re-scoring for Dual-Key, and cost
    busy: dict[str, float] = {}
    costs, sources = [], set()
    for r in recent:
        chk = r.get("checker", "")
        busy[chk] = busy.get(chk, 0.0) + float(r["rescore_s"])
        node = member_nodes.get(chk, "")
        price, src = price_for(node, node_types.get(node, ""), cost_spec)
        sources.add(src)
        if price is not None:
            costs.append(float(r["rescore_s"]) * price / 3600.0)
    cost_per_check = (sum(costs) / len(costs)) if costs else None
    med_rescore, med_answer = percentile(rescore, 0.5), percentile(answer, 0.5)
    return {
        "windowSeconds": round(span),
        "checks": len(recent),
        "checksPerMinute": round(len(recent) * 60.0 / span, 1),
        "rescoreP50": percentile(rescore, 0.5), "rescoreP95": percentile(rescore, 0.95),
        "queueP50": percentile(queue, 0.5), "queueP95": percentile(queue, 0.95),
        "answerP50": med_answer,
        "checkVsAnswer": (med_rescore / med_answer) if med_rescore and med_answer else None,
        "scoredTokensAvg": round(sum(tokens) / len(tokens)) if tokens else 0,
        "checksPerSample": round(checks_per_sample, 2),
        "checkerBusyPct": {k: round(100.0 * v / span, 1) for k, v in sorted(busy.items())},
        "costPer1000Checks": None if cost_per_check is None else round(cost_per_check * 1000, 4),
        "sampleRate": sample_rate,
        "costPer1000Requests": None if cost_per_check is None else
            round(cost_per_check * 1000 * sample_rate * checks_per_sample, 4),
        "priceSource": "; ".join(sorted(sources)),
        "note": "Re-score time is wall time per check (an upper bound on checker GPU time); "
                "vLLM skips its prefix cache for this scoring, so every check is a full prefill",
    }
