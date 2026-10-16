"""Thin async client for the two calls Dual-Key makes to a vLLM model server.

1. answer():  normal completion, plus per-token confidence and token IDs.
2. rescore(): teacher-forced scoring of someone else's answer (no new text).
"""
from __future__ import annotations

from typing import Any, Optional

import httpx


class VllmError(RuntimeError):
    pass


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _parse_token_id(tok: str) -> Optional[int]:
    # with return_tokens_as_token_ids, tokens look like "token_id:1234"
    if isinstance(tok, str) and tok.startswith("token_id:"):
        try:
            return int(tok.split(":", 1)[1])
        except ValueError:
            return None
    return None


async def answer(client: httpx.AsyncClient, endpoint: str, api_key: str,
                 body: dict[str, Any]) -> tuple[dict, dict]:
    """Forward a completion request. Returns (raw response, extracted record).

    record = {prompt_ids, output_ids, primary_logprobs}
    """
    req = dict(body)
    req["stream"] = False
    req["logprobs"] = max(int(req.get("logprobs") or 1), 1)
    req["return_token_ids"] = True
    req["return_tokens_as_token_ids"] = True
    r = await client.post(f"{endpoint}/v1/completions", json=req, headers=_headers(api_key))
    if r.status_code != 200:
        raise VllmError(f"{endpoint} answered {r.status_code}: {r.text[:300]}")
    data = r.json()
    choice = data["choices"][0]
    lp = choice.get("logprobs") or {}
    primary = list(lp.get("token_logprobs") or [])
    out_ids = choice.get("token_ids")
    if not out_ids:
        out_ids = [_parse_token_id(t) for t in lp.get("tokens") or []]
    prompt_ids = choice.get("prompt_token_ids") or data.get("prompt_token_ids")
    if not prompt_ids:
        prompt_ids = await tokenize(client, endpoint, api_key, req.get("model"), req.get("prompt"))
    return data, {"prompt_ids": prompt_ids, "output_ids": out_ids, "primary_logprobs": primary}


async def tokenize(client: httpx.AsyncClient, endpoint: str, api_key: str,
                   model: Optional[str], prompt: Any) -> list[int]:
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        return list(prompt)
    r = await client.post(f"{endpoint}/tokenize", json={"model": model, "prompt": prompt},
                          headers=_headers(api_key))
    if r.status_code != 200:
        raise VllmError(f"tokenize failed on {endpoint}: {r.status_code}")
    return r.json()["tokens"]


async def rescore(client: httpx.AsyncClient, endpoint: str, api_key: str, model: str,
                  prompt_ids: list[int], output_ids: list[int]) -> list[Optional[float]]:
    """Ask a checker GPU how likely it finds each of the answer's tokens."""
    body = {
        "model": model,
        "prompt": list(prompt_ids) + list(output_ids),
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 1,
    }
    r = await client.post(f"{endpoint}/v1/completions", json=body, headers=_headers(api_key))
    if r.status_code != 200:
        raise VllmError(f"rescore failed on {endpoint}: {r.status_code} {r.text[:300]}")
    choice = r.json()["choices"][0]
    pl = choice.get("prompt_logprobs") or []
    start = len(prompt_ids)
    result: list[Optional[float]] = []
    for j, tok in enumerate(output_ids):
        idx = start + j
        entry = pl[idx] if idx < len(pl) else None
        val = None
        if entry and tok is not None:
            hit = entry.get(str(tok)) if isinstance(entry, dict) else None
            if hit is None and isinstance(entry, dict):
                hit = entry.get(tok)
            if isinstance(hit, dict):
                val = hit.get("logprob")
            elif isinstance(hit, (int, float)):
                val = float(hit)
        result.append(val)
    return result
