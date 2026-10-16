"""A tiny fake vLLM server for tests: deterministic 'model', per-vendor rounding
noise, and a switchable fault that corrupts the GPU's confidence numbers."""
from __future__ import annotations

import json
import random
import zlib

import httpx

VOCAB = 50000


def tok(word: str) -> int:
    return zlib.crc32(word.encode()) % VOCAB


def true_logprob(position: int, token: int) -> float:
    # the 'model': a fixed logprob for each (position, token)
    return -3.0 * (((token * 31) + (position * 7)) % 997) / 997.0


class FakeGpu:
    def __init__(self, name: str, vendor: str, seed: int = 0):
        self.name = name
        self.vendor = vendor
        self.faulty = False
        self.rng = random.Random(seed)
        self.calls = 0

    def lp(self, position: int, token: int) -> float:
        v = true_logprob(position, token)
        # different chips round slightly differently
        sd, bias = (0.002, 0.0) if self.vendor == "nvidia" else (0.004, 0.003)
        v += self.rng.gauss(bias, sd)
        if self.faulty and self.rng.random() < 0.3:
            v -= 2.0   # silent corruption: confidently wrong numbers
        return min(v, 0.0)

    def handle(self, path: str, body: dict) -> dict:
        self.calls += 1
        if path == "/tokenize":
            return {"tokens": [tok(w) for w in str(body["prompt"]).split()]}
        prompt = body["prompt"]
        if isinstance(prompt, list):                      # re-scoring request
            ids = prompt
            pl = [None] + [{str(t): {"logprob": self.lp(i, t), "rank": 1}} for i, t in enumerate(ids) if i > 0]
            return {"choices": [{"index": 0, "text": "x", "prompt_logprobs": pl,
                                 "token_ids": [1], "prompt_token_ids": ids}]}
        p_ids = [tok(w) for w in str(prompt).split()]
        n_out = int(body.get("max_tokens", 16))
        out = [(sum(p_ids) * (k + 3)) % VOCAB for k in range(n_out)]
        start = len(p_ids)
        lps = [self.lp(start + k, t) for k, t in enumerate(out)]
        return {"choices": [{"index": 0, "text": " ".join(map(str, out)),
                             "logprobs": {"token_logprobs": lps,
                                          "tokens": [f"token_id:{t}" for t in out]},
                             "token_ids": out, "prompt_token_ids": p_ids,
                             "finish_reason": "length"}]}


def transport(gpus: dict[str, FakeGpu]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        gpu = gpus[request.url.host]
        body = json.loads(request.content or b"{}")
        return httpx.Response(200, json=gpu.handle(request.url.path.replace("/v1", ""), body))
    return httpx.MockTransport(handler)
