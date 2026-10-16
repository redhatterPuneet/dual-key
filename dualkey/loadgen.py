"""Test client: sends prompts to the Dual-Key Gateway, standing in for a real app.

Run: python -m dualkey.loadgen --url http://dualkey-gateway:8080 --requests 2000 --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from collections import Counter
from importlib import resources

import httpx


def load_prompts(path: str = "") -> list[str]:
    if path:
        with open(path) as f:
            text = f.read()
    else:
        text = resources.files("dualkey").joinpath("prompts.txt").read_text()
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


async def run(url: str, n: int, concurrency: int, max_tokens: int, model: str,
              prompts: list[str], sample_header: bool) -> Counter:
    stats: Counter = Counter()
    sem = asyncio.Semaphore(concurrency)
    headers = {"x-dualkey": "sample"} if sample_header else {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as http:
        async def one(i: int):
            body = {"prompt": random.choice(prompts), "max_tokens": max_tokens, "temperature": 0}
            if model:
                body["model"] = model
            async with sem:
                try:
                    r = await http.post(f"{url}/v1/completions", json=body, headers=headers)
                    stats[f"http_{r.status_code}"] += 1
                    stats[f"gpu_{r.headers.get('x-dualkey-gpu', '?')}"] += 1
                except httpx.HTTPError:
                    stats["error"] += 1
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{n} sent  {dict(stats)}", flush=True)

        t0 = time.time()
        await asyncio.gather(*(one(i) for i in range(n)))
        stats["seconds"] = int(time.time() - t0)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=os.environ.get("DK_GATEWAY_URL", "http://dualkey-gateway:8080"))
    p.add_argument("--requests", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--model", default="")
    p.add_argument("--prompts", default="")
    p.add_argument("--sample-all", action="store_true",
                   help="ask the gateway to sample every request (faster lab results)")
    a = p.parse_args()
    stats = asyncio.run(run(a.url, a.requests, a.concurrency, a.max_tokens, a.model,
                            load_prompts(a.prompts), a.sample_all))
    print("done:", dict(stats))


if __name__ == "__main__":
    main()
