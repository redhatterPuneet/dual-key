"""Put a sampled answer on the verify queue (shared by the Gateway and the Prober)."""
from __future__ import annotations

import json
import time
import uuid

from .config import STREAM_VERIFY


def valid_record(rec: dict) -> bool:
    return bool(rec.get("prompt_ids")) and bool(rec.get("output_ids")) and \
        all(t is not None for t in rec["output_ids"])


async def submit(redis, group: str, primary: str, rec: dict, source: str) -> bool:
    if not valid_record(rec):
        return False
    item = {
        "request_id": str(uuid.uuid4()),
        "ts": time.time(),
        "group": group,
        "primary": primary,
        "source": source,                 # "probe" or "gateway"
        "prompt_ids": rec["prompt_ids"],
        "output_ids": rec["output_ids"],
        "primary_logprobs": rec["primary_logprobs"],
        "answer_s": rec.get("answer_s"),     # how long the primary GPU took to answer
    }
    await redis.xadd(STREAM_VERIFY, {"data": json.dumps(item)}, maxlen=100000)
    return True
