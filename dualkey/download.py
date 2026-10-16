"""Init-container helper: download a model from Hugging Face into a local folder.

Run: python -m dualkey.download --model ibm-granite/granite-3.3-2b-instruct --dir /models/model
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=os.environ.get("DK_MODEL_REVISION") or None)
    p.add_argument("--dir", required=True)
    a = p.parse_args(argv)

    if os.path.exists(os.path.join(a.dir, "config.json")):
        print(f"model already present in {a.dir}")
        return 0
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=a.model,
        revision=a.revision,
        local_dir=a.dir,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
        token=os.environ.get("HF_TOKEN") or None,
    )
    print(f"downloaded {a.model} to {a.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
