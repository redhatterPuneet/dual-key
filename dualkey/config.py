"""Shared runtime config, written by the operator into the dualkey-config ConfigMap.

The fleet is described as comparison groups. Each group holds model servers
("members") that run the exact same model; checks only happen inside a group.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

CONFIG_PATH = os.environ.get("DK_CONFIG", "/etc/dualkey/config.json")

# Redis keys shared by all components
STREAM_VERIFY = "dk:verify"        # sampled answers waiting to be checked
STREAM_RESULTS = "dk:results"      # gap results waiting for the detector
HASH_STATE = "dk:state"            # member -> Learning/Healthy/Suspect/Quarantined
HASH_PAIRCOUNT = "dk:paircount"    # "group:memberA|memberB" -> number of checks
KEY_SCORER = "dk:scorer"           # detector's saved state (JSON)

AUTH_ENV = {"dualkey": "DK_API_KEY", "discovery": "DK_DISCOVERY_API_KEY", "none": ""}


@dataclass
class Member:
    name: str
    vendor: str            # "nvidia" or "amd"
    endpoint: str          # base URL of the vLLM server
    node: str = ""
    model: str = ""        # served model name to put in requests to THIS server
    auth: str = "dualkey"  # which API key to send: dualkey | discovery | none
    build: dict = field(default_factory=dict)   # vllm version, image digest, driver


@dataclass
class Group:
    name: str
    model_id: str = ""
    source: str = "manual"     # manual | discovered | reference
    identity: dict = field(default_factory=dict)  # model, revision, dtype, tokenizer, vLLM...
    members: list[Member] = field(default_factory=list)

    def member(self, name: str) -> Optional[Member]:
        return next((m for m in self.members if m.name == name), None)


@dataclass
class Config:
    groups: list[Group] = field(default_factory=list)
    redis_url: str = "redis://dualkey-redis:6379/0"
    sample_rate: float = 0.05
    suspect_sample_rate: float = 0.5
    cross_vendor_weight: float = 0.7
    check_all: bool = False          # lab: check every sample on every other member
    verifier_concurrency: int = 8
    probe_rate: float = 1.0          # probe prompts per second per group
    probe_max_tokens: int = 48
    detector: dict = field(default_factory=dict)
    namespace: str = "dualkey"
    cluster_name: str = "dualkey"

    def group(self, name: str) -> Optional[Group]:
        return next((g for g in self.groups if g.name == name), None)

    def default_group(self) -> Optional[Group]:
        return self.groups[0] if len(self.groups) == 1 else self.group("manual")


def api_key_for(member: Member) -> str:
    env = AUTH_ENV.get(member.auth, "DK_API_KEY")
    return os.environ.get(env, "") if env else ""


def from_dict(raw: dict) -> Config:
    raw = dict(raw)
    groups = []
    for g in raw.pop("groups", []) or []:
        members = [Member(**{k: v for k, v in m.items() if k in Member.__dataclass_fields__})
                   for m in g.pop("members", [])]
        groups.append(Group(members=members, **{k: v for k, v in g.items() if k in Group.__dataclass_fields__}))
    # v0.1 style: a flat list of GPUs becomes one "manual" group
    legacy = raw.pop("gpus", None)
    served = raw.pop("served_model_name", "granite")
    if legacy and not groups:
        groups = [Group(name="manual", model_id=served, source="manual",
                        members=[Member(model=served, **g) for g in legacy])]
    known = {k: v for k, v in raw.items() if k in Config.__dataclass_fields__}
    cfg = Config(groups=groups, **known)
    cfg.redis_url = os.environ.get("DK_REDIS_URL", cfg.redis_url)
    return cfg


def load(path: str = CONFIG_PATH) -> Config:
    with open(path) as f:
        return from_dict(json.load(f))
