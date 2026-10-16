"""Dual-Key scoring: gap math, baselines, pair evidence, per-GPU score and state.

Pure Python, no I/O, so it can be unit-tested and reused by the Detector.

How the decision works (matches "How the Dual-Key Flow Works"):
  1. Every check compares two GPUs (the one that answered, and a checker) and
     yields one gap number D.
  2. During the learning period, D values are collected per vendor pairing
     (nvidia|nvidia, amd|nvidia, ...) to learn what "normal" looks like.
  3. After learning, each D becomes z = how unusual it is for that pairing.
  4. Each pair of GPUs keeps a slowly moving average of its z values.
  5. A GPU's score is the LOWEST pair average among its partners (or a low
     quantile in big fleets). So a GPU only scores high when it disagrees with
     ALL of its partners - the majority vote from the docs.
  6. Healthy -> Suspect -> (confirmed over several more checks) -> Quarantined.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

HEALTHY = "Healthy"
SUSPECT = "Suspect"
QUARANTINED = "Quarantined"
LEARNING = "Learning"

MISSING_LOGPROB = -20.0  # used when a checker did not return the token's logprob


def compute_gap(primary_logprobs: list[Optional[float]],
                checker_logprobs: list[Optional[float]]) -> Optional[float]:
    """Mean absolute log-probability gap over the answer's tokens.

    Both lists are aligned to the generated tokens. Entries where the primary
    has no value are skipped. Missing checker values count as very unlikely.
    """
    diffs = []
    for p, c in zip(primary_logprobs, checker_logprobs):
        if p is None or (isinstance(p, float) and math.isnan(p)):
            continue
        c_val = MISSING_LOGPROB if c is None else max(c, MISSING_LOGPROB)
        p_val = max(p, MISSING_LOGPROB)
        diffs.append(abs(p_val - c_val))
    if not diffs:
        return None
    return sum(diffs) / len(diffs)


def vendor_pair_key(vendor_a: str, vendor_b: str) -> str:
    a, b = sorted([vendor_a.lower(), vendor_b.lower()])
    return f"{a}|{b}"


def gpu_pair_key(gpu_a: str, gpu_b: str) -> str:
    a, b = sorted([gpu_a, gpu_b])
    return f"{a}|{b}"


@dataclass
class Baseline:
    """Running mean/std of healthy gaps for one vendor pairing (Welford)."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    frozen: bool = False

    def add(self, x: float) -> None:
        if self.frozen:
            return
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def std(self, floor: float) -> float:
        if self.n < 2:
            return floor
        return max(math.sqrt(self.m2 / (self.n - 1)), floor)


@dataclass
class PairEvidence:
    ewma: float = 0.0
    n: int = 0


@dataclass
class GpuState:
    vendor: str
    state: str = LEARNING
    score: Optional[float] = None
    partners: int = 0
    checks: int = 0
    confirm_count: int = 0


@dataclass
class Settings:
    baseline_samples: int = 200      # checks per vendor pairing before "normal" is known
    std_floor: float = 0.01          # avoid dividing by ~0 when GPUs agree almost perfectly
    ewma_alpha: float = 0.05         # how fast pair evidence moves
    min_pair_checks: int = 20        # a pair counts only after this many checks
    min_partners: int = 2            # need at least 2 partners for a majority (3 GPUs)
    partner_quantile: float = 0.0    # 0.0 = lowest pair (3 GPUs); use ~0.2 in big fleets
    suspect_z: float = 3.0
    quarantine_z: float = 5.0
    confirm_checks: int = 10         # consecutive over-limit updates before quarantine
    z_clip: float = 20.0

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Transition:
    gpu: str
    old: str
    new: str
    score: Optional[float]


@dataclass
class FleetScorer:
    settings: Settings = field(default_factory=Settings)
    baselines: dict = field(default_factory=dict)   # vendor pair -> Baseline
    pairs: dict = field(default_factory=dict)       # gpu pair -> PairEvidence
    gpus: dict = field(default_factory=dict)        # gpu name -> GpuState

    # ---------- setup ----------
    def register_gpu(self, name: str, vendor: str) -> None:
        if name not in self.gpus:
            self.gpus[name] = GpuState(vendor=vendor)
        else:
            self.gpus[name].vendor = vendor

    def baseline_status(self) -> dict:
        return {k: {"samples": b.n, "ready": b.frozen,
                    "mean": round(b.mean, 4),
                    "std": round(b.std(self.settings.std_floor), 4)}
                for k, b in self.baselines.items()}

    def learning_complete(self) -> bool:
        """True once every vendor pairing present in the fleet has a baseline."""
        vendors = sorted({g.vendor for g in self.gpus.values()})
        needed = set()
        for i, a in enumerate(vendors):
            for b in vendors[i:]:
                # a same-vendor pairing only exists if 2+ GPUs of that vendor
                if a == b and sum(1 for g in self.gpus.values() if g.vendor == a) < 2:
                    continue
                needed.add(vendor_pair_key(a, b))
        return all(self.baselines.get(k) and self.baselines[k].frozen for k in needed) and bool(needed)

    # ---------- core ----------
    def observe(self, primary: str, checker: str, gap: float) -> list[Transition]:
        """Record one check result. Returns any state changes it caused."""
        s = self.settings
        gp, gc = self.gpus[primary], self.gpus[checker]
        gp.checks += 1
        gc.checks += 1
        vkey = vendor_pair_key(gp.vendor, gc.vendor)
        base = self.baselines.setdefault(vkey, Baseline())

        if not base.frozen:
            base.add(gap)
            if base.n >= s.baseline_samples:
                base.frozen = True
            return self._refresh_states([primary, checker])

        z = (gap - base.mean) / base.std(s.std_floor)
        z = max(-s.z_clip, min(s.z_clip, z))
        pkey = gpu_pair_key(primary, checker)
        ev = self.pairs.setdefault(pkey, PairEvidence())
        ev.ewma = z if ev.n == 0 else (s.ewma_alpha * z + (1 - s.ewma_alpha) * ev.ewma)
        ev.n += 1
        return self._refresh_states([primary, checker])

    def gpu_score(self, name: str) -> tuple[Optional[float], int]:
        s = self.settings
        vals = []
        for key, ev in self.pairs.items():
            a, b = key.split("|")
            if name not in (a, b) or ev.n < s.min_pair_checks:
                continue
            partner = b if a == name else a
            # a quarantined partner is not a trustworthy witness
            if self.gpus.get(partner) and self.gpus[partner].state == QUARANTINED:
                continue
            vals.append(ev.ewma)
        if len(vals) < s.min_partners:
            return None, len(vals)
        vals.sort()
        idx = int(math.floor(s.partner_quantile * (len(vals) - 1)))
        return vals[idx], len(vals)

    def _refresh_states(self, names: list[str]) -> list[Transition]:
        s = self.settings
        out = []
        learning = not self.learning_complete()
        for name in names:
            g = self.gpus[name]
            old = g.state
            if g.state == QUARANTINED:
                continue  # sticky until an admin releases it
            if learning:
                g.state = LEARNING
            else:
                score, partners = self.gpu_score(name)
                g.score, g.partners = score, partners
                if g.state == LEARNING:
                    g.state = HEALTHY
                if score is None:
                    pass
                elif g.state == HEALTHY:
                    if score >= s.suspect_z:
                        g.state = SUSPECT
                        g.confirm_count = 0
                elif g.state == SUSPECT:
                    if score >= s.quarantine_z:
                        g.confirm_count += 1
                        if g.confirm_count >= s.confirm_checks:
                            g.state = QUARANTINED
                    else:
                        g.confirm_count = 0
                        if score < 0.7 * s.suspect_z:
                            g.state = HEALTHY
            if g.state != old:
                out.append(Transition(name, old, g.state, g.score))
        return out

    def release(self, name: str) -> None:
        """Admin released a GPU after repair: forget its pair evidence."""
        g = self.gpus.get(name)
        if not g:
            return
        g.state = HEALTHY
        g.score = None
        g.confirm_count = 0
        for key in [k for k in self.pairs if name in k.split("|")]:
            del self.pairs[key]

    # ---------- persistence ----------
    def to_dict(self) -> dict:
        return {
            "settings": asdict(self.settings),
            "baselines": {k: asdict(v) for k, v in self.baselines.items()},
            "pairs": {k: asdict(v) for k, v in self.pairs.items()},
            "gpus": {k: asdict(v) for k, v in self.gpus.items()},
        }

    @classmethod
    def from_dict(cls, d: dict, settings: Optional[Settings] = None) -> "FleetScorer":
        fs = cls(settings=settings or Settings.from_dict(d.get("settings", {})))
        fs.baselines = {k: Baseline(**v) for k, v in d.get("baselines", {}).items()}
        fs.pairs = {k: PairEvidence(**v) for k, v in d.get("pairs", {}).items()}
        fs.gpus = {k: GpuState(**v) for k, v in d.get("gpus", {}).items()}
        return fs
