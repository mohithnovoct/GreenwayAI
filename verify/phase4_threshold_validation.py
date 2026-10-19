"""Phase 4 Task 4: semantic-cache threshold validation.

Preregistered in verify/phase4_threshold_prereg.md. Run only on a machine that
can load the embedding model and download GLUE QQP and PAWS-Wiki:

    uv run python verify/phase4_threshold_validation.py

Heavy dependencies (datasets, torch, sentence-transformers) are imported lazily
so the pure functions below can be unit-tested without them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

SEED = 20261019
GRID = [round(0.50 + i * 0.01, 2) for i in range(50)]
MAX_DEV_FPR = 0.01
MAX_TEST_FPR_UPPER = 0.02
MIN_PARAPHRASE_HIT_RATE = 0.20
QQP_PER_LABEL = 600
PAWS_PER_LABEL = 400
ADVERSARIAL_SOURCES = {"paws", "handbuilt"}
HANDBUILT_PATH = Path(__file__).with_name("phase4_handbuilt_pairs.json")
DEFAULT_OUTPUT = Path(__file__).with_name("phase4_threshold_results.json")

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_NEGATION = re.compile(r"\b(?:not|no|never|cannot|without|none)\b|n't", re.IGNORECASE)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator))


def guard_passes(a: str, b: str) -> bool:
    if sorted(_NUMBER.findall(a)) != sorted(_NUMBER.findall(b)):
        return False
    return len(_NEGATION.findall(a)) % 2 == len(_NEGATION.findall(b)) % 2


def split_pairs(pairs: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    """Assign dev/test within each (source, label) stratum, except fixed splits."""
    rng = random.Random(seed)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        if pair.get("split") in {"dev", "test"}:
            continue
        groups[(pair["source"], pair["label"])].append(pair)
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda p: p["id"])
        rng.shuffle(members)
        half = len(members) // 2
        for index, pair in enumerate(members):
            pair["split"] = "dev" if index < half else "test"
    return pairs


def hit_rate(sims: list[float], threshold: float) -> tuple[int, int]:
    return sum(1 for s in sims if s >= threshold), len(sims)


def choose_threshold(negative_sims: list[float], grid: list[float], max_fpr: float) -> float | None:
    for threshold in grid:
        hits, total = hit_rate(negative_sims, threshold)
        if total and hits / total <= max_fpr:
            return threshold
    return None


def _stat(hits: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(hits, total)
    return {
        "hits": hits,
        "n": total,
        "rate": (hits / total) if total else None,
        "wilson95": [low, high],
    }


def evaluate_arm(
    pairs: list[dict[str, Any]],
    is_hit: Callable[[dict[str, Any], float], bool],
) -> dict[str, Any]:
    """Apply the preregistered rule to scored pairs for one arm."""

    def rate_for(selected: list[dict[str, Any]], threshold: float) -> tuple[int, int]:
        return sum(1 for p in selected if is_hit(p, threshold)), len(selected)

    def select(split: str, label: int, sources: set[str] | None = None) -> list[dict[str, Any]]:
        return [
            p
            for p in pairs
            if p["split"] == split and p["label"] == label and (sources is None or p["source"] in sources)
        ]

    dev_adversarial_negatives = select("dev", 0, ADVERSARIAL_SOURCES)
    dev_positives = select("dev", 1)
    sweep = []
    for threshold in GRID:
        fp_hits, fp_n = rate_for(dev_adversarial_negatives, threshold)
        tp_hits, tp_n = rate_for(dev_positives, threshold)
        sweep.append(
            {
                "threshold": threshold,
                "dev_adversarial_fpr": fp_hits / fp_n if fp_n else None,
                "dev_paraphrase_hit_rate": tp_hits / tp_n if tp_n else None,
            }
        )
    chosen = None
    for row in sweep:
        if row["dev_adversarial_fpr"] is not None and row["dev_adversarial_fpr"] <= MAX_DEV_FPR:
            chosen = row["threshold"]
            break
    result: dict[str, Any] = {"chosen_threshold": chosen, "dev_sweep": sweep}
    if chosen is None:
        result.update(enable_by_default=False, reason="no grid threshold met the dev adversarial FPR limit")
        return result
    test = {
        "adversarial_fpr": _stat(*rate_for(select("test", 0, ADVERSARIAL_SOURCES), chosen)),
        "qqp_negative_fpr": _stat(*rate_for(select("test", 0, {"qqp"}), chosen)),
        "pooled_fpr": _stat(*rate_for(select("test", 0), chosen)),
        "paraphrase_hit_rate_pooled": _stat(*rate_for(select("test", 1), chosen)),
        "paraphrase_hit_rate_qqp": _stat(*rate_for(select("test", 1, {"qqp"}), chosen)),
        "paraphrase_hit_rate_paws": _stat(*rate_for(select("test", 1, {"paws"}), chosen)),
        "paraphrase_hit_rate_handbuilt": _stat(*rate_for(select("test", 1, {"handbuilt"}), chosen)),
    }
    fpr_ok = test["adversarial_fpr"]["wilson95"][1] <= MAX_TEST_FPR_UPPER
    hit_ok = (test["paraphrase_hit_rate_pooled"]["rate"] or 0.0) >= MIN_PARAPHRASE_HIT_RATE
    result["test"] = test
    result["enable_by_default"] = bool(fpr_ok and hit_ok)
    result["reason"] = (
        "test adversarial FPR upper bound <= 2% and pooled paraphrase hit rate >= 20%"
        if result["enable_by_default"]
        else f"adversarial_fpr_upper_ok={fpr_ok}; paraphrase_hit_rate_ok={hit_ok}"
    )
    return result


def load_handbuilt() -> list[dict[str, Any]]:
    data = json.loads(HANDBUILT_PATH.read_text(encoding="utf-8"))
    return [
        {
            "id": p["id"],
            "source": "handbuilt",
            "label": p["label"],
            "category": p["category"],
            "a": p["a"],
            "b": p["b"],
            "split": p["split"],
        }
        for p in data["pairs"]
    ]


def _sample(rows: list[dict[str, Any]], per_label: int, rng: random.Random) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for label in (0, 1):
        pool = [r for r in rows if r["label"] == label]
        chosen.extend(rng.sample(pool, per_label))
    return chosen


def load_qqp(rng: random.Random) -> tuple[list[dict[str, Any]], str]:
    from datasets import load_dataset

    dataset = load_dataset("glue", "qqp", split="validation")
    rows = [
        {"id": f"qqp{i}", "source": "qqp", "label": int(r["label"]), "a": r["question1"], "b": r["question2"]}
        for i, r in enumerate(dataset)
    ]
    return _sample(rows, QQP_PER_LABEL, rng), str(getattr(dataset, "_fingerprint", "unknown"))


def load_paws(rng: random.Random) -> tuple[list[dict[str, Any]], str]:
    from datasets import load_dataset

    last_error: Exception | None = None
    for name in ("google-research-datasets/paws", "paws"):
        try:
            dataset = load_dataset(name, "labeled_final", split="validation")
            break
        except Exception as error:  # noqa: BLE001 - try the alternate dataset id
            last_error = error
    else:
        raise RuntimeError(f"Could not load PAWS labeled_final: {last_error}")
    rows = [
        {"id": f"paws{i}", "source": "paws", "label": int(r["label"]), "a": r["sentence1"], "b": r["sentence2"]}
        for i, r in enumerate(dataset)
    ]
    return _sample(rows, PAWS_PER_LABEL, rng), str(getattr(dataset, "_fingerprint", "unknown"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rng = random.Random(SEED)
    qqp, qqp_fingerprint = load_qqp(rng)
    paws, paws_fingerprint = load_paws(rng)
    pairs = split_pairs(qqp + paws + load_handbuilt())

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from optimizers.semantic_cache import DEFAULT_MODEL_REVISION, SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    texts = sorted({t for p in pairs for t in (p["a"], p["b"])})
    started = time.perf_counter()
    vectors = {text: embedder.encode(text) for text in texts}
    embed_seconds = time.perf_counter() - started
    for pair in pairs:
        va, vb = vectors[pair["a"]], vectors[pair["b"]]
        pair["sim"] = sum(x * y for x, y in zip(va, vb))
        pair["guard_ok"] = guard_passes(pair["a"], pair["b"])

    plain = evaluate_arm(pairs, lambda p, t: p["sim"] >= t)
    guard = evaluate_arm(pairs, lambda p, t: p["sim"] >= t and p["guard_ok"])

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        head = "unknown"
    output = {
        "seed": SEED,
        "git_head": head,
        "model_id": embedder.model_id,
        "model_revision": embedder.revision,
        "expected_model_revision": DEFAULT_MODEL_REVISION,
        "dataset_fingerprints": {"qqp": qqp_fingerprint, "paws": paws_fingerprint},
        "embedded_texts": len(texts),
        "embed_seconds": round(embed_seconds, 2),
        "plain_arm": plain,
        "guard_arm": guard,
        "pairs": [
            {k: p[k] for k in ("id", "source", "label", "split", "sim", "guard_ok")} for p in pairs
        ],
    }
    args.output.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(f"plain arm: chosen={plain['chosen_threshold']} enable_by_default={plain['enable_by_default']}")
    print(f"           {plain['reason']}")
    if "test" in plain:
        for name, stat in plain["test"].items():
            print(f"           {name}: {stat['hits']}/{stat['n']} wilson95={stat['wilson95']}")
    print(f"guard arm: chosen={guard['chosen_threshold']} enable_by_default={guard['enable_by_default']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
