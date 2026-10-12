"""Run the quota-gated, replicated Gemini 2.5 Flash follow-up probe."""

import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import httpx

from ingress.normalization import build_gateway_request, count_tokens
from optimizers.prompt_optimizer import optimize_gateway_request
from verify.gemini_probe_round2 import (
    MODEL,
    PROMPT_FILE,
    persist_json,
    request_with_retries,
    response_metrics,
)
from verify.gemini_probe_round3_scoring import (
    DEFAULT_RUBRIC_FILE,
    create_blinded_sheet,
)

PREREGISTRATION_COMMIT = "f2a0db38a03ce0799c9f0cb85d7e46743197bfe4"
MAX_TOKENS = 2048
PAIRS_PER_SHAPE = 2
PLANNED_REQUESTS = 8
MIN_REMAINING_REQUESTS = 10
DEFAULT_RESULTS_FILE = Path(__file__).with_name("gemini_probe_round3_results.json")
DEFAULT_ATTEMPTS_FILE = Path(__file__).with_name("gemini_probe_round3_attempts.json")
DEFAULT_PASS1_FILE = Path(__file__).with_name("gemini_probe_round3_pass1.json")
DEFAULT_PASS2_FILE = Path(__file__).with_name("gemini_probe_round3_pass2.json")


def build_schedule(prompts: list[dict[str, str]]) -> list[dict[str, Any]]:
    if len(prompts) < 3:
        raise ValueError("Round 3 requires the focused, how-to, and paragraph prompts")

    schedule: list[dict[str, Any]] = []
    sample_number = 0
    for shape_index in range(2):
        for pair_number in range(1, PAIRS_PER_SHAPE + 1):
            first_arm = (
                ("baseline", "optimized")
                if pair_number % 2 == 1
                else ("optimized", "baseline")
            )
            for pair_position, arm in enumerate(first_arm, start=1):
                sample_number += 1
                schedule.append(
                    {
                        "sample_number": sample_number,
                        "shape_number": shape_index + 1,
                        "shape": prompts[shape_index]["shape"],
                        "pair_number": pair_number,
                        "pair_position": pair_position,
                        "measurement_type": "paired",
                        "arm": arm,
                        "prompt": prompts[shape_index]["prompt"],
                    }
                )
    return schedule


def git_snapshot() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "git_rev_parse_head": commit,
        "git_dirty_tree": bool(status.strip()),
        "rubric_preregistration_commit": PREREGISTRATION_COMMIT,
    }


def evaluate_results(samples: list[dict[str, Any]]) -> dict[str, Any]:
    shape_results: list[dict[str, Any]] = []

    for shape in ("focused_explanation", "short_how_to"):
        shape_samples = [
            sample
            for sample in samples
            if sample["shape"] == shape and sample["measurement_type"] == "paired"
        ]
        baselines = [sample for sample in shape_samples if sample["arm"] == "baseline"]
        optimized = [sample for sample in shape_samples if sample["arm"] == "optimized"]
        has_two_pairs = (
            len(shape_samples) == 4
            and {sample["pair_number"] for sample in shape_samples} == {1, 2}
        )
        complete = has_two_pairs and len(baselines) == 2 and len(optimized) == 2 and all(
            sample["metrics"]["finish_reason"] == "stop"
            for sample in [*baselines, *optimized]
        )
        optimizer_changed_each_prompt = len(optimized) == 2 and all(
            optimized_sample.get("prompt_optimization_changes")
            for optimized_sample in optimized
        )
        baseline_mean = sum(
            sample["metrics"]["word_count"] for sample in baselines
        ) / len(baselines) if baselines else None
        optimized_mean = sum(
            sample["metrics"]["word_count"] for sample in optimized
        ) / len(optimized) if optimized else None
        baseline_spread = (
            max(sample["metrics"]["word_count"] for sample in baselines)
            - min(sample["metrics"]["word_count"] for sample in baselines)
            if len(baselines) == 2
            else None
        )
        shorter_beyond_baseline_spread = bool(
            baseline_mean is not None
            and optimized_mean is not None
            and baseline_spread is not None
            and optimized_mean < baseline_mean - baseline_spread
        )
        coverage_scored = (
            len(shape_samples) == 4
            and all("coverage_score" in sample for sample in shape_samples)
        )
        baseline_coverage_mean = (
            sum(sample["coverage_score"] for sample in baselines) / len(baselines)
            if coverage_scored
            else None
        )
        optimized_coverage_mean = (
            sum(sample["coverage_score"] for sample in optimized) / len(optimized)
            if coverage_scored
            else None
        )
        coverage_not_lower = bool(
            coverage_scored
            and baseline_coverage_mean is not None
            and optimized_coverage_mean is not None
            and optimized_coverage_mean >= baseline_coverage_mean
        )
        qualifies = bool(
            complete
            and optimizer_changed_each_prompt
            and shorter_beyond_baseline_spread
            and coverage_not_lower
        )
        shorter_but_less_complete = bool(
            complete
            and optimizer_changed_each_prompt
            and shorter_beyond_baseline_spread
            and coverage_scored
            and not coverage_not_lower
        )
        shape_results.append(
            {
                "shape": shape,
                "baseline_mean_words": baseline_mean,
                "optimized_mean_words": optimized_mean,
                "baseline_word_spread": baseline_spread,
                "all_four_responses_complete": complete,
                "optimizer_changed_both_optimized_prompts": optimizer_changed_each_prompt,
                "baseline_coverage_mean": baseline_coverage_mean,
                "optimized_coverage_mean": optimized_coverage_mean,
                "coverage_scored": coverage_scored,
                "coverage_not_lower": coverage_not_lower,
                "shorter_beyond_baseline_spread": shorter_beyond_baseline_spread,
                "qualifies": qualifies,
                "finding": (
                    "shorter but less complete"
                    if shorter_but_less_complete
                    else "qualifies"
                    if qualifies
                    else "does not qualify"
                ),
            }
        )
    return {
        "shapes": shape_results,
        "both_shapes_qualify": all(item["qualifies"] for item in shape_results),
    }


async def main(args: argparse.Namespace) -> None:
    if args.word_cap <= 0:
        raise ValueError("--word-cap must be a positive integer")
    os.environ["PROMPT_OPT_WORD_CAP"] = str(args.word_cap)
    os.environ["PROMPT_OPT_COMBINE_VARIATIONS"] = "false"

    prompts = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    schedule = build_schedule(prompts)
    execution = git_snapshot()
    protocol = {
        "max_tokens": MAX_TOKENS,
        "word_cap": args.word_cap,
        "variations_rule_enabled": False,
        "planned_requests": PLANNED_REQUESTS,
        "quota_reserved": 2,
        "minimum_quota_remaining": MIN_REMAINING_REQUESTS,
        "baseline_noise_measure": "word-count range between the two baseline responses for each shape",
        "coverage_rubric": DEFAULT_RUBRIC_FILE.name,
        "coverage_scoring_passes": ["agent_pass1", "independent_pass2"],
        "agreement_measure": "raw percent agreement per rubric item across four responses per shape",
        "decision_rule": (
            "A shape counts only if all four responses finish with stop, both optimized "
            "prompts have non-empty change records, mean optimized words are below mean "
            "baseline words by more than baseline spread, and optimized mean coverage "
            "is at least baseline mean coverage. Plan a larger run only if both shapes count."
        ),
    }
    if args.dry_run:
        preview = []
        for task in schedule:
            request = build_gateway_request(
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": task["prompt"]}],
                    "max_tokens": MAX_TOKENS,
                    "reasoning_effort": "none",
                },
                None,
            )
            optimized = (
                optimize_gateway_request(request)
                if task["arm"] == "optimized"
                else None
            )
            preview.append(
                {
                    **task,
                    "url": (
                        args.baseline_gateway
                        if task["arm"] == "baseline"
                        else args.optimized_gateway
                    ),
                    "max_tokens": MAX_TOKENS,
                    "reasoning_effort": "none",
                    "prompt_optimization_changes": (
                        list(optimized.changes) if optimized else []
                    ),
                    "effective_prompt": (
                        optimized.request.raw_payload["messages"][-1]["content"]
                        if optimized
                        else task["prompt"]
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "execution": execution,
                    "protocol": protocol,
                    "schedule": preview,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.quota_remaining is None or args.quota_remaining < MIN_REMAINING_REQUESTS:
        raise ValueError(
            "Do not run: the displayed RPD remaining must be at least 10 "
            "(eight planned requests plus two reserved)"
        )

    attempt_budget = args.quota_remaining - 2
    results: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    last_request_at: float | None = None
    async with httpx.AsyncClient(timeout=120.0) as client:
        for task in schedule:
            baseline_request = build_gateway_request(
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": task["prompt"]}],
                    "max_tokens": MAX_TOKENS,
                    "reasoning_effort": "none",
                },
                None,
            )
            expected_result = (
                optimize_gateway_request(baseline_request)
                if task["arm"] == "optimized"
                else None
            )
            expected_prompt = (
                expected_result.request.raw_payload["messages"][-1]["content"]
                if expected_result
                else task["prompt"]
            )
            expected_changes = list(expected_result.changes) if expected_result else []
            expected_content_hash = (
                expected_result.request.content_hash
                if expected_result
                else baseline_request.content_hash
            )
            url = (
                args.baseline_gateway
                if task["arm"] == "baseline"
                else args.optimized_gateway
            )
            attempt_sample = {
                **task,
                "url": url,
                "model": MODEL,
                "reasoning_effort": "none",
                "max_tokens": MAX_TOKENS,
            }
            response, last_request_at = await request_with_retries(
                client,
                url=url,
                payload=baseline_request.raw_payload,
                sample=attempt_sample,
                attempts=attempts,
                attempts_path=args.attempts_output,
                last_request_at=last_request_at,
                attempt_budget=attempt_budget,
            )
            payload = response.json()
            actual_arm = response.headers.get("x-greenway-prompt-optimization")
            actual_changes_header = response.headers.get(
                "x-greenway-prompt-optimization-changes", ""
            )
            actual_changes = [
                change for change in actual_changes_header.split(",") if change
            ]
            record = {
                **task,
                "url": url,
                "model": MODEL,
                "reasoning_effort": "none",
                "max_tokens": MAX_TOKENS,
                "prompt_optimization_arm": actual_arm,
                "prompt_optimization_changes": actual_changes,
                "expected_prompt_optimization_changes": expected_changes,
                "effective_prompt_content_hash": response.headers.get(
                    "x-greenway-content-hash"
                ),
                "prompt": task["prompt"],
                "effective_prompt": expected_prompt,
                "prompt_tokens_tiktoken_estimate": count_tokens(task["prompt"], MODEL),
                "raw_response": payload,
                "metrics": response_metrics(
                    payload, int(payload.get("usage", {}).get("prompt_tokens", 0))
                ),
                "gateway": {
                    "token_count_source": response.headers.get("x-greenway-token-source"),
                    "telemetry_source": response.headers.get("x-greenway-telemetry-source"),
                    "energy_source_tier": response.headers.get("x-greenway-energy-source"),
                    "energy_kwh": response.headers.get("x-greenway-energy-kwh"),
                    "carbon_source_tier": response.headers.get("x-greenway-carbon-source"),
                    "carbon_gco2e": response.headers.get("x-greenway-carbon-gco2e"),
                },
                "attempt_numbers": [
                    index
                    for index, attempt in enumerate(attempts, start=1)
                    if attempt["sample_number"] == task["sample_number"]
                ],
            }
            results.append(record)
            persist_json(
                args.output,
                {
                    "execution": execution,
                    "protocol": {
                        **protocol,
                    },
                    "samples": results,
                    "attempts": attempts,
                },
            )
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if actual_arm != task["arm"]:
                raise RuntimeError(
                    f"Gateway selected arm {actual_arm!r}, expected {task['arm']!r}; "
                    "the response and attempt are saved, stopping the run"
                )
            if actual_changes != expected_changes:
                raise RuntimeError(
                    f"Gateway prompt changes {actual_changes!r} differ from locally "
                    f"expected {expected_changes!r}; stopping before the next request"
                )
            if record["effective_prompt_content_hash"] != expected_content_hash:
                raise RuntimeError(
                    "Gateway effective-prompt hash differs from the local preview; "
                    "stopping before the next request"
                )

    rubric = json.loads(DEFAULT_RUBRIC_FILE.read_text(encoding="utf-8"))
    score_sheet, assignments = create_blinded_sheet({"samples": results}, rubric)
    for sample in results:
        sample["coverage_blind_id"] = assignments[str(sample["sample_number"])]
    persist_json(args.pass1_output, copy.deepcopy(score_sheet))
    persist_json(args.pass2_output, copy.deepcopy(score_sheet))
    persist_json(
        args.output,
        {
            "execution": execution,
            "protocol": protocol,
            "samples": results,
            "attempts": attempts,
            "scoring_sheets": {
                "agent_pass1": args.pass1_output.name,
                "independent_pass2": args.pass2_output.name,
            },
            "evaluation": None,
        },
    )
    print(
        json.dumps(
            {
                "results": str(args.output),
                "agent_pass1_sheet": str(args.pass1_output),
                "independent_pass2_sheet": str(args.pass2_output),
                "next_step": (
                    "Score both shuffled sheets independently, record any "
                    "disagreement quote in pass 1, then apply both score sets."
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-gateway", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--optimized-gateway", default="http://127.0.0.1:8002/v1/chat/completions")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--attempts-output", type=Path, default=DEFAULT_ATTEMPTS_FILE)
    parser.add_argument("--pass1-output", type=Path, default=DEFAULT_PASS1_FILE)
    parser.add_argument("--pass2-output", type=Path, default=DEFAULT_PASS2_FILE)
    parser.add_argument("--quota-remaining", type=int)
    parser.add_argument("--word-cap", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(main(parser.parse_args()))
