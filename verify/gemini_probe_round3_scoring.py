"""Create blinded coverage sheets and attach independently assigned scores."""

import argparse
import json
import secrets
from pathlib import Path
from typing import Any

DEFAULT_RUBRIC_FILE = Path(__file__).with_name("gemini_probe_round3_rubric.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_blinded_sheet(
    results: dict[str, Any],
    rubric: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, Any], dict[str, str]]:
    random_source = secrets.SystemRandom()
    cases: list[dict[str, Any]] = []
    sample_to_blind_id: dict[str, str] = {}
    used_ids: set[str] = set()

    for sample in results["samples"]:
        shape = sample["shape"]
        if shape not in rubric:
            continue
        blind_id = f"case-{secrets.token_hex(6)}"
        while blind_id in used_ids:
            blind_id = f"case-{secrets.token_hex(6)}"
        used_ids.add(blind_id)
        sample_to_blind_id[str(sample["sample_number"])] = blind_id
        cases.append(
            {
                "blind_id": blind_id,
                "shape": shape,
                "prompt": sample["prompt"],
                "criteria": rubric[shape],
                "response_text": sample["raw_response"]["choices"][0]["message"]["content"],
                "scores": {
                    criterion["id"]: None for criterion in rubric[shape]
                },
            }
        )

    random_source.shuffle(cases)
    return {"cases": cases}, sample_to_blind_id


def apply_blinded_scores(
    results: dict[str, Any],
    sheet: dict[str, Any],
    rubric: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    rubric = rubric or load_json(DEFAULT_RUBRIC_FILE)
    sample_by_blind_id = {
        sample["coverage_blind_id"]: sample
        for sample in results["samples"]
        if sample.get("coverage_blind_id")
    }
    cases = sheet.get("cases", [])
    if len(cases) != len(sample_by_blind_id):
        raise ValueError("Scored sheet must contain every blinded response exactly once")

    seen: set[str] = set()
    for case in cases:
        blind_id = case.get("blind_id")
        if blind_id not in sample_by_blind_id or blind_id in seen:
            raise ValueError(f"Unknown or duplicate blind ID: {blind_id!r}")
        seen.add(blind_id)
        shape = sample_by_blind_id[blind_id]["shape"]
        criteria = case.get("criteria", [])
        expected_ids = {criterion["id"] for criterion in rubric[shape]}
        scores = case.get("scores", {})
        if set(scores) != expected_ids or any(
            type(value) is not int or value not in (0, 1)
            for value in scores.values()
        ):
            raise ValueError(
                f"{blind_id}: score each rubric item exactly 0 or 1"
            )
        if {criterion["id"] for criterion in criteria} != expected_ids:
            raise ValueError(f"{blind_id}: rubric criteria do not match the preregistration")
        sample = sample_by_blind_id[blind_id]
        sample["coverage_scores"] = scores
        sample["coverage_score"] = sum(scores.values())
        sample["coverage_max_score"] = len(expected_ids)
        sample["coverage_fraction"] = (
            sum(scores.values()) / len(expected_ids) if expected_ids else None
        )
    return results


def score_sheet(results_path: Path, sheet_path: Path) -> None:
    results = load_json(results_path)
    rubric = load_json(DEFAULT_RUBRIC_FILE)
    sheet, assignments = create_blinded_sheet(results, rubric)
    for sample in results["samples"]:
        if str(sample["sample_number"]) in assignments:
            sample["coverage_blind_id"] = assignments[str(sample["sample_number"])]
    save_json(results_path, results)
    save_json(sheet_path, sheet)


def apply_scores(results_path: Path, sheet_path: Path) -> None:
    results = load_json(results_path)
    scored = apply_blinded_scores(
        results,
        load_json(sheet_path),
        load_json(DEFAULT_RUBRIC_FILE),
    )
    from verify.gemini_probe_round3 import evaluate_results

    scored["evaluation"] = evaluate_results(scored["samples"])
    save_json(results_path, scored)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    blind_parser = subparsers.add_parser("blind")
    blind_parser.add_argument("--results", type=Path, required=True)
    blind_parser.add_argument("--sheet", type=Path, required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--results", type=Path, required=True)
    apply_parser.add_argument("--sheet", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "blind":
        score_sheet(args.results, args.sheet)
    else:
        apply_scores(args.results, args.sheet)


if __name__ == "__main__":
    main()
