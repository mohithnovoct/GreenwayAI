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
                "scores": {criterion["id"]: None for criterion in rubric[shape]},
            }
        )

    random_source.shuffle(cases)
    return {"cases": cases}, sample_to_blind_id


def apply_blinded_scores(
    results: dict[str, Any],
    pass1_sheet: dict[str, Any],
    pass2_sheet: dict[str, Any],
    rubric: dict[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    rubric = rubric or load_json(DEFAULT_RUBRIC_FILE)
    sample_by_blind_id = {
        sample["coverage_blind_id"]: sample
        for sample in results["samples"]
        if sample.get("coverage_blind_id")
    }
    pass1_by_id = validate_score_sheet(results, pass1_sheet, rubric, "agent_pass1")
    pass2_by_id = validate_score_sheet(results, pass2_sheet, rubric, "independent_pass2")
    pass1_cases_by_id = {
        case["blind_id"]: case for case in pass1_sheet.get("cases", [])
    }
    agreement: dict[str, dict[str, Any]] = {}
    for blind_id, sample in sample_by_blind_id.items():
        shape = sample["shape"]
        item_results: dict[str, Any] = {}
        final_scores: dict[str, int] = {}
        for criterion in rubric[shape]:
            item_id = criterion["id"]
            first_score = pass1_by_id[blind_id][item_id]
            second_score = pass2_by_id[blind_id][item_id]
            item_results[item_id] = {
                "agent_pass1": first_score,
                "independent_pass2": second_score,
                "agreement": first_score == second_score,
            }
            if first_score == second_score:
                final_scores[item_id] = first_score
                continue

            resolution = (
                pass1_cases_by_id[blind_id]
                .get("disagreement_resolutions", {})
                .get(item_id)
            )
            if not resolution:
                raise ValueError(
                    f"{blind_id}/{item_id}: disagreement needs a quoted-text resolution"
                )
            quote = resolution.get("evidence_quote", "")
            final_score = resolution.get("final_score")
            response_text = sample["raw_response"]["choices"][0]["message"]["content"]
            if not quote or quote not in response_text:
                raise ValueError(
                    f"{blind_id}/{item_id}: resolution quote must occur verbatim in the response"
                )
            if type(final_score) is not int or final_score not in (0, 1):
                raise ValueError(
                    f"{blind_id}/{item_id}: resolution final_score must be 0 or 1"
                )
            item_results[item_id]["resolution"] = resolution
            final_scores[item_id] = final_score

        sample["coverage_scores_agent_pass1"] = pass1_by_id[blind_id]
        sample["coverage_scores_independent_pass2"] = pass2_by_id[blind_id]
        sample["coverage_scores_final"] = final_scores
        sample["coverage_scores"] = final_scores
        sample["coverage_score"] = sum(final_scores.values())
        sample["coverage_max_score"] = len(final_scores)
        sample["coverage_fraction"] = (
            sum(final_scores.values()) / len(final_scores) if final_scores else None
        )
        sample["coverage_item_agreement"] = item_results

    for shape, criteria in rubric.items():
        shape_ids = [
            blind_id
            for blind_id, sample in sample_by_blind_id.items()
            if sample["shape"] == shape
        ]
        if not shape_ids:
            continue
        agreement[shape] = {}
        for criterion in criteria:
            item_id = criterion["id"]
            agreed = sum(
                pass1_by_id[blind_id][item_id] == pass2_by_id[blind_id][item_id]
                for blind_id in shape_ids
            )
            agreement[shape][item_id] = {
                "agreements": agreed,
                "responses_scored": len(shape_ids),
                "raw_percent_agreement": round(100 * agreed / len(shape_ids), 1),
            }
    results["coverage_inter_rater_agreement"] = agreement
    return results


def validate_score_sheet(
    results: dict[str, Any],
    sheet: dict[str, Any],
    rubric: dict[str, list[dict[str, str]]],
    pass_name: str,
) -> dict[str, dict[str, int]]:
    sample_by_blind_id = {
        sample["coverage_blind_id"]: sample
        for sample in results["samples"]
        if sample.get("coverage_blind_id")
    }
    cases = sheet.get("cases", [])
    if len(cases) != len(sample_by_blind_id):
        raise ValueError(f"{pass_name}: sheet must contain every response exactly once")
    scores_by_id: dict[str, dict[str, int]] = {}
    for case in cases:
        blind_id = case.get("blind_id")
        if blind_id not in sample_by_blind_id or blind_id in scores_by_id:
            raise ValueError(f"{pass_name}: unknown or duplicate blind ID {blind_id!r}")
        sample = sample_by_blind_id[blind_id]
        shape = sample["shape"]
        expected_criteria = rubric[shape]
        expected_ids = {criterion["id"] for criterion in expected_criteria}
        if {criterion["id"] for criterion in case.get("criteria", [])} != expected_ids:
            raise ValueError(f"{pass_name}/{blind_id}: criteria do not match the preregistered rubric")
        if case.get("response_text") != sample["raw_response"]["choices"][0]["message"]["content"]:
            raise ValueError(f"{pass_name}/{blind_id}: response text does not match saved response")
        scores = case.get("scores", {})
        if set(scores) != expected_ids or any(
            type(value) is not int or value not in (0, 1)
            for value in scores.values()
        ):
            raise ValueError(f"{pass_name}/{blind_id}: score each item exactly 0 or 1")
        scores_by_id[blind_id] = scores
    return scores_by_id


def apply_scores(
    results_path: Path,
    pass1_path: Path,
    pass2_path: Path,
) -> None:
    results = load_json(results_path)
    pass1_sheet = load_json(pass1_path)
    pass2_sheet = load_json(pass2_path)
    scored = apply_blinded_scores(
        results,
        pass1_sheet,
        pass2_sheet,
        load_json(DEFAULT_RUBRIC_FILE),
    )
    from verify.gemini_probe_round3 import evaluate_results

    scored["evaluation"] = evaluate_results(scored["samples"])
    save_json(results_path, scored)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--results", type=Path, required=True)
    apply_parser.add_argument("--pass1", type=Path, required=True)
    apply_parser.add_argument("--pass2", type=Path, required=True)
    args = parser.parse_args()
    apply_scores(args.results, args.pass1, args.pass2)


if __name__ == "__main__":
    main()
