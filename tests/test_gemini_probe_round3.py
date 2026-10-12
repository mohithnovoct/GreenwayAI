import copy

from verify.gemini_probe_round3 import build_schedule, evaluate_results
from verify.gemini_probe_round3_scoring import (
    apply_blinded_scores,
    create_blinded_sheet,
)


PROMPTS = [
    {"shape": "focused_explanation", "prompt": "explain focused"},
    {"shape": "short_how_to", "prompt": "give a how-to"},
    {"shape": "one_paragraph_analysis", "prompt": "analyze briefly"},
]


def test_round_three_schedule_has_eight_counterbalanced_requests():
    schedule = build_schedule(PROMPTS)

    assert len(schedule) == 8
    for shape in ("focused_explanation", "short_how_to"):
        shape_tasks = [
            task
            for task in schedule
            if task["shape"] == shape and task["measurement_type"] == "paired"
        ]
        shape_arms = [task["arm"] for task in shape_tasks]
        assert shape_arms.count("baseline") == 2
        assert shape_arms.count("optimized") == 2
        for pair_number in (1, 2):
            pair_arms = [
                task["arm"]
                for task in shape_tasks
                if task["pair_number"] == pair_number
            ]
            assert pair_arms in (["baseline", "optimized"], ["optimized", "baseline"])


def make_samples(
    *,
    optimized_words: int = 150,
    optimized_coverage: int = 5,
    baseline_words: tuple[int, int] = (200, 204),
    baseline_coverage: int = 5,
):
    samples = []
    shapes = ("focused_explanation", "short_how_to")
    sample_number = 0
    for shape in shapes:
        for pair_number in (1, 2):
            for arm in (("baseline", "optimized") if pair_number == 1 else ("optimized", "baseline")):
                sample_number += 1
                baseline_index = pair_number - 1
                is_optimized = arm == "optimized"
                samples.append(
                    {
                        "sample_number": sample_number,
                        "shape": shape,
                        "measurement_type": "paired",
                        "pair_number": pair_number,
                        "arm": arm,
                        "metrics": {
                            "finish_reason": "stop",
                            "word_count": (
                                optimized_words
                                if is_optimized
                                else baseline_words[baseline_index]
                            ),
                        },
                        "prompt_optimization_changes": (
                            ["word-cap-added"] if is_optimized else []
                        ),
                        "coverage_score": (
                            optimized_coverage if is_optimized else baseline_coverage
                        ),
                    }
                )
    return samples


def test_shape_qualifies_only_if_complete_changed_shorter_and_coverage_preserved():
    result = evaluate_results(make_samples())

    assert result["both_shapes_qualify"] is True
    assert all(shape["baseline_word_spread"] == 4 for shape in result["shapes"])

    samples = make_samples(optimized_coverage=4)
    result = evaluate_results(samples)
    assert result["both_shapes_qualify"] is False
    assert all(
        shape["finding"] == "shorter but less complete"
        for shape in result["shapes"]
    )

    samples[0]["metrics"]["finish_reason"] = "length"
    result = evaluate_results(samples)
    assert result["shapes"][0]["qualifies"] is False
    assert result["shapes"][0]["finding"] == "does not qualify"

    samples = make_samples()
    samples[1]["prompt_optimization_changes"] = []
    result = evaluate_results(samples)
    assert result["shapes"][0]["qualifies"] is False


def test_coverage_sheet_blinds_arm_and_applies_scores():
    rubric = {
        "focused_explanation": [
            {"id": "definition", "criterion": "Defines the concept."},
            {"id": "example", "criterion": "Gives an example."},
        ]
    }
    results = {
        "samples": [
            {
                "sample_number": 1,
                "shape": "focused_explanation",
                "arm": "baseline",
                "prompt": "Explain it.",
                "raw_response": {
                    "choices": [{"message": {"content": "Response A"}}]
                },
            },
            {
                "sample_number": 2,
                "shape": "focused_explanation",
                "arm": "optimized",
                "prompt": "Explain it.",
                "raw_response": {
                    "choices": [{"message": {"content": "Response B"}}]
                },
            },
        ]
    }
    sheet, assignments = create_blinded_sheet(results, rubric)
    assert len(sheet["cases"]) == 2
    assert all("arm" not in case and "sample_number" not in case for case in sheet["cases"])
    pass1_sheet = copy.deepcopy(sheet)
    pass2_sheet = copy.deepcopy(sheet)
    for case in pass1_sheet["cases"]:
        case["scores"] = {"definition": 1, "example": 0}
        case["disagreement_resolutions"] = {
            "example": {"evidence_quote": "Response", "final_score": 0}
        }
    for case in pass2_sheet["cases"]:
        case["scores"] = {"definition": 1, "example": 1}

    scored_results = copy.deepcopy(results)
    for sample in scored_results["samples"]:
        sample["coverage_blind_id"] = assignments[str(sample["sample_number"])]
    apply_blinded_scores(scored_results, pass1_sheet, pass2_sheet, rubric)

    assert all(sample["coverage_score"] == 1 for sample in scored_results["samples"])
    assert all(sample["coverage_max_score"] == 2 for sample in scored_results["samples"])
    assert all(
        sample["coverage_scores_agent_pass1"] == {"definition": 1, "example": 0}
        for sample in scored_results["samples"]
    )
    assert all(
        sample["coverage_scores_independent_pass2"] == {"definition": 1, "example": 1}
        for sample in scored_results["samples"]
    )
    agreement = scored_results["coverage_inter_rater_agreement"]["focused_explanation"]
    assert agreement["definition"]["raw_percent_agreement"] == 100.0
    assert agreement["example"]["raw_percent_agreement"] == 0.0
