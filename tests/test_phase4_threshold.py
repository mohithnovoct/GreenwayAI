import json
from collections import Counter

import pytest

from verify import phase4_threshold_validation as v


def test_wilson_interval_matches_closed_form_for_zero_successes():
    low, high = v.wilson_interval(0, 230)
    z2 = 1.959963984540054**2
    assert low == 0.0
    assert high == pytest.approx(z2 / (230 + z2))
    assert 0.0164 < high < 0.0165


def test_one_false_positive_in_230_exceeds_the_two_percent_bound():
    assert v.wilson_interval(1, 230)[1] > v.MAX_TEST_FPR_UPPER


def test_guard_blocks_number_and_negation_changes_but_not_paraphrases():
    assert not v.guard_passes("Convert 5 kilometers to miles.", "Convert 50 kilometers to miles.")
    assert not v.guard_passes("Explain when you should use recursion.", "Explain when you should not use recursion.")
    assert not v.guard_passes("What should I do if it compiles?", "What should I do if it doesn't compile?")
    assert v.guard_passes("How do I reverse a string?", "What is the way to reverse a string?")
    assert v.guard_passes("It is not safe.", "It cannot be called safe.")


def test_choose_threshold_is_the_smallest_grid_value_within_the_fpr_limit():
    # Two hits in 100 is 2% (> 1%); one hit in 100 is 1% (<= 1%), first reached at 0.91.
    sims = [0.6] * 98 + [0.9, 0.95]
    assert v.choose_threshold(sims, v.GRID, 0.01) == 0.91
    assert v.choose_threshold([0.99] * 10, v.GRID, 0.01) is None


def test_split_is_deterministic_stratified_and_leaves_fixed_splits_alone():
    def make():
        rows = [
            {"id": f"qqp{i}", "source": "qqp", "label": i % 2, "a": "a", "b": "b"} for i in range(40)
        ]
        rows.append({"id": "hbp01", "source": "handbuilt", "label": 1, "a": "a", "b": "b", "split": "test"})
        return rows

    first = v.split_pairs(make())
    second = v.split_pairs(make())
    assert [p["split"] for p in first] == [p["split"] for p in second]
    assert Counter((p["label"], p["split"]) for p in first if p["source"] == "qqp") == {
        (0, "dev"): 10,
        (0, "test"): 10,
        (1, "dev"): 10,
        (1, "test"): 10,
    }
    assert first[-1]["split"] == "test"


def test_handbuilt_file_has_the_preregistered_counts_and_balanced_splits():
    pairs = json.loads(v.HANDBUILT_PATH.read_text(encoding="utf-8"))["pairs"]
    assert Counter(p["label"] for p in pairs) == {1: 40, 0: 60}
    assert Counter((p["label"], p["split"]) for p in pairs) == {
        (0, "dev"): 30,
        (0, "test"): 30,
        (1, "dev"): 20,
        (1, "test"): 20,
    }
    assert len({p["id"] for p in pairs}) == 100


def _pairs(adv_neg_test_sims, adv_neg_dev_sims, pos_test_sims, pos_dev_sims):
    pairs = []
    for split, label, sims in (
        ("dev", 0, adv_neg_dev_sims),
        ("test", 0, adv_neg_test_sims),
        ("dev", 1, pos_dev_sims),
        ("test", 1, pos_test_sims),
    ):
        for i, sim in enumerate(sims):
            pairs.append(
                {"id": f"{split}{label}{i}", "source": "paws", "label": label, "split": split, "sim": sim}
            )
    return pairs


def test_rule_enables_when_safe_and_useful():
    pairs = _pairs([0.4] * 230, [0.4] * 200, [0.9] * 30 + [0.3] * 70, [0.9] * 100)
    result = v.evaluate_arm(pairs, lambda p, t: p["sim"] >= t)
    assert result["chosen_threshold"] == 0.5
    assert result["enable_by_default"] is True


def test_rule_disables_when_one_adversarial_test_false_positive_breaks_the_bound():
    test_negatives = [0.4] * 229 + [0.95]
    pairs = _pairs(test_negatives, [0.4] * 200, [0.9] * 100, [0.9] * 100)
    result = v.evaluate_arm(pairs, lambda p, t: p["sim"] >= t)
    assert result["chosen_threshold"] == 0.5
    assert result["test"]["adversarial_fpr"]["hits"] == 1
    assert result["enable_by_default"] is False


def test_rule_disables_when_safe_only_because_it_never_hits():
    pairs = _pairs([0.4] * 230, [0.4] * 200, [0.3] * 95 + [0.9] * 5, [0.3] * 100)
    result = v.evaluate_arm(pairs, lambda p, t: p["sim"] >= t)
    assert result["test"]["paraphrase_hit_rate_pooled"]["rate"] < v.MIN_PARAPHRASE_HIT_RATE
    assert result["enable_by_default"] is False


def test_rule_has_no_threshold_when_dev_negatives_overlap_every_grid_value():
    pairs = _pairs([0.4] * 10, [0.999] * 50, [0.9] * 10, [0.9] * 10)
    result = v.evaluate_arm(pairs, lambda p, t: p["sim"] >= t)
    assert result["chosen_threshold"] is None
    assert result["enable_by_default"] is False
