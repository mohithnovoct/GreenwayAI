import json

import pytest

from telemetry.parametric import ParametricEnergyEstimator


@pytest.fixture
def estimator_config(tmp_path):
    config_path = tmp_path / "coefficients.json"
    config_path.write_text(
        json.dumps(
            {
                "throughput_fallback": {
                    "prefill_kwh_per_input_token": 1.0e-7,
                    "decode_kwh_per_output_token": 2.0e-7,
                },
                "models": {
                    "vendor-model": {
                        "tier": "vendor",
                        "prefill_kwh_per_input_token": 2.0e-7,
                        "decode_kwh_per_output_token": 5.0e-7,
                    },
                    "lca-model": {
                        "tier": "lca",
                        "prefill_kwh_per_input_token": 3.0e-7,
                        "decode_kwh_per_output_token": 7.0e-7,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_vendor_coefficients_are_used_with_exact_energy(estimator_config):
    estimate = ParametricEnergyEstimator(estimator_config).estimate(
        model="vendor-model", input_tokens=10, output_tokens=4
    )

    assert estimate.energy_source_tier == "vendor"
    assert estimate.energy_kwh == (10 * 2.0e-7) + (4 * 5.0e-7)


def test_lca_coefficients_are_used_with_exact_energy(estimator_config):
    estimate = ParametricEnergyEstimator(estimator_config).estimate(
        model="lca-model", input_tokens=10, output_tokens=4
    )

    assert estimate.energy_source_tier == "lca"
    assert estimate.energy_kwh == (10 * 3.0e-7) + (4 * 7.0e-7)


def test_missing_model_uses_loud_throughput_fallback(estimator_config, caplog):
    with caplog.at_level("WARNING", logger="greenwayai.telemetry"):
        estimate = ParametricEnergyEstimator(estimator_config).estimate(
            model="unconfigured-model", input_tokens=10, output_tokens=4
        )

    assert estimate.energy_source_tier == "throughput_fallback"
    assert estimate.energy_kwh == (10 * 1.0e-7) + (4 * 2.0e-7)
    assert "low-confidence throughput fallback" in caplog.text
