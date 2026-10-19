from ingress import interleaved_ab, load_test


def _no_meminfo(*args, **kwargs):
    raise OSError("/proc/meminfo unavailable")


def test_load_test_resource_state_handles_missing_load_average(monkeypatch):
    monkeypatch.delattr(load_test.os, "getloadavg", raising=False)
    monkeypatch.setattr(load_test, "open", _no_meminfo, raising=False)

    load_average, available_memory = load_test.resource_state()

    assert load_average is None
    assert available_memory is None


def test_interleaved_resource_state_handles_missing_load_average(monkeypatch):
    monkeypatch.delattr(interleaved_ab.os, "getloadavg", raising=False)
    monkeypatch.setattr(interleaved_ab, "open", _no_meminfo, raising=False)

    load_average, available_memory = interleaved_ab.resource_state()

    assert load_average is None
    assert available_memory is None
    assert interleaved_ab.format_resource_state("test").startswith(
        "test: load_avg_1m=unavailable"
    )
