import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

from ingress import app as ingress_app
from optimizers.semantic_cache import SentenceTransformerEmbedder


def test_embedder_normalizes_model_output_to_float_list(monkeypatch):
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_id, revision, device):
            calls["model_id"] = model_id
            calls["revision"] = revision
            calls["device"] = device

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, text, **kwargs):
            calls["text"] = text
            calls["encode_kwargs"] = kwargs
            return [0.25, -0.5, 0.75]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    embedder = SentenceTransformerEmbedder()
    vector = embedder.encode("example")

    assert calls == {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "device": "cpu",
        "text": "example",
        "encode_kwargs": {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        },
    }
    assert embedder.dimension == 3
    assert vector == [0.25, -0.5, 0.75]


def test_enabled_cache_loads_embedding_model_during_application_startup(monkeypatch):
    loaded = []

    class FakeEmbedder:
        def __init__(self, model_id, revision):
            loaded.append((model_id, revision))
            self.model_id = model_id
            self.dimension = 384

    monkeypatch.setenv("CACHE_ENABLED", "true")
    monkeypatch.setenv("CACHE_EMBEDDING_MODEL_ID", "test-embedding-model")
    monkeypatch.setenv("CACHE_EMBEDDING_MODEL_REVISION", "test-revision")
    monkeypatch.setattr(ingress_app, "SentenceTransformerEmbedder", FakeEmbedder)

    with TestClient(ingress_app.app):
        assert loaded == [("test-embedding-model", "test-revision")]
        assert ingress_app.app.state.semantic_embedder.model_id == "test-embedding-model"


def test_disabled_cache_does_not_load_embedding_model(monkeypatch):
    loaded = []

    class UnexpectedEmbedder:
        def __init__(self, model_id, revision):
            loaded.append(model_id)

    monkeypatch.setenv("CACHE_ENABLED", "false")
    monkeypatch.setattr(ingress_app, "SentenceTransformerEmbedder", UnexpectedEmbedder)

    with TestClient(ingress_app.app):
        assert loaded == []
        assert ingress_app.app.state.semantic_embedder is None
