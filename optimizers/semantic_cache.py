import logging
from typing import Any


logger = logging.getLogger("greenwayai.semantic_cache")
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        revision: str = DEFAULT_MODEL_REVISION,
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "semantic embedding model loading model_id=%s revision=%s device=%s",
            model_id,
            revision,
            device,
        )
        self._model: Any = SentenceTransformer(
            model_id,
            revision=revision,
            device=device,
        )
        self.model_id = model_id
        self.revision = revision
        self.dimension = self._model.get_embedding_dimension()
        if not self.dimension:
            raise RuntimeError(f"Embedding model {model_id!r} did not report its dimension")
        logger.info(
            "semantic embedding model loaded model_id=%s revision=%s dimensions=%s device=%s",
            model_id,
            revision,
            self.dimension,
            device,
        )

    def encode(self, text: str) -> list[float]:
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [float(value) for value in embedding]
