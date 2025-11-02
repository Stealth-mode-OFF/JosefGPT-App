from __future__ import annotations

import importlib
class FakeCollection:
    def __init__(self, key):
        self.key = key
        self.entries = []

    def add(self, documents, embeddings, metadatas, ids):
        for doc, emb, meta, item_id in zip(documents, embeddings, metadatas, ids):
            self.entries.append(
                {
                    "document": doc,
                    "embedding": emb,
                    "metadata": meta,
                    "id": item_id,
                }
            )

    def delete(self, where):
        source = where.get("source")
        if source:
            self.entries = [
                entry for entry in self.entries if entry["metadata"].get("source") != source
            ]

    def query(self, query_embeddings, n_results, include):
        selected = self.entries[:n_results]
        documents = [entry["document"] for entry in selected]
        metadatas = [entry["metadata"] for entry in selected]
        ids = [entry["id"] for entry in selected]
        distances = [idx * 0.1 for idx, _ in enumerate(selected)]
        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "ids": [ids],
            "distances": [distances],
        }


class FakeClient:
    _registry: dict[tuple[str, str], FakeCollection] = {}

    def __init__(self, path: str):
        self.path = path

    def get_or_create_collection(self, name: str):
        key = (self.path, name)
        if key not in self._registry:
            self._registry[key] = FakeCollection(key)
        return self._registry[key]


def test_ingest_and_answer(monkeypatch, tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "sample.txt").write_text(
        "Josef loves building scalable businesses.\n"
        "Automation unlocks leverage for sales and brand.",
        encoding="utf-8",
    )

    embeddings_dir = tmp_path / "embeddings"
    monkeypatch.setenv("SOURCE_DIRS", str(source_dir))
    monkeypatch.setenv("EMBEDDINGS_PATH", str(embeddings_dir))
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("USE_OPENAI_EMBEDDINGS", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODE", "offline")

    config = importlib.import_module("app.config")
    importlib.reload(config)

    chromadb = importlib.import_module("chromadb")
    monkeypatch.setattr(chromadb, "PersistentClient", FakeClient)

    ingest_books = importlib.import_module("ingest_books")
    importlib.reload(ingest_books)

    query_engine = importlib.import_module("app.query_engine")
    importlib.reload(query_engine)

    class DummyEmbedding(list):
        def tolist(self):
            return list(self)

    class DummyEncoder:
        def encode(self, texts, show_progress_bar=False):
            return [
                DummyEmbedding([float(i + 1)] * 3) for i, _ in enumerate(texts)
            ]

    dummy_encoder = DummyEncoder()
    monkeypatch.setattr(ingest_books, "get_local_encoder", lambda: dummy_encoder)
    monkeypatch.setattr(
        ingest_books, "embed_chunks", lambda chunks: [[1.0, 1.0, 1.0]] * len(chunks)
    )
    monkeypatch.setattr(query_engine, "encoder", dummy_encoder)

    result = ingest_books.ingest_all([source_dir])
    assert result["files"] == 1
    assert result["chunks"] > 0

    response = query_engine.answer_with_context("How can Josef scale sales?")
    assert "Automation" in response["answer"]
    assert response["sources"]
    assert response["sources"][0]["source"].endswith("sample.txt")
    assert response["llm"]["mode"] == "offline"
