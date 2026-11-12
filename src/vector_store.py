"""ChromaDB vector database layer for PatentQ prior-art storage & semantic retrieval."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from src.models import PriorArtDocument


class DeterministicFallbackEmbedding(EmbeddingFunction[Documents]):
    """Lightweight offline fallback embedding function if ONNX model is unavailable."""

    def __init__(self, dims: int = 128) -> None:
        self.dims = dims

    def __call__(self, input: Documents) -> Embeddings:
        vectors: Embeddings = []
        for text in input:
            vec = [0.0] * self.dims
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:2], "big") % self.dims
                sign = 1.0 if digest[2] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


def _build_embedding_function() -> tuple[EmbeddingFunction[Documents], str]:
    """Use ChromaDB's local ONNX all-MiniLM-L6-v2 embedding model when available."""
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        ef = DefaultEmbeddingFunction()
        return ef, "ChromaDB ONNX (all-MiniLM-L6-v2)"
    except Exception:
        return DeterministicFallbackEmbedding(), "Offline Deterministic Hash Fallback"


_DEFAULT_PERSIST = "__DEFAULT_PERSIST__"


class ChromaPriorArtStore:
    """Manages persistent ChromaDB storage and similarity search for prior-art excerpts."""

    def __init__(
        self,
        persist_dir: Optional[str] = _DEFAULT_PERSIST,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        resolved_dir = CHROMA_PERSIST_DIR if persist_dir == _DEFAULT_PERSIST else persist_dir
        self.persist_dir = resolved_dir
        self.collection_name = collection_name
        self.embedding_fn, self.embedding_backend_name = _build_embedding_function()

        settings = Settings(anonymized_telemetry=False)
        if resolved_dir:
            self.client = chromadb.PersistentClient(path=resolved_dir, settings=settings)
        else:
            self.client = chromadb.EphemeralClient(settings=settings)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine", "app": "PatentQ"},
        )

    def count(self) -> int:
        return self.collection.count()

    def upsert_documents(self, docs: List[PriorArtDocument]) -> int:
        """Store or update prior-art documents in ChromaDB."""
        if not docs:
            return 0
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        seen = set()

        for doc in docs:
            if doc.doc_id in seen:
                continue
            seen.add(doc.doc_id)
            ids.append(doc.doc_id)
            documents.append(doc.to_embedding_text())
            metadatas.append(
                {
                    "doc_id": doc.doc_id,
                    "doc_type": doc.doc_type,
                    "title": doc.title[:500],
                    "abstract": doc.abstract[:1500],
                    "authors_or_assignee": doc.authors_or_assignee[:300],
                    "year": doc.year[:20],
                    "url": doc.url[:500],
                    "image_url": doc.image_url[:500],
                    "price": doc.price[:80],
                    "source_api": doc.source_api[:80],
                }
            )

        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def similarity_search(
        self,
        query: str,
        top_k: int = 6,
        doc_type_filter: Optional[str] = None,
    ) -> List[PriorArtDocument]:
        """Retrieve top-k semantically similar prior-art records from ChromaDB."""
        total = self.count()
        if total == 0 or not query.strip():
            return []

        n_results = min(top_k, total)
        where = {"doc_type": doc_type_filter} if doc_type_filter in {"patent", "paper", "product", "custom"} else None
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["metadatas", "distances", "documents"],
            )
        except Exception:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["metadatas", "distances", "documents"],
            )

        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        output: List[PriorArtDocument] = []

        for meta, dist in zip(metadatas, distances):
            if not meta:
                continue
            # Cosine distance in [0, 2] -> similarity score in [0, 1]
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            output.append(
                PriorArtDocument(
                    doc_id=str(meta.get("doc_id", "UNKNOWN")),
                    doc_type=meta.get("doc_type", "paper"),
                    title=str(meta.get("title", "Untitled")),
                    abstract=str(meta.get("abstract", "")),
                    authors_or_assignee=str(meta.get("authors_or_assignee", "Unknown")),
                    year=str(meta.get("year", "N/A")),
                    url=str(meta.get("url", "")),
                    image_url=str(meta.get("image_url", "")),
                    price=str(meta.get("price", "")),
                    source_api=str(meta.get("source_api", "ChromaDB")),
                    similarity_score=round(similarity, 4),
                )
            )
        return output

    def list_all_documents(self, limit: int = 100) -> List[PriorArtDocument]:
        """Return all stored documents (up to limit) for inspection in the UI."""
        total = self.count()
        if total == 0:
            return []
        data = self.collection.get(limit=min(limit, total), include=["metadatas"])
        docs: List[PriorArtDocument] = []
        for meta in data.get("metadatas") or []:
            if not meta:
                continue
            docs.append(
                PriorArtDocument(
                    doc_id=str(meta.get("doc_id", "UNKNOWN")),
                    doc_type=meta.get("doc_type", "paper"),
                    title=str(meta.get("title", "Untitled")),
                    abstract=str(meta.get("abstract", "")),
                    authors_or_assignee=str(meta.get("authors_or_assignee", "Unknown")),
                    year=str(meta.get("year", "N/A")),
                    url=str(meta.get("url", "")),
                    image_url=str(meta.get("image_url", "")),
                    price=str(meta.get("price", "")),
                    source_api=str(meta.get("source_api", "ChromaDB")),
                )
            )
        return docs

    def delete_document(self, doc_id: str) -> None:
        if doc_id:
            self.collection.delete(ids=[doc_id])

    def reset(self) -> None:
        """Delete and recreate the ChromaDB collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine", "app": "PatentQ"},
        )

    def as_langchain_retriever(self, top_k: int = 6) -> RunnableLambda:
        """Expose ChromaDB retrieval as a LangChain LCEL Runnable."""

        def _retrieve(query: str) -> List[Document]:
            hits = self.similarity_search(query, top_k=top_k)
            return [
                Document(
                    page_content=hit.to_embedding_text(),
                    metadata={
                        "doc_id": hit.doc_id,
                        "doc_type": hit.doc_type,
                        "title": hit.title,
                        "url": hit.url,
                        "year": hit.year,
                        "source_api": hit.source_api,
                        "similarity_score": hit.similarity_score,
                    },
                )
                for hit in hits
            ]

        return RunnableLambda(_retrieve)
