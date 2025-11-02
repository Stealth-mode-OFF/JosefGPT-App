import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import chromadb
import ebooklib
import pdfplumber
from ebooklib import epub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.config import get_settings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

settings = get_settings()
USE_OPENAI_EMBEDDINGS = settings.use_openai_embeddings
EMBEDDING_MODEL = settings.embedding_model
SOURCE_DIRS = settings.source_dirs
SUPPORTED_SUFFIXES = {suffix.lower() for suffix in settings.supported_suffixes}
TEXT_SUFFIXES = {suffix.lower() for suffix in settings.text_suffixes}

_local_encoder: Optional[SentenceTransformer] = None
_openai_client: Optional[OpenAI] = None


@dataclass
class IngestionStats:
    scanned: int = 0
    files: int = 0
    chunks: int = 0
    skipped: int = 0
    failed: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "files": self.files,
            "chunks": self.chunks,
            "scanned": self.scanned,
            "skipped": self.skipped,
            "failed": self.failed,
        }


class BookIngestor:
    def __init__(self, source_dirs: Optional[Iterable[Path]] = None):
        self.directories: List[Path] = [
            Path(directory).expanduser() for directory in (source_dirs or SOURCE_DIRS)
        ]
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.client = chromadb.PersistentClient(path=settings.embeddings_path.as_posix())
        self.collection = self.client.get_or_create_collection("josef_knowledge")
        self.stats = IngestionStats()

    def run(self) -> Dict[str, int]:
        files = list(iter_source_files(self.directories))
        if not files:
            print("⚠️ No sources found. Add files into 'books/', 'texts/' or 'data/'.")
            return self.stats.as_dict()

        for path, base_dir in tqdm(files, desc="Ingesting", unit="file"):
            self.stats.scanned += 1
            src = source_key(path, base_dir)
            try:
                signature = _file_signature(path)
            except OSError as exc:
                self.stats.failed += 1
                print(f"⚠️ {src}: cannot read file metadata ({exc}).")
                continue

            if not self._needs_refresh(src, signature):
                self.stats.skipped += 1
                print(f"ℹ️ {src}: up-to-date, skipping.")
                continue

            text = extract_text(path)
            if not text.strip():
                self.stats.skipped += 1
                print(f"⚠️ {src}: no readable text (maybe scan/OCR needed).")
                continue

            chunks = self._split_text(text)
            if not chunks:
                self.stats.skipped += 1
                print(f"⚠️ {src}: splitter produced no chunks.")
                continue

            try:
                embeddings = embed_chunks(chunks)
            except Exception as exc:  # pragma: no cover - defensive
                self.stats.failed += 1
                print(f"⚠️ {src}: embedding generation failed ({exc}).")
                continue

            if not embeddings:
                self.stats.failed += 1
                print(f"⚠️ {src}: embedding generation yielded no vectors.")
                continue

            self._persist(src, chunks, embeddings, signature)
            self.stats.files += 1
            self.stats.chunks += len(chunks)
            print(f"✅ {src}: stored {len(chunks)} chunks.")

        print(
            f"🏁 Done. {self.stats.chunks} chunks saved from {self.stats.files} files "
            f"(scanned {self.stats.scanned}, skipped {self.stats.skipped}, failed {self.stats.failed})."
        )
        return self.stats.as_dict()

    def _needs_refresh(self, source: str, signature: Dict[str, int]) -> bool:
        stored_mtime, stored_size = _existing_signature(self.collection, source)
        if stored_mtime is None or stored_size is None:
            return True
        return not (
            stored_mtime == signature.get("mtime") and stored_size == signature.get("size")
        )

    def _split_text(self, text: str) -> List[str]:
        chunks = self.splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _persist(
        self,
        source: str,
        chunks: List[str],
        embeddings: List[List[float]],
        signature: Dict[str, int],
    ) -> None:
        self.collection.delete(where={"source": source})
        base_metadata = {
            "source": source,
            "file_mtime": signature.get("mtime"),
            "file_size": signature.get("size"),
        }
        metadatas = [{**base_metadata, "chunk": idx} for idx in range(len(chunks))]
        ids = [f"{source}#{idx}" for idx in range(len(chunks))]
        MAX_BATCH_SIZE = 1000
        for i in range(0, len(chunks), MAX_BATCH_SIZE):
            batch_chunks = chunks[i : i + MAX_BATCH_SIZE]
            batch_embeddings = embeddings[i : i + MAX_BATCH_SIZE]
            batch_metadatas = metadatas[i : i + MAX_BATCH_SIZE]
            batch_ids = ids[i : i + MAX_BATCH_SIZE]
            self.collection.add(
                documents=batch_chunks,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )


def iter_source_files(directories: Iterable[Path]) -> Iterator[Tuple[Path, Path]]:
    for base_dir in directories:
        base_dir = Path(base_dir).expanduser()
        if not base_dir.exists():
            continue
        for path in base_dir.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_SUFFIXES
                and not path.name.startswith(".")
            ):
                yield path, base_dir


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".epub":
        return _extract_epub(path)
    if suffix in TEXT_SUFFIXES:
        return _read_text_file(path)
    return ""


def _extract_pdf(path: Path) -> str:
    try:
        with pdfplumber.open(path) as doc:
            texts = []
            for page in doc.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    texts.append(page_text)
            return "\n".join(texts)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️ {path.name}: PDF parsing failed ({exc}).")
        return ""


def _extract_epub(path: Path) -> str:
    try:
        book = epub.read_epub(path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️ {path.name}: EPUB parsing failed ({exc}).")
        return ""

    texts: List[str] = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            try:
                texts.append(item.get_body_content().decode("utf-8", errors="ignore"))
            except Exception:
                continue
    return "\n".join(texts)


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️ {path.name}: text file read failed ({exc}).")
        return ""


def get_local_encoder() -> SentenceTransformer:
    global _local_encoder
    if _local_encoder is None:
        _local_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_encoder


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    if USE_OPENAI_EMBEDDINGS:
        client = get_openai_client()
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
        return [item.embedding for item in resp.data]
    encoder = get_local_encoder()
    return encoder.encode(chunks, show_progress_bar=False).tolist()


def source_key(path: Path, base_dir: Path) -> str:
    return f"{base_dir.name}/{path.relative_to(base_dir).as_posix()}"


def _file_signature(path: Path) -> Dict[str, int]:
    stats = path.stat()
    return {"mtime": int(stats.st_mtime), "size": stats.st_size}


def _existing_signature(collection, source: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        existing = collection.get(where={"source": source}, include=["metadatas"], limit=1)
    except Exception:
        return None, None
    metadatas = existing.get("metadatas") or []
    if not metadatas:
        return None, None
    metadata = metadatas[0] if isinstance(metadatas[0], dict) else None
    if not isinstance(metadata, dict):
        return None, None
    return metadata.get("file_mtime"), metadata.get("file_size")


def ingest_all(source_dirs: Optional[Iterable[Path]] = None) -> Dict[str, int]:
    ingestor = BookIngestor(source_dirs)
    return ingestor.run()


def get_ingestion_report(source_dirs: Optional[Iterable[Path]] = None):
    directories = [Path(directory).expanduser() for directory in (source_dirs or SOURCE_DIRS)]
    client = chromadb.PersistentClient(path=settings.embeddings_path.as_posix())
    collection = client.get_or_create_collection("josef_knowledge")
    report = {"up_to_date": [], "stale": [], "pending": []}

    for path, base_dir in iter_source_files(directories):
        src = source_key(path, base_dir)
        try:
            signature = _file_signature(path)
        except OSError:
            continue
        stored_mtime, stored_size = _existing_signature(collection, src)
        entry = {
            "path": path,
            "source": src,
            "current_mtime": signature["mtime"],
            "current_size": signature["size"],
            "stored_mtime": stored_mtime,
            "stored_size": stored_size,
        }
        if stored_mtime is None or stored_size is None:
            report["pending"].append(entry)
        elif stored_mtime == signature["mtime"] and stored_size == signature["size"]:
            report["up_to_date"].append(entry)
        else:
            report["stale"].append(entry)
    return report


def main():
    ingest_all()


if __name__ == "__main__":
    main()
