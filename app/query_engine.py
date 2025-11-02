from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.llm import get_chat_llm

settings = get_settings()

OPENAI_MODEL = settings.openai_model
DEFAULT_TOP_K = settings.top_k
DEFAULT_MAX_TOKENS = settings.max_tokens
DEFAULT_TEMPERATURE = settings.temperature

encoder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path=settings.embeddings_path.as_posix())
collection = db.get_or_create_collection("josef_knowledge")
llm = get_chat_llm()

SYSTEM_PROMPT = (
    "You are Josef's elite business coach. "
    "Use insights from context only, focused on scaling, sales, psychology, negotiation, brand and automation."
)


def _distance_to_score(distance: Any) -> Optional[float]:
    if isinstance(distance, (int, float)):
        return max(0.0, min(1.0, 1.0 - float(distance)))
    return None


def retrieve_context(question: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    q_emb = encoder.encode([question])[0].tolist()
    try:
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    contexts: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        metadata = metas[idx] if idx < len(metas) else {}
        distance = distances[idx] if idx < len(distances) else None
        entry: Dict[str, Any] = {
            "id": ids[idx] if idx < len(ids) else None,
            "text": doc,
            "metadata": metadata or {},
        }
        if distance is not None:
            entry["distance"] = distance
            score = _distance_to_score(distance)
            if score is not None:
                entry["score"] = score
        contexts.append(entry)
    return contexts


def _format_prompt_context(contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return "No relevant context was retrieved from the knowledge base."
    blocks = []
    for idx, ctx in enumerate(contexts, start=1):
        snippet = ctx.get("text", "").strip()
        blocks.append(f"[Source {idx}]\n{snippet}")
    return "\n---\n".join(blocks)


def _summarise_sources(contexts: List[Dict[str, Any]], preview_chars: int = 260):
    summaries: List[Dict[str, Any]] = []
    seen = set()
    for ctx in contexts:
        meta = ctx.get("metadata") or {}
        source_id = meta.get("source")
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        preview = (ctx.get("text") or "").strip().replace("\n", " ")
        if len(preview) > preview_chars:
            preview = preview[:preview_chars].rstrip() + "..."
        summaries.append(
            {
                "source": source_id,
                "chunk": meta.get("chunk"),
                "score": ctx.get("score"),
                "preview": preview,
            }
        )
    return summaries


def build_user_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx = _format_prompt_context(contexts)
    return f"Context:\n{ctx}\n\nQuestion: {question}"


def answer_with_context(
    question: str,
    *,
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    effective_top_k = DEFAULT_TOP_K if top_k is None else int(top_k)
    effective_temperature = (
        DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    )
    effective_max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else int(max_tokens)

    contexts = retrieve_context(question, effective_top_k)
    user_prompt = build_user_prompt(question, contexts)
    raw_answer = llm.generate(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=effective_temperature,
        max_tokens=effective_max_tokens,
    )
    source_summaries = _summarise_sources(contexts)
    if "Sources" not in raw_answer and source_summaries:
        sources_text = "\n".join(summary["source"] for summary in source_summaries[:5])
        answer = f"{raw_answer}\n\nSources:\n{sources_text}"
    else:
        answer = raw_answer
    return {
        "question": question,
        "answer": answer,
        "raw_answer": raw_answer,
        "sources": source_summaries,
        "contexts": contexts,
        "prompt": user_prompt,
        "config": {
            "top_k": effective_top_k,
            "temperature": effective_temperature,
            "max_tokens": effective_max_tokens,
        },
        "llm": {"mode": llm.mode, "model": getattr(llm, "model_name", OPENAI_MODEL)},
    }


def answer_with_gpt5(question: str) -> str:
    result = answer_with_context(question)
    return result["answer"]


if __name__ == "__main__":
    print("🧠 JosefGPT (Hybrid local retrieval + GPT-5 reasoning). Type 'exit' to quit.")
    while True:
        q = input("\n🧠 Josef> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print(answer_with_gpt5(q))
