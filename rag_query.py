import os
import json
import numpy as np
import faiss
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from openai import OpenAI


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def load_meta_jsonl(path: str) -> List[Dict[str, Any]]:
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas


def embed_query_st(model: SentenceTransformer, query: str) -> np.ndarray:
    v = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    return v


def embed_query_openai(client: OpenAI, model_name: str, query: str) -> np.ndarray:
    resp = client.embeddings.create(model=model_name, input=[query])
    v = np.array([resp.data[0].embedding], dtype=np.float32)
    v = l2_normalize(v)
    return v


def search(index: faiss.Index, qvec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    scores, ids = index.search(qvec, k)
    return scores[0], ids[0]


def format_context(hits: List[Dict[str, Any]]) -> str:
    # Keep context compact but useful
    blocks = []
    for i, h in enumerate(hits, start=1):
        raw = h["raw"]
        source = h["source"]
        pid = h["primary_id"]
        blocks.append(f"[{i}] source={source} id={pid} | raw={json.dumps(raw, ensure_ascii=False)}")
    return "\n".join(blocks)


def generate_answer_openai(client: OpenAI, chat_model: str, query: str, context: str) -> str:
    prompt = (
        "You are a support analytics assistant.\n"
        "Use ONLY the provided context to answer. If not present, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query}\n"
    )

    # Using Responses API (recommended by OpenAI docs)
    resp = client.responses.create(
        model=chat_model,
        input=prompt,
    )
    return resp.output_text


def main() -> None:
    load_dotenv()

    # Choose which index to query: "st" or "openai"
    index_choice = os.getenv("INDEX_CHOICE", "st").lower()

    k = int(os.getenv("TOP_K", "5"))
    query = os.getenv("QUERY", "Why did the payment fail? Provide the related account number.")

    idx_dir = "indexes"

    if index_choice == "st":
        index_path = os.path.join(idx_dir, "faiss_st.index")
        meta_path = os.path.join(idx_dir, "st_meta.jsonl")

        st_model_name = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        st_model = SentenceTransformer(st_model_name)

        index = faiss.read_index(index_path)
        metas = load_meta_jsonl(meta_path)

        qvec = embed_query_st(st_model, query)
        scores, ids = search(index, qvec, k)

    elif index_choice == "openai":
        index_path = os.path.join(idx_dir, "faiss_openai.index")
        meta_path = os.path.join(idx_dir, "openai_meta.jsonl")

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is missing in environment (.env).")

        embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        client = OpenAI(api_key=openai_key)

        index = faiss.read_index(index_path)
        metas = load_meta_jsonl(meta_path)

        qvec = embed_query_openai(client, embed_model, query)
        scores, ids = search(index, qvec, k)

    else:
        raise ValueError("INDEX_CHOICE must be either 'st' or 'openai'")

    hits = []
    for score, idx in zip(scores, ids):
        if idx < 0:
            continue
        m = metas[int(idx)]
        m = {**m, "score": float(score)}
        hits.append(m)

    print("\n=== TOP HITS ===")
    for h in hits:
        print(f"- score={h['score']:.4f} source={h['source']} id={h['primary_id']}")

    context = format_context(hits)
    print("\n=== CONTEXT (raw) ===")
    print(context)

    # Optional: generate an answer using OpenAI (set GENERATE_ANSWER=1)
    if os.getenv("GENERATE_ANSWER", "0") == "1":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is missing in environment (.env).")

        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        client = OpenAI(api_key=openai_key)

        answer = generate_answer_openai(client, chat_model, query, context)
        print("\n=== ANSWER ===")
        print(answer)


if __name__ == "__main__":
    main()
