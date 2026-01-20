import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple

from utils_io import load_json_array, event_to_text, transcript_to_text

# Sentence-Transformers
from sentence_transformers import SentenceTransformer

# OpenAI
from openai import OpenAI


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def build_faiss_ip_index(vectors: np.ndarray) -> faiss.Index:
    """
    Inner product index. If vectors are L2-normalized, IP ~= cosine similarity.
    """
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors.astype(np.float32))
    return index


def write_meta_jsonl(meta_path: str, metadatas: List[Dict[str, Any]]) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadatas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def embed_sentence_transformers(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    # encode returns numpy array if convert_to_numpy True
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # important for cosine/IP
    )
    return embs.astype(np.float32)


def embed_openai(client: OpenAI, model_name: str, texts: List[str], batch_size: int = 128) -> np.ndarray:
    """
    Calls OpenAI embeddings endpoint in batches.
    """
    all_vecs: List[List[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"OpenAI embed ({model_name})"):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        # resp.data is in the same order as input
        batch_vecs = [item.embedding for item in resp.data]
        all_vecs.extend(batch_vecs)
    vecs = np.array(all_vecs, dtype=np.float32)
    # normalize for cosine similarity on IP index
    vecs = l2_normalize(vecs)
    return vecs


def make_chunks(
    events_path: str,
    transcripts_path: str
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Builds one combined corpus (events + transcripts).
    Each JSON object becomes one chunk.
    """
    chunks: List[str] = []
    metas: List[Dict[str, Any]] = []

    # Events
    events = load_json_array(events_path)
    for obj in events:
        txt = event_to_text(obj)
        chunks.append(txt)
        metas.append({
            "source": "events",
            "primary_id": obj.get("Id"),
            "raw": obj
        })

    # Transcripts
    transcripts = load_json_array(transcripts_path)
    for obj in transcripts:
        txt = transcript_to_text(obj)
        chunks.append(txt)
        metas.append({
            "source": "transcripts",
            "primary_id": obj.get("id"),
            "raw": obj
        })

    return chunks, metas


def main() -> None:
    load_dotenv()

    events_path = os.path.join("data", "telecom_call_center_events_100.json")
    transcripts_path = os.path.join("data", "telecom_transcripts_100.json")

    out_dir = "indexes"
    os.makedirs(out_dir, exist_ok=True)

    texts, metas = make_chunks(events_path, transcripts_path)

    # ---------- 1) Sentence-Transformers index ----------
    st_model_name = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    st_model = SentenceTransformer(st_model_name)

    st_vecs = embed_sentence_transformers(st_model, texts, batch_size=64)
    st_index = build_faiss_ip_index(st_vecs)

    st_index_path = os.path.join(out_dir, "faiss_st.index")
    st_meta_path = os.path.join(out_dir, "st_meta.jsonl")

    faiss.write_index(st_index, st_index_path)
    write_meta_jsonl(st_meta_path, metas)

    # ---------- 2) OpenAI embeddings index ----------
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is missing in environment (.env).")

    openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    client = OpenAI(api_key=openai_key)

    oa_vecs = embed_openai(client, openai_embed_model, texts, batch_size=128)
    oa_index = build_faiss_ip_index(oa_vecs)

    oa_index_path = os.path.join(out_dir, "faiss_openai.index")
    oa_meta_path = os.path.join(out_dir, "openai_meta.jsonl")

    faiss.write_index(oa_index, oa_index_path)
    write_meta_jsonl(oa_meta_path, metas)

    print("\n✅ Done.")
    print(f"ST index:     {st_index_path}")
    print(f"ST metadata:  {st_meta_path}")
    print(f"OpenAI index: {oa_index_path}")
    print(f"OpenAI meta:  {oa_meta_path}")


if __name__ == "__main__":
    main()
