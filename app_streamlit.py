import os
import json
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ----------------------------------------------------
# Load .env (optional, for local dev)
# ----------------------------------------------------
load_dotenv()

# ====================================================
# HELPER FUNCTIONS
# ====================================================

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def load_meta_jsonl(path: str):
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def embed_query_st(model, query: str) -> np.ndarray:
    v = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    return v

def embed_query_openai(client, model_name: str, query: str) -> np.ndarray:
    resp = client.embeddings.create(model=model_name, input=[query])
    v = np.array([resp.data[0].embedding], dtype=np.float32)
    v = l2_normalize(v)
    return v

def search(index, qvec, k=5):
    scores, ids = index.search(qvec, k)
    return scores[0], ids[0]

def format_hit(meta):
    raw = meta["raw"]
    return {
        "source": meta["source"],
        "primary_id": meta["primary_id"],
        "score": round(meta["score"], 4),
        "raw": raw
    }

def build_llm_answer(client, model, query, hits):
    context_blocks = []
    for i, h in enumerate(hits, 1):
        context_blocks.append(
            f"[{i}] source={h['source']} id={h['primary_id']} | "
            f"raw={json.dumps(h['raw'], ensure_ascii=False)}"
        )
    context = "\n".join(context_blocks)

    prompt = (
        "You are a support analytics assistant.\n"
        "Use ONLY the provided context to answer. "
        "If not present, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query}\n"
    )

    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    return resp.output_text

# ====================================================
# STREAMLIT APP
# ====================================================

def main():
    st.set_page_config(page_title="Dual-Index RAG Comparator", layout="wide")

    st.title("🔍 Embedding Models Comparison")

    # ------------------------------------------------
    # SIDEBAR SETTINGS (INCLUDING API KEY INPUT)
    # ------------------------------------------------
    with st.sidebar:
        st.header("Settings")

        # ----- OPENAI KEY (UI INPUT, NO VALIDATION YET) -----
        st.subheader("🔑 OpenAI API Key")
        secret_key = None

        # Try Streamlit secrets (if deployed)
        try:
            secret_key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            secret_key = None

        env_key = os.getenv("OPENAI_API_KEY")

        # UI input (only used if no secrets or .env)
        user_key = st.text_input(
            "Paste your OpenAI API key",
            type="password",
            help="Key is used only for this session.",
        )

        # Decide which key to use
        raw_key = secret_key or env_key or user_key

        # CLEAN THE KEY (CRITICAL FIX)
        OPENAI_API_KEY = raw_key.strip() if isinstance(raw_key, str) else None

        # Store for downstream use
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""


        if secret_key:
            st.success("Using Streamlit Secrets API key")
        # elif env_key:
        #     st.success("Using local .env API key")
        elif user_key:
            st.info("Using user-provided API key")


        # ----- RAG SETTINGS -----
        st.subheader("RAG Config")

        k = st.slider("Top-K retrieved chunks", min_value=1, max_value=10, value=5)

        st_model_name = st.text_input(
            "Sentence-Transformer Model",
            value="sentence-transformers/all-MiniLM-L6-v2"
        )

        openai_embed_model = st.text_input(
            "OpenAI Embedding Model",
            value=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        )

        openai_chat_model = st.text_input(
            "OpenAI Chat Model",
            value=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        )

    # ------------------------------------------------
    # LOAD INDEXES (CACHED)
    # ------------------------------------------------
    @st.cache_resource
    def load_indexes(st_model_name: str):
        idx_dir = "indexes"

        # Auto-build indexes if missing (Streamlit Cloud case)
        if not os.path.exists(os.path.join(idx_dir, "faiss_st.index")):
            st.warning("Indexes not found — building them now...")
            from build_indexes import main as build_indexes
            build_indexes()

        st_index = faiss.read_index(os.path.join(idx_dir, "faiss_st.index"))
        st_meta = load_meta_jsonl(os.path.join(idx_dir, "st_meta.jsonl"))

        oa_index = faiss.read_index(os.path.join(idx_dir, "faiss_openai.index"))
        oa_meta = load_meta_jsonl(os.path.join(idx_dir, "openai_meta.jsonl"))

        st_model = SentenceTransformer(st_model_name)

        # DO NOT create OpenAI client yet (we validate key later)
        return st_index, st_meta, oa_index, oa_meta, st_model

    st_index, st_meta, oa_index, oa_meta, st_model = load_indexes(st_model_name)

    # ------------------------------------------------
    # USER QUESTION
    # ------------------------------------------------
    question = st.text_input(
        "Enter your question:",
        placeholder="Example: Which agent handled customer with broadband 80020000008?"
    )

    if st.button("Run RAG Comparison") and question.strip():

        # ===== VALIDATE API KEY ONLY NOW =====
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            st.error("❌ Please enter your OpenAI API key in the sidebar first.")
            st.stop()

        openai_client = OpenAI(api_key=api_key)

        # -------- Sentence-Transformer Retrieval --------
        with st.spinner("Retrieving with Sentence-Transformers..."):
            qvec_st = embed_query_st(st_model, question)
            scores_st, ids_st = search(st_index, qvec_st, k)

            st_hits = []
            for score, idx in zip(scores_st, ids_st):
                if idx >= 0:
                    m = st_meta[int(idx)].copy()
                    m["score"] = float(score)
                    st_hits.append(format_hit(m))

            st_answer = build_llm_answer(
                openai_client, openai_chat_model, question, st_hits
            )

        # -------- OpenAI Embedding Retrieval --------
        with st.spinner("Retrieving with OpenAI embeddings..."):
            qvec_oa = embed_query_openai(
                openai_client, openai_embed_model, question
            )
            scores_oa, ids_oa = search(oa_index, qvec_oa, k)

            oa_hits = []
            for score, idx in zip(scores_oa, ids_oa):
                if idx >= 0:
                    m = oa_meta[int(idx)].copy()
                    m["score"] = float(score)
                    oa_hits.append(format_hit(m))

            oa_answer = build_llm_answer(
                openai_client, openai_chat_model, question, oa_hits
            )

        # -------- Display Results (Side-by-Side) --------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🟦 Sentence-Transformers")

            st.markdown("**LLM Answer (Grounded in ST Retrieval)**")
            st.markdown(f"> {st_answer}")

            st.markdown("**Retrieved Chunks**")
            for i, h in enumerate(st_hits, 1):
                with st.expander(f"Chunk {i} | score={h['score']}"):
                    st.json(h)

        with col2:
            st.subheader("🟪 OpenAI Embeddings")

            st.markdown("**LLM Answer (Grounded in OpenAI Retrieval)**")
            st.markdown(f"> {oa_answer}")

            st.markdown("**Retrieved Chunks**")
            for i, h in enumerate(oa_hits, 1):
                with st.expander(f"Chunk {i} | score={h['score']}"):
                    st.json(h)

    else:
        st.info("Enter a question and click **Run RAG Comparison**.")

if __name__ == "__main__":
    main()
