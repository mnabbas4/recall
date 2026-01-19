# modules/embeddings_engine.py

import os
import json
from pathlib import Path
from openai import OpenAI
import streamlit as st

# =====================================================
# API KEY HANDLING
# =====================================================
def get_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except:
        return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_api_key()

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None


# =====================================================
# EMBEDDINGS ENGINE
# =====================================================
class EmbeddingsEngine:
    """
    Responsible for:
    - Building semantic text per row
    - Generating embeddings
    - Persisting them safely
    """

    # Only semantic columns (VERY IMPORTANT)
    SEMANTIC_COLS = [
        'TIPO MACCHINA',
        'APPLICAZIONE',
        'TIPO PROBLEMA',
        'DESCRIZIONE',
        'SOLUZIONE LESSON LEARNED'
    ]

    def __init__(self):
        if client is None:
            raise ValueError(
                "OPENAI_API_KEY is not set. Configure it in Streamlit secrets or environment variables."
            )
        self.client = client

    # -------------------------------------------------
    def _text_for_row(self, row) -> str:
        """
        Build clean semantic text for one row.
        Metadata is intentionally excluded.
        """
        parts = []
        for col in self.SEMANTIC_COLS:
            val = str(row.get(col, "")).strip()
            if val:
                parts.append(f"{col}: {val}")

        return " | ".join(parts)

    # -------------------------------------------------
    def embed_texts(self, texts):
        """
        Generate embeddings for list of texts.
        Empty texts are skipped upstream.
        """
        resp = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [item.embedding for item in resp.data]

    # -------------------------------------------------
    def index_dataframe(self, memory_path, df, id_prefix="mem"):
        """
        Create and save embeddings for a dataframe.

        Output format:
        {
            "row_ids": [...],
            "embeddings": [...]
        }
        """
        texts = []
        row_ids = []

        for idx, row in df.iterrows():
            text = self._text_for_row(row)
            if text:
                texts.append(text)
                row_ids.append(int(idx))

        if not texts:
            raise ValueError("No semantic content found to embed.")

        embeddings = self.embed_texts(texts)

        mem_path = Path(memory_path)
        mem_id = mem_path.stem

        out_path = mem_path.parent / f"{mem_id}_embeddings.json"

        payload = {
            "model": "text-embedding-3-large",
            "semantic_columns": self.SEMANTIC_COLS,
            "row_ids": row_ids,
            "embeddings": embeddings
        }

        out_path.write_text(json.dumps(payload, ensure_ascii=False))

        return str(out_path)
