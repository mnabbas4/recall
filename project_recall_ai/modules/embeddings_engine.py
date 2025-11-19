# modules/embeddings_engine.py
import os
import json
from pathlib import Path
from openai import OpenAI
import streamlit as st

# Secure API key handling - uses Streamlit secrets or environment variable
def get_api_key():
    """Get OpenAI API key from Streamlit secrets or environment"""
    try:
        # Try Streamlit secrets first
        return st.secrets.get("OPENAI_API_KEY")
    except:
        # Fall back to environment variable
        return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_api_key()

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

class EmbeddingsEngine:
    def __init__(self):
        if client is None:
            raise ValueError("OPENAI_API_KEY is not set. Please configure it in Streamlit secrets or environment variables.")
        self.client = client

    def _text_for_row(self, row):
        """Combine row data into searchable text"""
        cols = ['Project Category','Project Reference','Phase','Problems Encountered','Solutions Adopted']
        return " || ".join([str(row.get(c,'')) for c in cols if row.get(c,'')])

    def embed_texts(self, texts):
        """
        Generate embeddings for list of texts
        Returns: list of embeddings (lists of floats)
        """
        resp = self.client.embeddings.create(
            model="text-embedding-3-large", 
            input=texts
        )
        return [item.embedding for item in resp.data]

    def index_dataframe(self, memory_path, df, id_prefix='mem'):
        """
        Create and save embeddings for a dataframe
        
        Args:
            memory_path: path to saved dataframe file (e.g., data/memories/memory_1.parquet)
            df: pandas DataFrame to index
            id_prefix: prefix for embedding file name
            
        Returns:
            str: path to saved embeddings file
        """
        texts = [self._text_for_row(r) for _, r in df.iterrows()]
        embeddings = self.embed_texts(texts)

        mem_path = Path(memory_path)
        mem_id = mem_path.stem  # e.g., memory_1
        out = mem_path.with_suffix('').parent / f"{mem_id}_embeddings.json"
        out.write_text(json.dumps(embeddings))
        return str(out)
