# modules/recall_engine.py

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from textblob import TextBlob
from rapidfuzz import fuzz
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

_OPENAI_KEY = get_api_key()
_openai_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None


# =====================================================
# UTILS
# =====================================================
def _cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =====================================================
# RECALL ENGINE
# =====================================================
class RecallEngine:
    def __init__(self, emb_engine, mem_manager,
                 phase_match_threshold=75,
                 category_match_threshold=75):

        self.emb_engine = emb_engine
        self.mem_manager = mem_manager
        self.phase_threshold = phase_match_threshold
        self.category_threshold = category_match_threshold

    # -------------------------------------------------
    def _correct_spelling(self, text):
        try:
            return str(TextBlob(text).correct())
        except Exception:
            return text

    # -------------------------------------------------
    def _load_embeddings(self, mem_id):
        path = Path(self.mem_manager.base) / "memories" / f"{mem_id}_embeddings.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())

        if not isinstance(payload, dict):
            return None
        if "embeddings" not in payload or "row_ids" not in payload:
            return None

        return payload

    # -------------------------------------------------
    def _extract_context_hints(self, query, df):
        q = query.lower()
        matched_phase = None
        matched_category = None

        applicazioni = [
            str(x) for x in pd.unique(df["APPLICAZIONE"].dropna())
            if str(x).strip()
        ]
        macchine = [
            str(x) for x in pd.unique(df["TIPO MACCHINA"].dropna())
            if str(x).strip()
        ]

        for a in applicazioni:
            if fuzz.partial_ratio(a.lower(), q) >= self.phase_threshold:
                matched_phase = a
                break

        for m in macchine:
            if fuzz.partial_ratio(m.lower(), q) >= self.category_threshold:
                matched_category = m
                break

        return matched_phase, matched_category

    # -------------------------------------------------
    def query_memory(
        self,
        mem_id,
        query,
        min_score=0.25,
        spell_correction=True,
        hard_limit=None,
        enforce_context=False,
        weight_text=0.70,
        weight_phase=0.15,
        weight_category=0.15,
        fallback_k=3,
    ):

        q_text = self._correct_spelling(query) if spell_correction else query

        df = self.mem_manager.load_memory_dataframe(mem_id)
        if df is None or df.empty:
            return pd.DataFrame()

        payload = self._load_embeddings(mem_id)
        if payload is None:
            raise FileNotFoundError(
                f"Embeddings for memory '{mem_id}' not found or invalid."
            )

        embeddings = payload["embeddings"]
        row_ids = payload["row_ids"]

        matched_phase, matched_category = self._extract_context_hints(q_text, df)

        # Candidate indices are embedding indices
        candidate_idxs = list(range(len(embeddings)))

        if enforce_context:
            filtered = []
            for emb_idx in candidate_idxs:
                df_idx = row_ids[emb_idx]
                row = df.loc[df_idx]

                if matched_phase and str(row["APPLICAZIONE"]).lower() != matched_phase.lower():
                    continue
                if matched_category and str(row["TIPO MACCHINA"]).lower() != matched_category.lower():
                    continue

                filtered.append(emb_idx)

            candidate_idxs = filtered or candidate_idxs

        q_emb = self.emb_engine.embed_texts([q_text])[0]

        scored = []
        for emb_idx in candidate_idxs:
            df_idx = row_ids[emb_idx]
            row = df.loc[df_idx]

            text_score = _cosine_similarity(q_emb, embeddings[emb_idx])

            phase_bonus = 0.0
            if matched_phase:
                s = fuzz.partial_ratio(
                    str(row["APPLICAZIONE"]).lower(),
                    matched_phase.lower()
                )
                phase_bonus = 1.0 if s >= 85 else 0.5 if s >= 65 else 0.0

            category_bonus = 0.0
            if matched_category:
                s = fuzz.partial_ratio(
                    str(row["TIPO MACCHINA"]).lower(),
                    matched_category.lower()
                )
                category_bonus = 1.0 if s >= 85 else 0.5 if s >= 65 else 0.0

            final_score = (
                weight_text * text_score +
                weight_phase * phase_bonus +
                weight_category * category_bonus
            )

            scored.append({
                "df_idx": df_idx,
                "TextScore": round(text_score, 4),
                "PhaseBonus": phase_bonus,
                "CategoryBonus": category_bonus,
                "FinalScore": round(final_score, 4),
            })

        scored_df = pd.DataFrame(scored).sort_values("FinalScore", ascending=False)
        matches = scored_df[scored_df["FinalScore"] >= min_score]

        used_fallback = False
        if matches.empty:
            used_fallback = True
            matches = scored_df.sort_values("TextScore", ascending=False).head(fallback_k)

        if hard_limit:
            matches = matches.head(hard_limit)

        out_rows = []
        for _, r in matches.iterrows():
            row = df.loc[r["df_idx"]]
            rec = row.to_dict()
            rec.update({
                "FinalScore": r["FinalScore"],
                "TextScore": r["TextScore"],
                "PhaseBonus": r["PhaseBonus"],
                "CategoryBonus": r["CategoryBonus"],
            })
            out_rows.append(rec)

        out_df = pd.DataFrame(out_rows)
        out_df.attrs["used_fallback"] = used_fallback
        out_df.attrs["matched_phase"] = matched_phase
        out_df.attrs["matched_category"] = matched_category

        return out_df
