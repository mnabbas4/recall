# apma_app.py — Deployment Ready (Append-Safe + Dynamic Semantic Embeddings)

print("hello")

import streamlit as st
import os
import pandas as pd
from streamlit import rerun
from datetime import date

from modules.file_manager import MemoryManager
from modules.data_handler import DataHandler
from modules.embeddings_engine import EmbeddingsEngine
from modules.recall_engine import RecallEngine
from modules.utils import ensure_data_dirs
from modules import user_manager
from modules.manual_config import load_config, save_config

# =====================================================
# CONFIGURATION
# =====================================================
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
except:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

REQUIRED_COLS = [
    'COMMESSA',
    'CLIENTE',
    'ANNO',
    'TIPO MACCHINA',
    'APPLICAZIONE',
    'TIPO PROBLEMA',
    'DESCRIZIONE',
    'SOLUZIONE LESSON LEARNED',
    'DATA INSERIMENTO',
    'RCPRD',
    'REPORT CANTIERE',
    'CONCERNED DEPARTMENTS',
    'REPORT RIUNIONE CHIUSURA PROGETTO'
]

# =====================================================
# HELPERS
# =====================================================
def build_semantic_text(df: pd.DataFrame) -> pd.Series:
    """
    Build semantic text from user-configured columns.
    """
    cfg = load_config()

    semantic_cols = [
        c for c, meta in cfg.items()
        if c in df.columns and meta.get("type") in ("text", "select", "date")
    ]

    if not semantic_cols:
        semantic_cols = list(df.columns)

    return (
        df[semantic_cols]
        .astype(str)
        .fillna("")
        .agg(" | ".join, axis=1)
    )

def append_to_memory(mem_manager, memory_name, new_df):
    try:
        existing = mem_manager.load_memory_dataframe(memory_name)
        if existing is not None and not existing.empty:
            return pd.concat([existing, new_df], ignore_index=True)
    except FileNotFoundError:
        pass  # memory does not exist yet → create new

    return new_df


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="APMA — AI Project Memory Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

ensure_data_dirs()
st.title("🧠 AI Project Memory Assistant (APMA)")

# =====================================================
# AUTH
# =====================================================
if 'user' not in st.session_state:
    st.session_state['user'] = None

if st.session_state['user'] is None:
    st.sidebar.header("🔐 Login")

    auth_mode = st.sidebar.radio("Mode", ["Login", "Create account"])

    if auth_mode == "Create account":
        fn = st.sidebar.text_input("First name")
        ln = st.sidebar.text_input("Last name")
        uid = st.sidebar.text_input("ID Number")
        pw = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Create"):
            ok, msg = user_manager.create_user(fn, ln, uid, pw)
            st.sidebar.success(msg) if ok else st.sidebar.error(msg)

    else:
        uid = st.sidebar.text_input("ID Number")
        pw = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            ok, msg, profile = user_manager.authenticate(uid, pw)
            if ok:
                st.session_state['user'] = profile
                rerun()
            else:
                st.sidebar.error(msg)

# =====================================================
# CORE OBJECTS
# =====================================================
mem_manager = MemoryManager(data_dir="data")

emb_engine = None
if OPENAI_API_KEY:
    try:
        emb_engine = EmbeddingsEngine()
    except:
        emb_engine = None

recall_engine = RecallEngine(
    emb_engine=emb_engine,
    mem_manager=mem_manager,
    category_col="TIPO MACCHINA",
    phase_col="APPLICAZIONE",
    problem_col="DESCRIZIONE",
    solution_col="SOLUZIONE LESSON LEARNED"
)

# =====================================================
# USER BAR
# =====================================================
if st.session_state.get("user"):
    u = st.session_state["user"]
    col1, col2 = st.columns([4, 1])
    col1.markdown(f"**Logged in:** {u['first_name']} {u['last_name']} ({u['id']})")
    if col2.button("Logout"):
        st.session_state['user'] = None
        rerun()

# =====================================================
# MODE SELECT
# =====================================================
mode = st.sidebar.selectbox("📂 Mode", [
    "Upload / Update Memory",
    "Query Knowledge Base",
    "Settings"
])

# =====================================================
# UPLOAD / UPDATE MEMORY
# =====================================================
if mode == "Upload / Update Memory":
    st.header("📤 Upload / Update Memory")

    uploaded = st.file_uploader("Upload CSV / Excel", ["csv", "xlsx"])

    if uploaded:
        df, err = DataHandler.read_and_validate(uploaded, required_cols=REQUIRED_COLS)
        if err:
            st.error(err)
        else:
            st.dataframe(df.head(), use_container_width=True)
            mem_name = st.text_input("Memory name")

            if st.button("Save"):
                df["AddedBy"] = st.session_state["user"]["id"]
                df["__semantic_text__"] = build_semantic_text(df)

                df = append_to_memory(mem_manager, mem_name, df)
                meta = mem_manager.create_or_update_memory(mem_name, df)

                if emb_engine:
                    emb_engine.index_dataframe(meta["memory_path"], df)

                st.success("Memory saved and indexed")

    # ---------------- MANUAL ENTRY ----------------
    st.markdown("---")
    st.subheader("✍️ Manual Entry")

    cfg = load_config()
    manual_data = {}

    memories = mem_manager.list_memories()
    mem_mode = st.radio(
        "Save manual entry to:",
        ["Create new memory", "Append to existing memory"],
        horizontal=True
    )

    target_memory = (
        st.text_input("New memory name")
        if mem_mode == "Create new memory"
        else st.selectbox("Select memory", memories)
    )

    with st.form("manual_form"):
        cols = st.columns(4)
        i = 0
        for field, meta in cfg.items():
            with cols[i]:
                if meta["type"] == "text":
                    manual_data[field] = st.text_input(field)
                elif meta["type"] == "select":
                    manual_data[field] = st.selectbox(field, meta.get("options", []))
                elif meta["type"] == "date":
                    if meta.get("mode") == "year":
                        manual_data[field] = str(
                            st.selectbox(field, range(2000, date.today().year + 1))
                        )
                    else:
                        manual_data[field] = st.date_input(field).isoformat()
            i = (i + 1) % 4

        submitted = st.form_submit_button("Add row")

    if submitted:
        st.session_state.setdefault("manual_rows", []).append(manual_data)
        st.success("Row added")

    if st.session_state.get("manual_rows"):
        df_manual = pd.DataFrame(st.session_state["manual_rows"])
        st.dataframe(df_manual, use_container_width=True)

        if st.button("💾 Save Manual Entries"):
            for col in REQUIRED_COLS:
                if col not in df_manual.columns:
                    df_manual[col] = ""

            df_manual["AddedBy"] = st.session_state["user"]["id"]
            df_manual["__semantic_text__"] = build_semantic_text(df_manual)

            df_manual = append_to_memory(mem_manager, target_memory, df_manual)
            meta = mem_manager.create_or_update_memory(target_memory, df_manual)

            if emb_engine:
                emb_engine.index_dataframe(meta["memory_path"], df_manual)

            st.session_state["manual_rows"] = []
            st.success("Manual entries saved and indexed")
            st.rerun()

# =====================================================
# QUERY MODE (UNCHANGED)
# =====================================================
elif mode == "Query Knowledge Base":
    st.header("🔍 Query")

    mem = st.selectbox("Memory", mem_manager.list_memories())
    q = st.text_area("Query")

    if st.button("Search") and emb_engine:
        res = recall_engine.query_memory(mem, q)
        st.dataframe(res, use_container_width=True)

# =====================================================
# SETTINGS (UNCHANGED)
# =====================================================
else:
    st.header("⚙️ Settings")

    if emb_engine and st.button("Rebuild embeddings"):
        for mid, path in mem_manager.list_memories_full().items():
            df = mem_manager.load_memory_dataframe(mid)
            df["__semantic_text__"] = build_semantic_text(df)
            emb_engine.index_dataframe(path, df)
        st.success("Rebuilt")
