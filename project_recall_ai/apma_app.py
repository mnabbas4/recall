# apma_app.py - Deployment Ready Version (Corrected)

import streamlit as st
import os
import pandas as pd
from streamlit import rerun

from modules.file_manager import MemoryManager
from modules.data_handler import DataHandler
from modules.embeddings_engine import EmbeddingsEngine
from modules.recall_engine import RecallEngine
from modules.utils import ensure_data_dirs
from modules import user_manager

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

    mode = st.sidebar.radio("Mode", ["Login", "Create account"])

    if mode == "Create account":
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
    col1, col2 = st.columns([4,1])
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
# UPLOAD MODE
# =====================================================
if mode == "Upload / Update Memory":
    st.header("📤 Upload / Update Memory")

    if not st.session_state.get("user"):
        st.warning("Login required.")
        st.stop()

    uploaded = st.file_uploader("Upload CSV / Excel", ["csv","xlsx"])

    if uploaded:
        df, err = DataHandler.read_and_validate(uploaded, required_cols=REQUIRED_COLS)
        if err:
            st.error(err)
        else:
            st.success(f"Validated {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)

            mem_name = st.text_input("Memory name")
            if st.button("Save"):
                df["AddedBy"] = st.session_state["user"]["id"]
                meta = mem_manager.create_or_update_memory(mem_name, df)
                st.success("Saved")

                if emb_engine:
                    emb_engine.index_dataframe(meta["memory_path"], df, id_prefix=meta["memory_id"])

    # ================= MANUAL ENTRY ==================
    st.markdown("---")
    st.subheader("✍️ Manual Entry")

    with st.form("manual"):
        r1 = st.columns(4)
        r2 = st.columns(4)
        r3 = st.columns(4)
        r4 = st.columns(3)

        data = {
            'COMMESSA': r1[0].text_input("COMMESSA"),
            'CLIENTE': r1[1].text_input("CLIENTE"),
            'ANNO': r1[2].text_input("ANNO"),
            'TIPO MACCHINA': r1[3].text_input("TIPO MACCHINA"),

            'APPLICAZIONE': r2[0].text_input("APPLICAZIONE"),
            'TIPO PROBLEMA': r2[1].text_input("TIPO PROBLEMA"),
            'DESCRIZIONE': r2[2].text_input("DESCRIZIONE"),
            'SOLUZIONE LESSON LEARNED': r2[3].text_input("SOLUZIONE"),

            'DATA INSERIMENTO': r3[0].text_input("DATA INSERIMENTO"),
            'RCPRD': r3[1].text_input("RCPRD"),
            'REPORT CANTIERE': r3[2].text_input("REPORT CANTIERE"),
            'CONCERNED DEPARTMENTS': r3[3].text_input("DEPARTMENTS"),

            'REPORT RIUNIONE CHIUSURA PROGETTO': r4[0].text_input("REPORT RIUNIONE")
        }

        submit = st.form_submit_button("Add")

        if submit:
            st.session_state.setdefault("manual_rows", []).append(data)
            st.success("Added to buffer")

    if st.session_state.get("manual_rows"):
        buf = pd.DataFrame(st.session_state["manual_rows"])[REQUIRED_COLS]
        st.dataframe(buf, use_container_width=True)

# =====================================================
# QUERY MODE
# =====================================================
elif mode == "Query Knowledge Base":
    st.header("🔍 Query")

    mems = mem_manager.list_memories()
    if not mems:
        st.warning("No memories")
    else:
        mem = st.selectbox("Memory", mems)
        q = st.text_area("Query")

        if st.button("Search") and emb_engine:
            res = recall_engine.query_memory(mem, q)
            st.dataframe(res, use_container_width=True)

            insights = recall_engine.generate_structured_insights(res)
            answer = recall_engine.generate_natural_language_answer(
                insights=insights,
                query=q
            )

            st.markdown("### 🧠 Answer")
            st.markdown(answer)


# =====================================================
# SETTINGS
# =====================================================
else:
    st.header("⚙️ Settings")

    st.write("Saved memories:")
    for m in mem_manager.list_memories():
        st.write("-", m)

    if emb_engine and st.button("Rebuild embeddings"):
        for mid, path in mem_manager.list_memories_full().items():
            df = mem_manager.load_memory_dataframe(mid)
            emb_engine.index_dataframe(path, df, id_prefix=mid)
        st.success("Done")

    st.markdown("""
    **APMA**
    - Stores DESCRIZIONE & SOLUZIONE
    - Searches by APPLICAZIONE & TIPO MACCHINA
    - Prevents repeat project issues
    """)
