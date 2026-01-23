# apma_app.py - Deployment Ready Version (Corrected)
print ("hello")
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
    from modules.manual_config import load_config
    from datetime import date
    
    st.subheader("✍️ Manual Entry")
    
    config = load_config()
    manual_data = {}
    
    with st.form("manual_dynamic"):
        cols = st.columns(4)
        col_idx = 0
    
        for field, meta in config.items():
            ftype = meta["type"]
            key = f"manual_{field}"
    
            with cols[col_idx]:
                # TYPE 1 — TEXT
                if ftype == "text":
                    if meta.get("multiline"):
                        manual_data[field] = st.text_area(field, key=key)
                    else:
                        manual_data[field] = st.text_input(field, key=key)
    
                # TYPE 2 — DROPDOWN
                elif ftype == "select":
                    manual_data[field] = st.selectbox(
                        field,
                        options=meta.get("options", []),
                        key=key
                    )
    
                # TYPE 3 — DATE / YEAR
                elif ftype == "date":
                    if meta.get("mode") == "year":
                        year = st.selectbox(
                            field,
                            options=list(range(2000, date.today().year + 1)),
                            key=key
                        )
                        manual_data[field] = str(year)
                    else:
                        d = st.date_input(field, key=key)
                        manual_data[field] = d.isoformat()
    
            col_idx = (col_idx + 1) % 4
    
        submitted = st.form_submit_button("Add")
    
    if submitted:
        st.session_state.setdefault("manual_rows", []).append(manual_data)
        st.success("Row added")

# =====================================================
# QUERY MODE
# =====================================================
elif mode == "Query Knowledge Base":
    st.header("🔍 Query")

    mems = mem_manager.list_memories()
    if not mems:
        st.warning("No memories")
        st.stop()

    mem = st.selectbox("Memory", mems)

    # 🔁 QUERY TYPE
    query_mode = st.radio(
        "Query type",
        ["AI Semantic Search", "Structured Filter Search"],
        horizontal=True
    )

    # =====================================================
    # STRUCTURED FILTER SEARCH
    # =====================================================
    if query_mode == "Structured Filter Search":

        FILTERABLE_COLUMNS = {
            "COMMESSA": "COMMESSA",
            "CLIENTE": "CLIENTE",
            "ANNO": "ANNO",
            "TIPO MACCHINA": "TIPO MACCHINA",
            "APPLICAZIONE": "APPLICAZIONE",
            "TIPO PROBLEMA": "TIPO PROBLEMA"
        }

        col1, col2 = st.columns(2)

        selected_col_label = col1.selectbox(
            "Filter by",
            list(FILTERABLE_COLUMNS.keys())
        )

        filter_value = col2.text_input("Value")

        exact_match = st.checkbox("Exact match", value=False)

        if st.button("Filter"):
            df = recall_engine.filter_memory(
                mem_id=mem,
                column=FILTERABLE_COLUMNS[selected_col_label],
                value=filter_value,
                exact=exact_match
            )

            st.dataframe(df, use_container_width=True)
            st.info(f"{len(df)} records found.")

    # =====================================================
    # AI SEMANTIC SEARCH (EXISTING)
    # =====================================================
    else:
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

    st.markdown("---")

    # ================= MANUAL ENTRY CONFIG =================
    from modules.manual_config import load_config, save_config

    st.subheader("🛠 Manual Entry Configuration")

    cfg = load_config()

    field = st.selectbox("Select field", list(cfg.keys()) + ["➕ Add new"])

    # ADD NEW FIELD
    if field == "➕ Add new":
        new_name = st.text_input("Column name")
        new_type = st.selectbox("Type", ["text", "select", "date"])

        if st.button("Create") and new_name:
            cfg[new_name] = {"type": new_type}
            save_config(cfg)
            st.success("Field added")
            st.rerun()

    # EDIT EXISTING FIELD
    else:
        meta = cfg[field]

        meta["type"] = st.selectbox(
            "Field type",
            ["text", "select", "date"],
            index=["text", "select", "date"].index(meta["type"])
        )

        if meta["type"] == "select":
            options = st.text_area(
                "Dropdown options (one per line)",
                value="\n".join(meta.get("options", []))
            )
            meta["options"] = [o.strip() for o in options.splitlines() if o.strip()]

        if meta["type"] == "date":
            meta["mode"] = st.radio("Date mode", ["full", "year"])

        if st.button("Save changes"):
            cfg[field] = meta
            save_config(cfg)
            st.success("Updated")
            st.rerun()

    st.markdown("""
    **APMA**
    - Stores DESCRIZIONE & SOLUZIONE
    - Searches by APPLICAZIONE & TIPO MACCHINA
    - Prevents repeat project issues
    """)

