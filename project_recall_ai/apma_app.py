# apma_app.py - Deployment Ready Version (Manual Entry + Memory Fixed)

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

    if not st.session_state.get("user"):
        st.warning("Login required.")
        st.stop()

    uploaded = st.file_uploader("Upload CSV / Excel", ["csv", "xlsx"])

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

                if emb_engine:
                    emb_engine.index_dataframe(
                        meta["memory_path"], df, id_prefix=meta["memory_id"]
                    )

                st.success("Saved")

    # ---------------- MANUAL ENTRY ----------------
    st.markdown("---")
    st.subheader("✍️ Manual Entry")

    config = load_config()
    manual_data = {}

    memories = mem_manager.list_memories()

    mem_mode = st.radio(
        "Save manual entry to:",
        ["Create new memory", "Append to existing memory"],
        horizontal=True
    )

    target_memory = None
    if mem_mode == "Create new memory":
        target_memory = st.text_input("New memory name")
    else:
        if memories:
            target_memory = st.selectbox("Select memory", memories)
        else:
            st.warning("No existing memories available")

    with st.form("manual_dynamic"):
        cols = st.columns(4)
        col_idx = 0

        for field, meta in config.items():
            key = f"manual_{field}"

            with cols[col_idx]:
                if meta["type"] == "text":
                    manual_data[field] = (
                        st.text_area(field, key=key)
                        if meta.get("multiline")
                        else st.text_input(field, key=key)
                    )

                elif meta["type"] == "select":
                    manual_data[field] = st.selectbox(
                        field, meta.get("options", []), key=key
                    )

                elif meta["type"] == "date":
                    if meta.get("mode") == "year":
                        y = st.selectbox(
                            field,
                            list(range(2000, date.today().year + 1)),
                            key=key
                        )
                        manual_data[field] = str(y)
                    else:
                        d = st.date_input(field, key=key)
                        manual_data[field] = d.isoformat()

            col_idx = (col_idx + 1) % 4

        submitted = st.form_submit_button("Add row")

    if submitted:
        st.session_state.setdefault("manual_rows", []).append(manual_data)
        st.success("Row added")

    if st.session_state.get("manual_rows"):
        st.markdown("### 📄 Pending Manual Entries")
        st.dataframe(pd.DataFrame(st.session_state["manual_rows"]), use_container_width=True)

        if st.button("💾 Save Manual Entries"):
            if not target_memory:
                st.error("Please select or create a memory.")
            else:
                df_manual = pd.DataFrame(st.session_state["manual_rows"])

                for col in REQUIRED_COLS:
                    if col not in df_manual.columns:
                        df_manual[col] = ""

                df_manual = df_manual[REQUIRED_COLS]
                df_manual["AddedBy"] = st.session_state["user"]["id"]

                meta = mem_manager.create_or_update_memory(target_memory, df_manual)

                if emb_engine:
                    emb_engine.index_dataframe(
                        meta["memory_path"],
                        df_manual,
                        id_prefix=meta["memory_id"]
                    )

                st.session_state["manual_rows"] = []
                st.success(f"Saved to memory '{target_memory}'")
                st.rerun()

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

    query_mode = st.radio(
        "Query type",
        ["AI Semantic Search", "Structured Filter Search"],
        horizontal=True
    )

    if query_mode == "Structured Filter Search":
        FILTERABLE_COLUMNS = {
            "COMMESSA": "COMMESSA",
            "CLIENTE": "CLIENTE",
            "ANNO": "ANNO",
            "TIPO MACCHINA": "TIPO MACCHINA",
            "APPLICAZIONE": "APPLICAZIONE",
            "TIPO PROBLEMA": "TIPO PROBLEMA"
        }

        c1, c2 = st.columns(2)
        col = c1.selectbox("Filter by", FILTERABLE_COLUMNS.keys())
        val = c2.text_input("Value")
        exact = st.checkbox("Exact match", False)

        if st.button("Filter"):
            df = recall_engine.filter_memory(mem, FILTERABLE_COLUMNS[col], val, exact)
            st.dataframe(df, use_container_width=True)
            st.info(f"{len(df)} records found")

    else:
        q = st.text_area("Query")
        if st.button("Search") and emb_engine:
            res = recall_engine.query_memory(mem, q)
            st.dataframe(res, use_container_width=True)

            insights = recall_engine.generate_structured_insights(res)
            answer = recall_engine.generate_natural_language_answer(insights, q)

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
    st.subheader("🛠 Manual Entry Configuration")

    cfg = load_config()
    field = st.selectbox("Select field", list(cfg.keys()) + ["➕ Add new"])

    if field == "➕ Add new":
        new_name = st.text_input("Column name")
        new_type = st.selectbox("Type", ["text", "select", "date"])

        if st.button("Create") and new_name:
            cfg[new_name] = {"type": new_type}
            save_config(cfg)
            st.success("Field added")
            st.rerun()

    else:
        meta = cfg[field]

        # ---------- RENAME ----------
        new_field_name = st.text_input("Rename field", value=field)

        meta["type"] = st.selectbox(
            "Field type",
            ["text", "select", "date"],
            index=["text", "select", "date"].index(meta["type"])
        )

        if meta["type"] == "select":
            opts = st.text_area(
                "Dropdown options (one per line)",
                "\n".join(meta.get("options", []))
            )
            meta["options"] = [o.strip() for o in opts.splitlines() if o.strip()]

        if meta["type"] == "date":
            meta["mode"] = st.radio("Date mode", ["full", "year"])

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("💾 Save changes"):
                if new_field_name != field:
                    cfg[new_field_name] = meta
                    del cfg[field]
                else:
                    cfg[field] = meta

                save_config(cfg)
                st.success("Updated")
                st.rerun()

        with col_b:
            if st.button("🗑 Delete field"):
                del cfg[field]
                save_config(cfg)
                st.warning(f"Field '{field}' deleted")
                st.rerun()

    st.markdown("""
    **APMA**
    - Stores DESCRIZIONE & SOLUZIONE
    - Searches by APPLICAZIONE & TIPO MACCHINA
    - Prevents repeat project issues
    """)
