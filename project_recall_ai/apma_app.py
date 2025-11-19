# apma_app.py - Deployment Ready Version
import streamlit as st
from modules.file_manager import MemoryManager
from modules.data_handler import DataHandler
from modules.embeddings_engine import EmbeddingsEngine
from modules.recall_engine import RecallEngine
from modules.utils import ensure_data_dirs
from modules import user_manager
import pandas as pd
from streamlit import rerun
import os

# ============================================
# CONFIGURATION - Uses Streamlit Secrets
# ============================================
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
except:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Required columns used across app
REQUIRED_COLS = ['Project Category', 'Project Reference', 'Phase', 'Problems Encountered', 'Solutions Adopted']

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="APMA — AI Project Memory Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# INITIALIZATION
# ============================================
ensure_data_dirs()

# Show deployment warning if on Streamlit Cloud
if 'STREAMLIT_SHARING_MODE' in os.environ or 'STREAMLIT_SERVER_PORT' in os.environ:
    st.sidebar.info("🌐 **Running on Streamlit Cloud**\n\nNote: Data resets on app restart. For production, consider external storage.")

st.title("🧠 AI Project Memory Assistant (APMA)")

# ============================================
# AUTHENTICATION UI
# ============================================
if 'user' not in st.session_state:
    st.session_state['user'] = None

if st.session_state['user'] is None:
    st.sidebar.header("🔐 Account / Login")

    auth_mode = st.sidebar.radio("Choose", ["Login", "Create account"])
    if auth_mode == "Create account":
        st.sidebar.subheader("Create account")
        c_first = st.sidebar.text_input("First name")
        c_last = st.sidebar.text_input("Last name")
        c_id = st.sidebar.text_input("ID Number")
        c_pw = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Create account"):
            ok, msg = user_manager.create_user(c_first, c_last, c_id, c_pw)
            if ok:
                st.sidebar.success(msg + " Please log in.")
            else:
                st.sidebar.error(msg)

        st.sidebar.info("ℹ️ One account per ID. Password minimum 6 characters.")

    else:  # Login
        st.sidebar.subheader("Login")
        l_id = st.sidebar.text_input("ID Number", key="login_id")
        l_pw = st.sidebar.text_input("Password", type="password", key="login_pw")
        
        if st.sidebar.button("Login"):
            ok, msg, profile = user_manager.authenticate(l_id, l_pw)
            if ok:
                st.session_state['user'] = profile
                rerun()
            else:
                st.sidebar.error(msg)

    st.sidebar.markdown("---")
    st.sidebar.write("💡 You can view/query memory without logging in. Login required for upload/create/append operations.")

# ============================================
# MAIN APP VARIABLES
# ============================================
mem_manager = MemoryManager(data_dir='data')

# Embeddings engine
emb_engine = None
emb_error_msg = None
if not OPENAI_API_KEY:
    emb_error_msg = "⚠️ OpenAI API key not configured. Embeddings disabled. Add OPENAI_API_KEY in Streamlit secrets."
else:
    try:
        emb_engine = EmbeddingsEngine()
    except Exception as e:
        emb_engine = None
        emb_error_msg = f"⚠️ Embeddings engine initialization failed: {str(e)}"

recall_engine = RecallEngine(emb_engine=emb_engine, mem_manager=mem_manager)

# Initialize session state keys
if 'last_results' not in st.session_state:
    st.session_state['last_results'] = None
if 'last_query' not in st.session_state:
    st.session_state['last_query'] = None
if 'last_mem_id' not in st.session_state:
    st.session_state['last_mem_id'] = None
if 'last_insights' not in st.session_state:
    st.session_state['last_insights'] = None
if 'last_narrative' not in st.session_state:
    st.session_state['last_narrative'] = None
if 'manual_rows' not in st.session_state:
    st.session_state['manual_rows'] = []

# ============================================
# USER INFO BAR
# ============================================
if st.session_state.get('user'):
    u = st.session_state['user']
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Logged in as:** {u.get('first_name','')} {u.get('last_name','')} — **ID:** `{u.get('id')}`")
    with col2:
        if st.button("Logout"):
            st.session_state['user'] = None
            rerun()

# Show embeddings status
if emb_error_msg:
    st.warning(emb_error_msg)

# ============================================
# MODE SELECTION
# ============================================
mode = st.sidebar.selectbox("📂 Mode", ["Upload / Update Memory File", "Query Knowledge Base", "Settings"])

# ============================================
# MODE: UPLOAD / UPDATE
# ============================================
if mode == "Upload / Update Memory File":
    st.header("📤 Upload / Update Memory File")

    # Check if user is logged in
    if not st.session_state.get('user'):
        st.info("ℹ️ Please login using the sidebar to enable upload and manual entry features.")
        st.stop()

    # ---------- File upload ----------
    uploaded = st.file_uploader("Upload CSV or Excel", type=['csv','xlsx'])
    if uploaded is not None:
        df, err = DataHandler.read_and_validate(uploaded)
        if err:
            st.error(err)
        else:
            st.success(f"✅ Validated: {len(df)} rows")
            st.dataframe(df.head(10), use_container_width=True)

            existing = mem_manager.list_memories()
            choice = st.radio("Save options:", ["Create new memory file", "Append to existing memory", "Edit existing memory"])
            sel = None
            if choice != "Create new memory file":
                if existing:
                    sel = st.selectbox("Select memory to modify", ["-- choose --"] + existing)
                else:
                    st.info("No existing memories found. Will create new.")
            mem_name = st.text_input("Memory file name", value=f"memory_{len(existing)+1}")

            if st.button("💾 Save to memory"):
                user_id = st.session_state['user']['id']
                df['AddedBy'] = user_id

                if choice == "Append to existing memory" and sel and sel != "-- choose --":
                    try:
                        existing_df = mem_manager.load_memory_dataframe(sel)
                        new_df = pd.concat([existing_df, df], ignore_index=True)
                        meta = mem_manager.create_or_update_memory(sel, new_df, mode='Append to existing memory', target_memory=sel)
                        st.success(f"✅ Appended to memory: {sel}")
                        
                        if emb_engine is not None:
                            with st.spinner("Computing embeddings for appended rows..."):
                                try:
                                    emb_engine.index_dataframe(meta['memory_path'], df, id_prefix=sel)
                                    st.success("✅ Embeddings computed and saved.")
                                except Exception as e:
                                    st.error(f"❌ Embeddings indexing failed: {e}")
                        else:
                            st.warning("⚠️ Embeddings engine not active. Configure OPENAI_API_KEY to compute embeddings.")
                    except Exception as e:
                        st.error(f"❌ Failed to append: {e}")
                else:
                    meta = mem_manager.create_or_update_memory(mem_name, df, mode=choice, target_memory=sel)
                    st.success(f"✅ Saved to memory: {meta['memory_id']}")
                    
                    if emb_engine is not None:
                        with st.spinner("Computing embeddings..."):
                            try:
                                emb_engine.index_dataframe(meta['memory_path'], df, id_prefix=meta['memory_id'])
                                st.success("✅ Embeddings computed and saved.")
                            except Exception as e:
                                st.error(f"❌ Embeddings indexing failed: {e}")
                    else:
                        st.warning("⚠️ Embeddings engine not active. Configure OPENAI_API_KEY.")

    st.markdown("---")

    # ---------- Manual entry ----------
    st.subheader("✍️ Manual entry (add rows one-by-one)")
    st.write("Add rows manually that follow the required columns.")

    with st.form("manual_row_form", clear_on_submit=True):
        cols = st.columns([2, 2, 2, 3, 3])
        project_category = cols[0].text_input("Project Category")
        project_reference = cols[1].text_input("Project Reference")
        phase = cols[2].text_input("Phase")
        problems = cols[3].text_input("Problems Encountered")
        solutions = cols[4].text_input("Solutions Adopted")

        submitted = st.form_submit_button("➕ Add row to buffer")
        if submitted:
            if not (project_category.strip() or project_reference.strip() or phase.strip()):
                st.error("Please provide at least one of Project Category / Project Reference / Phase.")
            else:
                row = {
                    'Project Category': project_category.strip(),
                    'Project Reference': project_reference.strip(),
                    'Phase': phase.strip(),
                    'Problems Encountered': problems.strip(),
                    'Solutions Adopted': solutions.strip()
                }
                st.session_state['manual_rows'].append(row)
                st.success("✅ Row added to buffer.")

    if st.session_state['manual_rows']:
        st.write(f"📋 **Buffer ({len(st.session_state['manual_rows'])} rows):**")
        tmp_df = pd.DataFrame(st.session_state['manual_rows'])[REQUIRED_COLS].fillna('')
        st.dataframe(tmp_df.reset_index(drop=True), use_container_width=True)

        colA, colB, colC = st.columns([2,2,1])
        with colA:
            save_choice = st.radio("Save manual rows as:", ["Create new memory", "Append to existing memory"], index=0)
        with colB:
            existing = mem_manager.list_memories()
            append_to = None
            if save_choice == "Append to existing memory":
                if existing:
                    append_to = st.selectbox("Select memory to append to", ["-- choose --"] + existing)
                else:
                    st.info("No existing memories. Choose 'Create new memory'.")
        with colC:
            mem_name_manual = st.text_input("Memory file name", value=f"manual_{len(existing)+1}")

        if st.button("💾 Save manual rows to memory"):
            user_id = st.session_state['user']['id']
            save_df = pd.DataFrame(st.session_state['manual_rows'])[REQUIRED_COLS].fillna('')
            save_df['AddedBy'] = user_id

            if save_choice == "Create new memory":
                meta = mem_manager.create_or_update_memory(mem_name_manual, save_df, mode='Create new memory file')
                st.success(f"✅ Saved manual rows to: {meta['memory_id']}")
                
                if emb_engine is not None:
                    with st.spinner("Computing embeddings..."):
                        try:
                            emb_engine.index_dataframe(meta['memory_path'], save_df, id_prefix=meta['memory_id'])
                            st.success("✅ Embeddings computed.")
                        except Exception as e:
                            st.error(f"❌ Embeddings failed: {e}")
                else:
                    st.warning("⚠️ Embeddings engine not active.")
                
                st.session_state['manual_rows'] = []
                st.rerun()
            else:
                if not append_to or append_to == "-- choose --":
                    st.error("Please select a memory to append to.")
                else:
                    try:
                        existing_df = mem_manager.load_memory_dataframe(append_to)
                        new_df = pd.concat([existing_df, save_df], ignore_index=True)
                        meta = mem_manager.create_or_update_memory(append_to, new_df, mode='Append to existing memory', target_memory=append_to)
                        st.success(f"✅ Appended manual rows to: {append_to}")
                        
                        if emb_engine is not None:
                            with st.spinner("Computing embeddings..."):
                                try:
                                    emb_engine.index_dataframe(meta['memory_path'], save_df, id_prefix=append_to)
                                    st.success("✅ Embeddings computed.")
                                except Exception as e:
                                    st.error(f"❌ Embeddings failed: {e}")
                        else:
                            st.warning("⚠️ Embeddings engine not active.")
                        
                        st.session_state['manual_rows'] = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to append: {e}")
    else:
        st.info("No rows in buffer. Use the form above to add rows.")

# ============================================
# MODE: QUERY KNOWLEDGE BASE
# ============================================
elif mode == "Query Knowledge Base":
    st.header("🔍 Query Knowledge Base")
    memories = mem_manager.list_memories()
    
    if not memories:
        st.warning("No memories found. Upload a file first.")
    else:
        sel_mem = st.selectbox("Select memory to query", memories)
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            top_n = st.slider("Hard cap (0 = no limit):", 0, 200, 0)
            similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.25)
        with col2:
            spell = st.checkbox("Enable spell correction", value=True)
            enforce_context = st.checkbox("Enforce detected phase/category (strict)", value=False)
        
        q = st.text_area("Enter your query (natural language)", height=120)

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔍 Search") and q.strip():
                if emb_engine is None:
                    st.error("❌ Embeddings engine not available. Configure OPENAI_API_KEY.")
                else:
                    try:
                        hard_limit = top_n if top_n > 0 else None
                        res_df = recall_engine.query_memory(
                            mem_id=sel_mem,
                            query=q,
                            min_score=similarity_threshold,
                            spell_correction=spell,
                            hard_limit=hard_limit,
                            enforce_context=enforce_context
                        )
                        st.session_state['last_results'] = res_df
                        st.session_state['last_query'] = q
                        st.session_state['last_mem_id'] = sel_mem
                        st.session_state['last_insights'] = None
                        st.session_state['last_narrative'] = None
                    except Exception as e:
                        st.error(f"❌ Query failed: {e}")

            if st.button("🔄 Reset Query"):
                st.session_state['last_results'] = None
                st.session_state['last_query'] = None
                st.session_state['last_insights'] = None
                st.session_state['last_narrative'] = None
                st.rerun()

        with col2:
            st.markdown("**💡 Tips:** Include phase names (e.g., 'Order Confirmation') or categories to improve precision.")

        st.markdown("---")

        # Display results
        if st.session_state['last_results'] is None:
            st.info("No previous results. Run a search to see matches.")
        else:
            results_df = st.session_state['last_results']
            used_fallback = results_df.attrs.get('used_fallback', False)
            matched_phase = results_df.attrs.get('matched_phase', None)
            matched_category = results_df.attrs.get('matched_category', None)

            # Context info
            ctx_lines = []
            if matched_phase:
                ctx_lines.append(f"Detected phase: **{matched_phase}**")
            if matched_category:
                ctx_lines.append(f"Detected category: **{matched_category}**")
            if used_fallback:
                ctx_lines.append("⚠️ No matches passed threshold — showing nearest neighbors.")
            if ctx_lines:
                st.info(" — ".join(ctx_lines))

            # Results table
            display_cols = ['FinalScore','TextScore','PhaseBonus','CategoryBonus','Project Category','Project Reference','Phase','Problems Encountered','Solutions Adopted','AddedBy']
            available_cols = [c for c in display_cols if c in results_df.columns]
            
            st.write(f"### 📊 Results ({len(results_df)} matches)")
            st.dataframe(results_df[available_cols].reset_index(drop=True), use_container_width=True)

            # Structured insights
            st.markdown("### 📈 Structured Insights (data-driven)")
            insights = recall_engine.generate_structured_insights(results_df)
            st.session_state['last_insights'] = insights
            
            if insights['top_problems']:
                top_table = pd.DataFrame([{
                    'Problem': p['problem'],
                    'Count': p['count'],
                    'AvgScore': p['avg_score'],
                    'ExampleSolutions': " | ".join(p['solutions'][:3])
                } for p in insights['top_problems']])
                st.write("**Top problems (aggregated):**")
                st.dataframe(top_table, use_container_width=True)
            else:
                st.write("No aggregated problems to show.")

            if insights['per_phase_summary']:
                st.write("**Per-phase summary:**")
                st.json(insights['per_phase_summary'])

            # LLM narrative
            if st.button("🤖 Generate human-readable insights (LLM)"):
                if not OPENAI_API_KEY:
                    st.error("❌ OpenAI key not configured. Add OPENAI_API_KEY in Streamlit secrets.")
                else:
                    with st.spinner("Generating narrative..."):
                        narrative = recall_engine.generate_insights_narrative(insights)
                        st.session_state['last_narrative'] = narrative
            
            if st.session_state['last_narrative']:
                st.markdown("**📝 Narrative (based on recorded facts):**")
                st.info(st.session_state['last_narrative'])

# ============================================
# MODE: SETTINGS
# ============================================
else:
    st.header("⚙️ Settings")
    
    st.subheader("📁 Saved Memories")
    memories = mem_manager.list_memories()
    if memories:
        for m in memories:
            st.write(f"- `{m}`")
    else:
        st.info("No memories saved yet.")
    
    st.markdown("---")
    
    st.subheader("🔧 Embeddings Engine Status")
    if emb_engine is None:
        st.warning("⚠️ Embeddings engine not active. Configure OPENAI_API_KEY in secrets.")
    else:
        st.success("✅ Embeddings engine active.")
        
        if st.button("🔄 Rebuild embeddings for all memories"):
            with st.spinner("Rebuilding embeddings..."):
                try:
                    mems = mem_manager.list_memories_full()
                    for mid, path in mems.items():
                        df = mem_manager.load_memory_dataframe(mid)
                        emb_engine.index_dataframe(path, df, id_prefix=mid)
                    st.success("✅ Rebuilt embeddings for all memories.")
                except Exception as e:
                    st.error(f"❌ Failed to rebuild embeddings: {e}")
    
    st.markdown("---")
    
    st.subheader("ℹ️ About APMA")
    st.write("""
    **AI Project Memory Assistant (APMA)** helps teams:
    - Store project problems and solutions
    - Query past experiences using AI
    - Get insights to avoid repeating mistakes
    
    **Version:** 1.0.0 (Deployment Ready)
    """)
