import streamlit as st
import requests
import streamlit.components.v1 as components
import html
import subprocess
import os
import sys


API_BASE_URL = "http://127.0.0.1:8000"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SCRAPER_SCRIPT = os.path.join(PROJECT_ROOT, "src", "ingestion.py")

st.set_page_config(page_title="SHL Search", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
.block-container { padding-top: 2rem; padding-bottom: 5rem; }
body { font-family: 'Inter', sans-serif; background-color: #f8f9fa; }
.stButton button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


def render_architecture():
    st.graphviz_chart("""
    digraph RAG {
        rankdir=LR;
        User -> UI;
        UI -> API;
        API -> Embed;
        Embed -> DB;
        DB -> Rank;
        Rank -> API;
        API -> UI;
    }
    """)


def render_specs():
    components.html("""
    <div>
        <div>Embedding: all-mpnet-base-v2</div>
        <div>Reranker: MiniLM</div>
        <div>Vector DB: FAISS</div>
        <div>Backend: FastAPI</div>
    </div>
    """, height=200)


def render_result_card(item, rank):
    name = html.escape(item.get("name", ""))
    url = item.get("url", "#")
    desc = html.escape(item.get("description", ""))
    duration = item.get("duration", 0)
    level = html.escape(item.get("job_levels", "All"))
    tags = " ".join(item.get("test_type", []))

    components.html(f"""
    <div style="border:1px solid #ddd;padding:12px;border-radius:8px;margin-bottom:10px;">
        <a href="{url}" target="_blank">{name}</a>
        <div>{duration} mins | {level}</div>
        <div>{tags}</div>
        <div>{desc}</div>
    </div>
    """, height=160)


def is_api_alive():
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=1).ok
    except:
        return False


def run_scraper_process():
    if not os.path.exists(SCRAPER_SCRIPT):
        return None
    env = os.environ.copy()
    return subprocess.Popen(
        [sys.executable, SCRAPER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT_ROOT,
        env=env,
        text=True
    )


with st.sidebar:
    if is_api_alive():
        st.success("Online")
    else:
        st.error("Offline")


tab1, tab2, tab3 = st.tabs(["Overview", "Search", "Admin"])


with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        render_architecture()
    with c2:
        render_specs()


with tab2:
    if "query" not in st.session_state:
        st.session_state.query = ""

    if st.button("Data Analyst"):
        st.session_state.query = "Senior Data Analyst with SQL and Python"
    if st.button("Python Developer"):
        st.session_state.query = "Senior Python Developer"
    if st.button("Project Manager"):
        st.session_state.query = "Project Manager with Scrum"

    query = st.text_area("Query", value=st.session_state.query, height=100)

    if st.button("Search"):
        if query and is_api_alive():
            res = requests.post(f"{API_BASE_URL}/recommend", json={"query": query})
            if res.status_code == 200:
                for i, item in enumerate(res.json()["recommended_assessments"]):
                    render_result_card(item, i)


with tab3:
    if st.button("Run Scraper"):
        proc = run_scraper_process()
        if proc:
            out, err = proc.communicate()
            if proc.returncode == 0:
                st.success("Updated")
                st.code(out)
            else:
                st.error(err)
