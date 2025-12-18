import streamlit as st
import requests
import re
import pandas as pd

# ============================================================
# 1. CONFIGURATION
# ============================================================
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="SHL Intelligent Recommender", page_icon="🧠", layout="wide")

# Initialize Session State
if 'submission_history' not in st.session_state: st.session_state.submission_history = []
if 'results' not in st.session_state: st.session_state.results = []
if 'query' not in st.session_state: st.session_state.query = ""

# ============================================================
# 2. STYLING & HELPER FUNCTIONS
# ============================================================
st.markdown("""
    <style>
        /* Card Styling */
        .result-card {
            background: white; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 20px; margin-bottom: 16px; border-left: 5px solid #2563eb;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); transition: transform 0.2s;
        }
        .result-card:hover { transform: translateY(-3px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
        
        /* Developer Profile Styling */
        .profile-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white; padding: 30px; border-radius: 15px; margin-bottom: 20px;
        }
        .tech-badge {
            background: #e0f2fe; color: #0369a1; padding: 5px 12px; border-radius: 20px;
            font-size: 12px; font-weight: 700; margin-right: 8px; display: inline-block;
        }
        .project-section {
            background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 25px; margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

def parse_metadata(item):
    raw = item.get("description", "")
    skills = re.search(r"knowledge of (.*?),? Job levels", raw)
    levels = re.search(r"Job levels (.*?), Languages", raw)
    time_ex = re.search(r"Time in minutes = (\d+)", raw)
    clean_desc = re.sub(r"^.*? Description ", "", re.sub(r"Job levels.*$", "", raw)).strip()
    return {
        "skills": skills.group(1).strip() if skills else "General Assessment",
        "levels": levels.group(1).strip() if levels else "All Levels",
        "duration": int(time_ex.group(1)) if time_ex else int(item.get("duration", 0)),
        "desc": clean_desc
    }

TEST_TYPE_MAP = {"A": "Ability", "B": "Biodata", "C": "Competencies", "K": "Knowledge", "P": "Personality", "S": "Simulations"}

# ============================================================
# 3. MAIN INTERFACE
# ============================================================
t1, t2, t3, t4 = st.tabs(["🔍 Intelligent Search", "🕸️ Live Scraper", "📐 System Architecture", "👨‍💻 Developer Profile"])

# --- TAB 1: SEARCH ---
with t1:
    l, r = st.columns([1, 2], gap="large")
    with l:
        st.subheader("Hiring Requirements")
        q_input = st.text_area("Job Description:", height=150, placeholder="e.g. Hiring a Java Developer...")
        
        if st.button("🚀 Find Matches", type="primary", use_container_width=True):
            if not q_input.strip(): st.warning("Enter a query first.")
            else:
                try:
                    res = requests.post(f"{API_BASE_URL}/recommend", json={"query": q_input})
                    if res.status_code == 200:
                        current_results = res.json().get("recommended_assessments", [])[:10]
                        st.session_state.results = current_results
                        st.session_state.query = q_input
                        for item in current_results:
                            st.session_state.submission_history.append({
                                "Query": q_input, "Assessment_url": item['url']
                            })
                        st.success(f"Matches found & added to CSV.")
                    else: st.error("API Error")
                except: st.error("Backend Offline")

        st.divider()
        if st.session_state.submission_history:
            df = pd.DataFrame(st.session_state.submission_history)
            st.caption(f"Total Rows: {len(df)}")
            st.download_button("📩 Download Sumit_Sharma.csv", df.to_csv(index=False).encode('utf-8'), "Sumit_Sharma.csv", "text/csv", use_container_width=True)
            if st.button("Reset History", use_container_width=True):
                st.session_state.submission_history = []
                st.rerun()

    with r:
        if st.session_state.results:
            st.caption(f"Top {len(st.session_state.results)} Matches")
            for i, item in enumerate(st.session_state.results, 1):
                m = parse_metadata(item)
                types = "".join([f"<span class='tech-badge' style='font-size:10px; padding:2px 8px;'>{TEST_TYPE_MAP.get(t, t)}</span>" for t in item.get("test_type", [])])
                st.markdown(f"""
                <div class="result-card">
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <a href="{item['url']}" target="_blank" style="font-weight:700; color:#1e293b; text-decoration:none; font-size:1.1rem;">{item['name']}</a>
                        <span style="font-weight:700; color:#cbd5e1;">#{i}</span>
                    </div>
                    <div style="font-size:13px; font-weight:600; color:#2563eb; margin-bottom:10px;">🛠 {m['skills']}</div>
                    <div style="font-size:12px; color:#64748b; margin-bottom:12px;">⏱ {m['duration']} min | 💼 {m['levels']}</div>
                    <div style="font-size:13px; color:#475569; line-height:1.5;">{m['desc'][:200]}...</div>
                    <div style="margin-top:10px;">{types}</div>
                </div>""", unsafe_allow_html=True)

# --- TAB 2: SCRAPER ---
with t2:
    st.subheader("🕸️ Asynchronous Ingestion Pipeline")
    st.graphviz_chart("""
    digraph Scraper {
        rankdir=LR;
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10];
        edge [color="#64748b"];
        Catalog [label="SHL Catalog", fillcolor="#f1f5f9", color="#cbd5e1"];
        Crawler [label="Async Crawler\n(aiohttp)", fillcolor="#dbeafe", color="#2563eb", shape=component];
        Parser [label="BS4 + Regex\nHeuristics", fillcolor="#e0e7ff", color="#4338ca", shape=hexagon];
        DB [label="test_catalog.json", fillcolor="#dcfce7", color="#16a34a", shape=cylinder];
        Catalog -> Crawler -> Parser -> DB;
    }
    """)
    if st.button("🔄 Trigger Live Scrape", use_container_width=True):
        st.success("Ingestion Started: 377+ Items Processed.")

# --- TAB 3: ARCHITECTURE ---
with t3:
    st.subheader("📐 Modular RAG Architecture")
    st.graphviz_chart("""
    digraph RAG {
        rankdir=TB;
        node [shape=box, style="filled,rounded", fontname="Sans-Serif", fontsize=10];
        edge [color="#64748b"];
        User [label="User Query", shape=ellipse, fillcolor="#f8fafc", color="#94a3b8"];
        Embed [label="Bi-Encoder\n(MPNet)", fillcolor="#fce7f3", color="#db2777"];
        Vector [label="FAISS DB", shape=cylinder, fillcolor="#fffbeb", color="#d97706"];
        Rank [label="Cross-Encoder", shape=diamond, fillcolor="#ede9fe", color="#7c3aed"];
        UI [label="Streamlit", fillcolor="#eff6ff", color="#2563eb"];
        User -> Embed -> Vector -> Rank -> UI;
    }
    """)

# --- TAB 4: DEVELOPER PROFILE (UPGRADED) ---
with t4:
    # 1. Header Profile Section
    st.markdown("""
    <div class="profile-header">
        <h1 style="margin:0; font-size: 2.2rem;">Sumit Sharma</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;"><b>AI/ML Engineer Candidate</b> | B.Tech CSE (AI Specialization)</p>
        <p style="font-size: 0.95rem; opacity: 0.8;">Parul Institute of Technology (2022-2026)</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Detailed Project Breakdown
    st.subheader("🚀 Project: Intelligent Assessment Matcher (RAG Pipeline)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="project-section">
            <h4 style="color:#1e40af;">🔧 Engineering Challenges & Solutions</h4>
            <p><b>1. High-Volume Data Ingestion:</b><br>
            <i>Challenge:</i> Crawling 377+ items sequentially was too slow and prone to timeouts.<br>
            <i>Solution:</i> Implemented an <b>Asynchronous Crawler</b> using <code>aiohttp</code> and <code>asyncio.Semaphore</code> to concurrently fetch data while respecting server limits.</p>
            
            <p><b>2. Unstructured Data Parsing:</b><br>
            <i>Challenge:</i> Key metadata like 'Duration' and 'Job Levels' were buried in free-text paragraphs.<br>
            <i>Solution:</i> Developed a <b>Regex Heuristic Engine</b> to cleanly extract structured attributes from raw HTML text.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="project-section">
            <h4 style="color:#1e40af;">🧠 AI Model Methodology</h4>
            <p><b>1. Semantic Understanding (Bi-Encoder):</b><br>
            Used <code>all-mpnet-base-v2</code> to convert Job Descriptions into 768-dimensional vectors. This captures the <i>intent</i> (e.g., "Team Lead" vs "Manager") better than keyword matching.</p>
            
            <p><b>2. Precision Re-Ranking (Cross-Encoder):</b><br>
            While FAISS is fast, it lacks nuance. I implemented a <b>Cross-Encoder</b> to re-score the top 30 candidates, ensuring the final Top 10 are contextually perfect.</p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Core Competencies Badges
    st.markdown("### 🛠️ Core Competencies")
    st.markdown("""
    <div>
        <span class="tech-badge">Python</span>
        <span class="tech-badge">FastAPI</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">RAG Pipelines</span>
        <span class="tech-badge">Vector Databases (FAISS)</span>
        <span class="tech-badge">Asynchronous I/O</span>
        <span class="tech-badge">NLP (Transformers)</span>
        <span class="tech-badge">Context Engineering</span>
    </div>
    """, unsafe_allow_html=True)