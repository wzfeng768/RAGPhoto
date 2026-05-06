"""
Modern Optics & Photonics RAG + Knowledge Graph Interface
Professional, clean, and technology-focused UI for optoelectronics domain
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time
import json
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# CRITICAL PATH SETUP - Project has TWO 'core' modules:
#   1. agentic_rag/core/     - RAG system (AgentOrchestrator) ← WE NEED THIS
#   2. data_pipeline/core/   - Database builder (RAGSystem, VectorManager)
# 
# Solution: Add BOTH paths but in correct order
# ============================================================================

agentic_rag_root = Path(__file__).resolve().parent.parent  # .../agentic_rag/
project_root = agentic_rag_root.parent                 # .../Data_Agentic_RAG/

# CRITICAL: Add agentic_rag FIRST (highest priority) to ensure 'core' resolves correctly
if str(agentic_rag_root) not in sys.path:
    sys.path.insert(0, str(agentic_rag_root))

# Then add project root for 'examples' and 'agentic_rag' package access
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))  # Also insert at 0 to ensure package access works

# Import using agentic_rag prefix to be absolutely explicit
from agentic_rag.config.config import config, load_env_file
from agentic_rag.core import AgentOrchestrator
from examples.example_queries import ALL_QUERIES
from agentic_rag.ui.kg_visualizer import (query_kg_data, create_kg_visualization, 
                                           get_kg_statistics, get_full_kg_overview, 
                                           create_overview_visualization)
from agentic_rag.ui.utils import format_latex_for_display

# Load environment
load_env_file()

# Page configuration
st.set_page_config(
    page_title="OptiRAG - Optics & Photonics Knowledge System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern optics theme + MathJax support
st.markdown("""
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<style>
    /* Main theme colors */
    :root {
        --primary-blue: #1e3a8a;
        --secondary-cyan: #06b6d4;
        --accent-purple: #7c3aed;
        --bg-light: #f8fafc;
        --text-dark: #1e293b;
        --card-bg: #ffffff;
    }
    
    /* Global styles */
    .main {
        background-color: var(--bg-light);
    }
    
    /* Header */
    .opti-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(30, 58, 138, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .opti-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 50%, rgba(255,255,255,0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .opti-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        text-align: center;
    }
    
    .opti-subtitle {
        color: #dbeafe;
        font-size: 1.1rem;
        margin-top: 0.8rem;
        font-weight: 500;
        position: relative;
        z-index: 1;
        text-align: center;
    }
    
    /* RAG Flow Chart */
    .rag-flow {
        background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
        padding: 2rem 1rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e0f2fe;
    }
    
    .flow-step-card {
        flex: 1;
        background: white;
        padding: 1.2rem 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        min-width: 140px;
        border: 2px solid transparent;
    }
    
    .flow-step-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.2);
        border-color: #3b82f6;
    }
    
    .flow-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .flow-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.3rem;
    }
    
    .flow-desc {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .flow-arrow {
        color: #3b82f6;
        font-size: 2rem;
        font-weight: bold;
        margin: 0 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
    }
    
    /* Cards */
    .opti-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid var(--secondary-cyan);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .opti-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .card-title {
        color: var(--primary-blue);
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    
    .card-content {
        color: var(--text-dark);
        line-height: 1.6;
    }
    
    /* Document cards */
    .doc-card {
        background: linear-gradient(to right, #f0f9ff, #ffffff);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
        border-left: 3px solid #3b82f6;
        transition: all 0.2s ease;
    }
    
    .doc-card:hover {
        background: linear-gradient(to right, #e0f2fe, #f8fafc);
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.15);
        transform: translateX(-2px);
    }
    
    .doc-score {
        background: #10b981;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-blue);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.3rem;
    }
    
    /* Query examples */
    .example-query {
        background: #eff6ff;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s;
        border: 2px solid transparent;
    }
    
    .example-query:hover {
        background: #dbeafe;
        border-color: #3b82f6;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-processing {
        background: #fef3c7;
        color: #92400e;
    }
    
    /* Knowledge Graph */
    .kg-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Scrollable container for retrieved documents */
    .scrollable-docs {
        max-height: 65vh;
        overflow-y: auto;
        overflow-x: hidden;
        padding-right: 0.5rem;
        scroll-behavior: smooth;
    }
    
    /* Custom scrollbar for documents */
    .scrollable-docs::-webkit-scrollbar {
        width: 10px;
    }
    
    .scrollable-docs::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 5px;
        margin: 4px 0;
    }
    
    .scrollable-docs::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #94a3b8, #64748b);
        border-radius: 5px;
        border: 2px solid #f1f5f9;
    }
    
    .scrollable-docs::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #64748b, #475569);
    }
    
    .scrollable-docs::-webkit-scrollbar-thumb:active {
        background: #475569;
    }
    
    /* Firefox scrollbar */
    .scrollable-docs {
        scrollbar-width: thin;
        scrollbar-color: #94a3b8 #f1f5f9;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #06b6d4);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the modern header"""
    st.markdown("""
    <div class="opti-header">
        <h1 class="opti-title">🔬 OptiRAG - Intelligent Optics & Photonics Knowledge System</h1>
        <p class="opti-subtitle">
            ⚡ Advanced RAG + Knowledge Graph | 🤖 Powered by Agentic AI | 🔆 Specialized in Optoelectronics Research
        </p>
        <div style="margin-top: 1rem; padding: 0.8rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #fff;">67.8K+</div>
                    <div style="font-size: 0.85rem; color: #bfdbfe;">Vectors</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #fff;">76.5K+</div>
                    <div style="font-size: 0.85rem; color: #bfdbfe;">Entities</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #fff;">363K+</div>
                    <div style="font-size: 0.85rem; color: #bfdbfe;">Relations</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #fff;">30K+</div>
                    <div style="font-size: 0.85rem; color: #bfdbfe;">Papers</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_rag_flow():
    """Render RAG process flow diagram"""
    st.markdown("""
    <div class="rag-flow" style="text-align: center; padding: 2rem 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1400px; margin: 0 auto; gap: 0.5rem;">
            <div class="flow-step-card">
                <div class="flow-icon">📝</div>
                <div class="flow-title">Query</div>
                <div class="flow-desc">Input your question</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step-card">
                <div class="flow-icon">🔍</div>
                <div class="flow-title">Retrieval</div>
                <div class="flow-desc">Search 67K vectors</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step-card">
                <div class="flow-icon">⚡</div>
                <div class="flow-title">Rerank</div>
                <div class="flow-desc">Qwen3-Reranker</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step-card">
                <div class="flow-icon">🤖</div>
                <div class="flow-title">Generation</div>
                <div class="flow-desc">GPT-4o-mini</div>
            </div>
            <div class="flow-arrow">→</div>
            <div class="flow-step-card">
                <div class="flow-icon">🕸️</div>
                <div class="flow-title">KG Linking</div>
                <div class="flow-desc">76K+ entities</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_example_queries():
    """Render example queries for optics domain"""
    st.markdown("### 💡 Example Queries")
    
    # Use queries from example_queries.py
    optics_examples = ALL_QUERIES[:6] if len(ALL_QUERIES) >= 6 else ALL_QUERIES
    
    for example in optics_examples:
        if st.button(f"💡 {example}", key=f"example_{hash(example)}"):
            st.session_state.current_query = example
            st.rerun()


def _format_source_name(source: str) -> str:
    """Format source for display; web:arxiv -> arXiv, web:pubmed -> PubMed, etc."""
    if not source or source == 'Unknown':
        return 'Unknown'
    if str(source).startswith('web:'):
        label = str(source)[4:].replace('_', ' ').title()
        return f"🌐 {label}"
    return Path(source).name


def render_document_card(doc: Dict[str, Any], index: int):
    """Render a single document result card with expandable content"""
    full_content = doc.get('content', '')
    content_preview = full_content[:300] if len(full_content) > 300 else full_content
    source = doc.get('source', 'Unknown')
    similarity = doc.get('similarity', 0.0)
    rerank_score = doc.get('rerank_score', None)
    
    source_name = _format_source_name(source)
    
    # Format content with LaTeX support
    content_preview_formatted = format_latex_for_display(content_preview)
    full_content_formatted = format_latex_for_display(full_content)
    
    # Prioritize rerank_score, then similarity
    # Note: rerank_score can be None or 0, similarity might also be 0
    score_display = rerank_score if rerank_score is not None else similarity
    
    # If both are 0 or very low, there might be a data issue
    if score_display == 0.0:
        score_display = 0.001  # Show minimal score to indicate potential issue
        score_label = "Score: ~0 (check data)"
    else:
        score_label = f"Score: {score_display:.3f}"
    
    score_color = "#10b981" if score_display > 0.7 else "#f59e0b" if score_display > 0.5 else "#ef4444"
    
    # Preview card
    st.markdown(f"""
    <div class="doc-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <strong style="color: #1e40af;">📄 Document {index + 1}</strong>
            <span class="doc-score" style="background: {score_color};">
                {score_label}
            </span>
        </div>
        <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.8rem;">
            📁 {source_name[:60]}{'...' if len(source_name) > 60 else ''}
        </div>
        <div style="color: #334155; line-height: 1.5;">
            {content_preview_formatted}{'...' if len(full_content) > 300 else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Expandable full content
    if len(full_content) > 300:
        with st.expander(f"📖 View Full Content ({len(full_content)} characters)"):
            st.markdown(f"""
            <div style="padding: 1rem; background: #f8fafc; border-radius: 8px; line-height: 1.8;">
                {full_content_formatted}
            </div>
            """, unsafe_allow_html=True)
            
            # Trigger MathJax for expanded content
            st.markdown("""
            <script>
            if (window.MathJax) {
                MathJax.typesetPromise();
            }
            </script>
            """, unsafe_allow_html=True)


def create_knowledge_graph(query_result: Dict[str, Any], agent):
    """Create interactive knowledge graph visualization using real Neo4j data"""
    
    # Get query from result
    query = query_result.get('query', '')
    
    if not query:
        # Return empty graph
        fig = go.Figure()
        fig.add_annotation(
            text="No query provided",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#f8fafc',
            height=400
        )
        return fig
    
    # Query real Neo4j data
    try:
        neo4j_manager = agent.rag_components.neo4j_manager
        kg_data = query_kg_data(neo4j_manager, query, max_nodes=20)
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])
        
        # Create visualization
        fig = create_kg_visualization(
            nodes, 
            edges,
            title=f'Knowledge Graph: "{query[:50]}..."' if len(query) > 50 else f'Knowledge Graph: "{query}"'
        )
        
        return fig
        
    except Exception as e:
        # Fallback to error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading knowledge graph: {str(e)[:100]}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#ef4444")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#f8fafc',
            height=400
        )
        return fig


def main():
    """Main application"""
    
    # Initialize session state
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'query_result' not in st.session_state:
        st.session_state.query_result = None
    if 'show_all_docs' not in st.session_state:
        st.session_state.show_all_docs = True
    if 'enable_web_search' not in st.session_state:
        st.session_state.enable_web_search = bool(getattr(config, 'enable_web_search', False))
    if 'web_search_mode' not in st.session_state:
        st.session_state.web_search_mode = getattr(config, 'web_search_mode', 'both')
    if 'web_search_results' not in st.session_state:
        st.session_state.web_search_results = int(getattr(config, 'web_search_results', 5))
    if 'agent' not in st.session_state:
        with st.spinner("🔧 Initializing Optics RAG System..."):
            st.session_state.agent = AgentOrchestrator(
                max_iterations=config.max_iterations,
                quality_threshold=config.quality_threshold,
                enable_web_search=st.session_state.enable_web_search,
                verbose=False
            )
            # Ensure UI state (mode/results) is applied on first load.
            st.session_state.agent.configure_web_search(
                enabled=st.session_state.enable_web_search,
                mode=st.session_state.web_search_mode,
                max_results=st.session_state.web_search_results
            )
    
    # Render header
    render_header()
    
    # Render RAG flow
    render_rag_flow()
    
    # Main layout: 3 columns
    col_left, col_middle, col_right = st.columns([1, 2, 1])
    
    # ===== LEFT COLUMN: Query Input =====
    with col_left:
        st.markdown("### 🔍 Query Interface")
        
        # Query input
        query_input = st.text_area(
            "Enter your optics/photonics question:",
            value=st.session_state.current_query,
            height=120,
            placeholder="e.g., What is the highest efficiency of perovskite solar cells?"
        )
        
        # Configuration
        with st.expander("⚙️ Advanced Settings"):
            st.markdown("**Query Parameters**")
            max_iter = st.slider("Max Iterations", 1, 5, config.max_iterations,
                               help="Maximum number of refinement iterations")
            quality_thresh = st.slider("Quality Threshold", 0.5, 1.0, config.quality_threshold, 0.05,
                                     help="Stop when answer quality exceeds this threshold")
            enable_kg = st.checkbox("Enable KG Enhancement", value=True,
                                   help="Use knowledge graph to enhance answers")
            top_k = st.slider("Top-K Documents", 3, 20, 10,
                            help="Number of documents to retrieve")

            st.markdown("**Web Search Settings**")
            enable_web_search = st.checkbox(
                "Enable Web Search",
                value=st.session_state.enable_web_search,
                help="Enable online search tool for latest/recent queries"
            )
            web_modes = ["both", "academic", "duckduckgo"]
            mode_value = st.session_state.web_search_mode if st.session_state.web_search_mode in web_modes else "both"
            web_search_mode = st.selectbox(
                "Web Search Mode",
                options=web_modes,
                index=web_modes.index(mode_value),
                disabled=not enable_web_search,
                help="both: academic first + DDG fallback; academic: only academic APIs; duckduckgo: only DDG"
            )
            web_search_results = st.slider(
                "Web Results",
                min_value=1,
                max_value=10,
                value=int(st.session_state.web_search_results),
                disabled=not enable_web_search,
                help="Maximum number of web contexts per web-search tool call"
            )
            
            st.markdown("**Display Options**")
            show_all_docs = st.checkbox("Show all retrieved documents", value=True, 
                                       help="If unchecked, only top 5 will be shown")
        
        # Store settings in session state
        st.session_state.show_all_docs = show_all_docs
        st.session_state.max_iter = max_iter
        st.session_state.quality_thresh = quality_thresh
        st.session_state.enable_kg = enable_kg
        st.session_state.top_k = top_k
        st.session_state.enable_web_search = enable_web_search
        st.session_state.web_search_mode = web_search_mode
        st.session_state.web_search_results = web_search_results
        
        # Query button
        if st.button("🚀 Run Query", type="primary", use_container_width=True):
            if query_input.strip():
                st.session_state.current_query = query_input
                
                # Apply settings to agent before query
                agent = st.session_state.agent
                agent.max_iterations = max_iter
                agent.quality_threshold = quality_thresh
                agent.top_k = top_k
                agent.enable_kg = enable_kg
                config.enable_web_search = enable_web_search
                config.web_search_mode = web_search_mode
                config.web_search_results = web_search_results
                agent.configure_web_search(
                    enabled=enable_web_search,
                    mode=web_search_mode,
                    max_results=web_search_results
                )
                
                # Also update reflection engine's threshold
                agent.reflection_engine.quality_threshold = quality_thresh
                
                with st.spinner("🤔 Processing your query..."):
                    result = agent.query(query_input)
                    st.session_state.query_result = result
                
                st.success("✅ Query completed!")
                st.rerun()
            else:
                st.warning("Please enter a query first")
        
        # Example queries
        st.markdown("---")
        render_example_queries()
    
    # ===== MIDDLE COLUMN: Results Display =====
    with col_middle:
        st.markdown("### 🤖 AI-Generated Explanation")
        
        if st.session_state.query_result:
            result = st.session_state.query_result
            
            # Metrics row
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{result.get('iterations', 0)}</div>
                    <div class="metric-label">Iterations</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[1]:
                quality = result.get('quality_score', 0.0)
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{quality:.2f}</div>
                    <div class="metric-label">Quality</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[2]:
                contexts = result.get('context_count', 0)
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{contexts}</div>
                    <div class="metric-label">Contexts</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[3]:
                time_taken = result.get('total_time', 0.0)
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{time_taken:.1f}s</div>
                    <div class="metric-label">Time</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Answer display with MathJax support
            answer = result.get('final_answer', 'No answer generated')
            
            # Format LaTeX for display
            answer_html = format_latex_for_display(answer)
            
            st.markdown(f"""
            <div class="opti-card">
                <div class="card-title">💡 Answer</div>
                <div class="card-content">{answer_html}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Trigger MathJax rendering
            st.markdown("""
            <script>
            if (window.MathJax) {
                MathJax.typesetPromise();
            }
            </script>
            """, unsafe_allow_html=True)
            
            # Query Configuration Info
            with st.expander("⚙️ Query Configuration Used"):
                config_info = result.get('config', {})
                agent = st.session_state.agent
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Query Parameters:**")
                    st.write(f"• Max Iterations: {config_info.get('max_iterations', agent.max_iterations)}")
                    st.write(f"• Quality Threshold: {config_info.get('quality_threshold', agent.quality_threshold):.2f}")
                    st.write(f"• Top-K Documents: {agent.top_k}")
                
                with col2:
                    st.markdown("**Features:**")
                    st.write(f"• KG Enhancement: {'✅ Enabled' if agent.enable_kg else '❌ Disabled'}")
                    st.write(f"• Reranking: {'✅ Enabled' if config.enable_reranking else '❌ Disabled'}")
                    st.write(f"• Web Search: {'✅ Enabled' if agent.enable_web_search else '❌ Disabled'}")
                    st.write(f"• Web Mode: {getattr(config, 'web_search_mode', 'both')}")
            
            # Iteration history
            if result.get('iteration_history'):
                with st.expander("📊 View Iteration Details"):
                    for i, iter_data in enumerate(result['iteration_history'], 1):
                        st.markdown(f"**Iteration {i}**")
                        st.write(f"- Query: {iter_data.get('query', 'N/A')}")
                        st.write(f"- Tools: {iter_data.get('plan', {}).get('required_tools', [])}")
                        st.write(f"- Quality: {iter_data.get('quality_score', 0.0):.3f}")
                        st.markdown("---")
        
        else:
            st.info("👈 Enter a query to see AI-generated results here")
            
            # Placeholder
            st.markdown("""
            <div class="opti-card">
                <div class="card-title">Welcome to OptiRAG!</div>
                <div class="card-content">
                    This system combines advanced RAG with knowledge graph technology to provide
                    accurate answers to your optics and photonics questions. 
                    <br><br>
                    <strong>Features:</strong>
                    <ul>
                        <li>Multi-iteration refinement for better answers</li>
                        <li>Document reranking with qwen3-reranker-8b</li>
                        <li>Knowledge graph integration</li>
                        <li>Real-time quality assessment</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ===== RIGHT COLUMN: Retrieved Documents =====
    with col_right:
        if st.session_state.query_result:
            result = st.session_state.query_result
            contexts = result.get('contexts', [])
            web_contexts = [c for c in contexts if str(c.get('source', '')).startswith('web')]
            other_contexts = [c for c in contexts if not str(c.get('source', '')).startswith('web')]

            # 🌐 Web Search Results (dedicated section)
            st.markdown("### 🌐 Web Search Results")
            web_status = result.get('web_search', {})
            web_triggered = web_status.get('triggered', False)
            web_fetched = web_status.get('total_fetched', 0)
            if web_contexts:
                st.markdown(f"<span class='status-badge status-success'>✓ {len(web_contexts)} web results</span>",
                            unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="scrollable-docs">', unsafe_allow_html=True)
                for i, doc in enumerate(web_contexts):
                    render_document_card(doc, i)
                st.markdown('</div>', unsafe_allow_html=True)
            elif web_triggered and web_fetched == 0:
                st.warning(
                    "Web search ran but returned 0 results. Possible causes: network restrictions, "
                    "API limits, or DuckDuckGo unavailable. Try checking network or using academic mode."
                )
            elif web_triggered:
                st.info("Web search ran but results were filtered. Try a different query.")
            else:
                st.info("Enable Web Search in sidebar (⚙️ Advanced Settings) and run a definition/latest query.")

            st.markdown("---")
            st.markdown("### 📚 Retrieved Documents")
            
            if other_contexts:
                st.markdown(f"<span class='status-badge status-success'>✓ {len(other_contexts)} documents found</span>", 
                          unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="text-align: center; padding: 0.5rem; background: #eff6ff; 
                            border-radius: 6px; margin-bottom: 0.5rem;">
                    <span style="color: #1e40af; font-size: 0.85rem;">
                        📚 {len(other_contexts)} documents (vector + KG)
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="scrollable-docs">', unsafe_allow_html=True)
                for i, doc in enumerate(other_contexts):
                    render_document_card(doc, i)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No documents retrieved")
            
            if contexts:
                st.markdown("""
                <script>
                if (window.MathJax) {
                    MathJax.typesetPromise();
                }
                </script>
                """, unsafe_allow_html=True)
        else:
            st.markdown("### 🌐 Web Search Results")
            st.info("Web results will appear here after running a query")
            st.markdown("---")
            st.markdown("### 📚 Retrieved Documents")
            st.info("Documents will appear here after running a query")
    
    # ===== BOTTOM: Knowledge Graph =====
    st.markdown("---")
    
    # Knowledge Graph Section with dual view
    st.markdown("### 🕸️ Knowledge Graph Visualization")
    
    if st.session_state.query_result:
        # Get query-relevant KG data
        query = st.session_state.query_result.get('query', '')
        neo4j_manager = st.session_state.agent.rag_components.neo4j_manager
        kg_data = query_kg_data(neo4j_manager, query, max_nodes=20)
        stats = get_kg_statistics(kg_data.get('nodes', []), kg_data.get('edges', []))
        
        # Create two-column layout
        overview_col, detail_col = st.columns([1, 2])
        
        with overview_col:
            st.markdown("#### 🌐 Full Graph Overview")
            st.markdown(f"""
            <div style="padding: 0.5rem; background: #f1f5f9; border-radius: 6px; text-align: center; margin-bottom: 0.5rem;">
                <div style="font-size: 0.75rem; color: #64748b;">
                    <span style="color: #ef4444; font-weight: bold;">●</span> Query-Related
                    <span style="color: #cbd5e1; margin-left: 0.5rem;">●</span> Other
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get full graph overview
            try:
                with st.spinner("Loading full graph overview..."):
                    full_kg_data = get_full_kg_overview(neo4j_manager, sample_size=100)
                    relevant_node_ids = [node['id'] for node in kg_data.get('nodes', [])]
                    
                    total_entities = full_kg_data.get('total_entities', 0)
                    
                    if full_kg_data.get('nodes') and len(full_kg_data.get('nodes', [])) > 0:
                        overview_fig = create_overview_visualization(
                            full_kg_data.get('nodes', []),
                            full_kg_data.get('edges', []),
                            highlighted_nodes=relevant_node_ids
                        )
                        st.plotly_chart(overview_fig, use_container_width=True, key="overview_graph")
                        
                        # Show total stats
                        full_stats = get_kg_statistics(full_kg_data.get('nodes', []), full_kg_data.get('edges', []))
                        st.markdown(f"""
                        <div style="padding: 0.3rem; background: #eff6ff; border-radius: 4px; text-align: center; font-size: 0.8rem;">
                            📊 Sample: {full_stats['total_nodes']} of {total_entities} total • {full_stats['total_edges']} edges
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Show helpful message
                        if total_entities == 0:
                            st.warning("📊 Knowledge Graph is Empty")
                            st.info("""
                            **To build the knowledge graph:**
                            
                            ```bash
                            cd /home/wzfeng/RAGPhoto/RAGPhoto_5
                            python run_pipeline_parallel.py --kg --force-rebuild --num-workers 10
                            ```
                            
                            This will extract entities and relationships from your documents.
                            """)
                        else:
                            st.info(f"Found {total_entities} entities but unable to sample them")
            except Exception as e:
                st.error(f"Unable to load knowledge graph overview")
                with st.expander("🐛 Show Error Details"):
                    st.code(str(e))
                    import traceback
                    st.code(traceback.format_exc())
        
        with detail_col:
            st.markdown("#### 🔍 Query-Relevant Subgraph")
            st.markdown(f"""
            <div style="padding: 0.5rem; background: #eff6ff; border-radius: 6px; text-align: center; margin-bottom: 0.5rem;">
                <div style="font-size: 0.8rem; color: #1e40af;">Matched Entities & Relationships</div>
                <div style="font-size: 1.1rem; font-weight: bold; color: #1e40af;">
                    {stats['total_nodes']} nodes • {stats['total_edges']} edges
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create detailed view
            try:
                with st.spinner("Loading query-relevant subgraph..."):
                    fig = create_knowledge_graph(st.session_state.query_result, st.session_state.agent)
                    st.plotly_chart(fig, use_container_width=True, key="detail_graph")
            except Exception as e:
                st.error(f"Unable to load detailed graph")
                st.caption(f"Error: {str(e)[:100]}")
        
        # Show detailed data in tables
        if stats['total_nodes'] > 0:
            st.markdown("---")
            st.markdown("#### 📊 Detailed Knowledge Graph Data")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📋 Entity List", "🔗 Relationship List", "📈 Statistics"])
            
            with tab1:
                st.markdown("**All Entities in Query-Relevant Subgraph:**")
                
                # Create dataframe for entities
                entity_data = []
                for node in kg_data.get('nodes', []):
                    entity_data.append({
                        'Name': node.get('name', 'Unknown'),
                        'Type': node.get('type', 'Unknown'),
                        'Description': (node.get('description', '')[:100] + '...') if len(node.get('description', '')) > 100 else node.get('description', 'N/A')
                    })
                
                if entity_data:
                    import pandas as pd
                    df_entities = pd.DataFrame(entity_data)
                    st.dataframe(
                        df_entities,
                        use_container_width=True,
                        hide_index=True,
                        height=300
                    )
                    
                    # Show entities with LaTeX rendering in expandable cards
                    with st.expander("🔬 View Entities with LaTeX Rendering"):
                        for node in kg_data.get('nodes', []):
                            name = node.get('name', 'Unknown')
                            node_type = node.get('type', 'Unknown')
                            desc = node.get('description', 'N/A')
                            
                            # Format with LaTeX
                            formatted_name = format_latex_for_display(name)
                            formatted_desc = format_latex_for_display(desc)
                            
                            st.markdown(f"""
                            <div style="padding: 0.5rem; margin: 0.3rem 0; background: #f8fafc; border-left: 3px solid #3b82f6; border-radius: 4px;">
                                <div style="font-weight: bold; color: #1e40af;">
                                    {formatted_name} <span style="color: #64748b; font-size: 0.85em;">({node_type})</span>
                                </div>
                                <div style="font-size: 0.9em; color: #475569; margin-top: 0.2rem;">
                                    {formatted_desc[:200]}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Trigger MathJax rendering
                        st.markdown("""
                        <script>
                        if (window.MathJax) {
                            MathJax.typesetPromise();
                        }
                        </script>
                        """, unsafe_allow_html=True)
                    
                    # Download button
                    csv = df_entities.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Entities CSV",
                        data=csv,
                        file_name="knowledge_graph_entities.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No entities found")
            
            with tab2:
                st.markdown("**All Relationships in Query-Relevant Subgraph:**")
                
                # Create dataframe for relationships
                relationship_data = []
                nodes_dict = {node['id']: node.get('name', 'Unknown') for node in kg_data.get('nodes', [])}
                
                for edge in kg_data.get('edges', []):
                    source_name = nodes_dict.get(edge.get('source'), 'Unknown')
                    target_name = nodes_dict.get(edge.get('target'), 'Unknown')
                    rel_type = edge.get('type', 'Unknown')
                    
                    relationship_data.append({
                        'Source': source_name,
                        'Relationship': rel_type,
                        'Target': target_name
                    })
                
                if relationship_data:
                    import pandas as pd
                    df_relationships = pd.DataFrame(relationship_data)
                    st.dataframe(
                        df_relationships,
                        use_container_width=True,
                        hide_index=True,
                        height=300
                    )
                    
                    # Show relationships with LaTeX rendering
                    with st.expander("🔬 View Relationships with LaTeX Rendering"):
                        for edge in kg_data.get('edges', []):
                            source_name = nodes_dict.get(edge.get('source'), 'Unknown')
                            target_name = nodes_dict.get(edge.get('target'), 'Unknown')
                            rel_type = edge.get('type', 'Unknown')
                            
                            # Format with LaTeX
                            formatted_source = format_latex_for_display(source_name)
                            formatted_target = format_latex_for_display(target_name)
                            formatted_rel = format_latex_for_display(rel_type)
                            
                            st.markdown(f"""
                            <div style="padding: 0.5rem; margin: 0.3rem 0; background: #f0f9ff; border-radius: 4px;">
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <span style="color: #1e40af; font-weight: bold;">{formatted_source}</span>
                                    <span style="color: #64748b;">→</span>
                                    <span style="color: #059669; font-style: italic; font-size: 0.9em;">{formatted_rel}</span>
                                    <span style="color: #64748b;">→</span>
                                    <span style="color: #1e40af; font-weight: bold;">{formatted_target}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Trigger MathJax rendering
                        st.markdown("""
                        <script>
                        if (window.MathJax) {
                            MathJax.typesetPromise();
                        }
                        </script>
                        """, unsafe_allow_html=True)
                    
                    # Download button
                    csv = df_relationships.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Relationships CSV",
                        data=csv,
                        file_name="knowledge_graph_relationships.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No relationships found")
            
            with tab3:
                st.markdown("**Summary Statistics:**")
                
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.markdown("**Entity Types:**")
                    for entity_type, count in sorted(stats['node_types'].items(), key=lambda x: x[1], reverse=True):
                        st.write(f"• {entity_type}: {count}")
                
                with stats_col2:
                    st.markdown("**Relationship Types:**")
                    for rel_type, count in sorted(stats['edge_types'].items(), key=lambda x: x[1], reverse=True):
                        st.write(f"• {rel_type}: {count}")
                
                # Overall stats
                st.markdown("---")
                st.markdown("**Overall Metrics:**")
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Total Entities", stats['total_nodes'])
                with metric_cols[1]:
                    st.metric("Total Relationships", stats['total_edges'])
                with metric_cols[2]:
                    st.metric("Entity Types", len(stats['node_types']))
                with metric_cols[3]:
                    st.metric("Relation Types", len(stats['edge_types']))
    else:
        st.info("Knowledge graph will be displayed here after running a query")


if __name__ == "__main__":
    main()

