#!/bin/bash
# OptiRAG UI Launcher
# Launches the Optics & Photonics RAG interface with all checks and optimizations
# Requires: Neo4j (port 7688) and Milvus (port 19531)

echo "🔬 OptiRAG - Optics & Photonics Knowledge System"
echo "=========================================="

# Check if in correct directory
if [ ! -f "ui/app_optics.py" ]; then
    echo "❌ Error: Please run this script from the Data_Agentic_RAG/agentic_rag directory"
    exit 1
fi

# Option to clean cache
if [ "$1" == "--clean" ] || [ "$1" == "-c" ]; then
    echo "🧹 Cleaning Python cache..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    echo "✅ Cache cleaned"
    echo ""
fi

# Check Neo4j service
echo "🔍 Checking services..."
if ! docker ps | grep -q test-neo4j; then
    echo "⚠️  Neo4j is not running!"
    echo "   Start it with: cd ../data_pipeline/config && docker compose up -d"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Neo4j is running (port 7688)"
fi

# Check Milvus service
if ! docker ps | grep -q test-milvus-standalone; then
    echo "⚠️  Milvus container is not running!"
    echo "   Start it with: cd ../data_pipeline/config && docker compose up -d"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Milvus is running (port 19531)"
fi

echo ""

# Check if environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Recommended: conda activate agentic_rag"
    echo ""
fi

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import streamlit" 2>/dev/null || {
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
}

python3 -c "import plotly" 2>/dev/null || {
    echo "📊 Installing plotly..."
    pip install plotly kaleido
}

python3 -c "import networkx" 2>/dev/null || {
    echo "📊 Installing networkx..."
    pip install networkx
}

echo ""
echo "✅ All dependencies ready"
echo ""
echo "🚀 Launching OptiRAG UI..."
echo "   URL: http://localhost:8503"
echo "   Database: academic_papers (67,821 vectors)"
echo "   Knowledge Graph: 76,520 entities"
echo ""
echo "   Press Ctrl+C to stop"
echo ""
echo "💡 Tip: Use './run_optics_ui.sh --clean' to clear cache before starting"
echo "=========================================="
echo ""

# Set PYTHONPATH to project root and agentic_rag root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PARENT_DIR}:${PYTHONPATH}"

# Run streamlit
streamlit run ui/app_optics.py \
    --server.port 8503 \
    --server.address 0.0.0.0 \
    --theme.base light \
    --theme.primaryColor "#3b82f6" \
    --theme.backgroundColor "#f8fafc" \
    --theme.secondaryBackgroundColor "#ffffff"

