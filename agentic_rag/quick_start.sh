#!/bin/bash

# Agentic RAG v2 - Quick Start Script
# This script helps you get started with Agentic RAG v2

set -e

PROJECT_ROOT="/data/wzfeng/RAGPhoto/Data_Agentic_RAG/agentic_rag"
RAGPHOTO5_ROOT="/data/wzfeng/RAGPhoto/Data_Agentic_RAG/data_pipeline"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=================================================="
echo "🤖 Agentic RAG v2 - Quick Start"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "$PROJECT_ROOT/run_agentic.py" ]; then
    echo -e "${RED}❌ Error: Please run this script from the project root${NC}"
    exit 1
fi

cd "$PROJECT_ROOT"

# Step 1: Check environment file
echo -e "${YELLOW}Step 1: Checking configuration...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}  ⚠️  .env file not found. Creating from template...${NC}"
    cp config/env_example.txt .env
    echo -e "${GREEN}  ✅ .env file created. Please edit it and set OPENAI_API_KEY${NC}"
    echo -e "${YELLOW}  📝 Run: nano .env${NC}"
    exit 0
else
    echo -e "${GREEN}  ✅ .env file exists${NC}"
fi

# Step 2: Check Python dependencies
echo -e "${YELLOW}Step 2: Checking dependencies...${NC}"
if python3 -c "import streamlit" 2>/dev/null; then
    echo -e "${GREEN}  ✅ Dependencies installed${NC}"
else
    echo -e "${YELLOW}  ⚠️  Installing dependencies...${NC}"
    pip3 install -r requirements.txt
    echo -e "${GREEN}  ✅ Dependencies installed${NC}"
fi

# Step 3: Check services
echo -e "${YELLOW}Step 3: Checking services...${NC}"

# Check Milvus
if docker ps | grep -q milvus; then
    echo -e "${GREEN}  ✅ Milvus is running${NC}"
else
    echo -e "${RED}  ❌ Milvus is not running${NC}"
    echo -e "${YELLOW}  💡 Start with: cd $RAGPHOTO5_ROOT && docker compose -f config/docker-compose.yml up -d${NC}"
fi

# Check Neo4j
if docker ps | grep -q neo4j; then
    echo -e "${GREEN}  ✅ Neo4j is running${NC}"
else
    echo -e "${RED}  ❌ Neo4j is not running${NC}"
    echo -e "${YELLOW}  💡 Start with: docker start neo4j${NC}"
fi

# Step 4: Show usage options
echo ""
echo "=================================================="
echo "🚀 Ready to Use!"
echo "=================================================="
echo ""
echo "Choose how to run Agentic RAG v2:"
echo ""
echo "1️⃣  Web UI (OptiRAG) - Recommended:"
echo "   ./run_optics_ui.sh"
echo "   (Or with cache cleaning: ./run_optics_ui.sh --clean)"
echo ""
echo "2️⃣  CLI - Single Query:"
echo "   python run_agentic.py \"What is perovskite solar cell efficiency?\""
echo ""
echo "3️⃣  CLI - Interactive Mode:"
echo "   python run_agentic.py --interactive"
echo ""
echo "4️⃣  CLI - Batch Evaluation:"
echo "   python run_agentic.py --evaluate --sample-size 5"
echo ""
echo "📚 Documentation:"
echo "   - Project README: ../README.md"
echo "   - Configuration: config/env_example.txt"
echo ""
echo "🎯 Useful Scripts:"
echo "   - run_optics_ui.sh     → Start OptiRAG web interface"
echo "   - run_agentic.py       → CLI tool for queries"
echo ""
echo "=================================================="

