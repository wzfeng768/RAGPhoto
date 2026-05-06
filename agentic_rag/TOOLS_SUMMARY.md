# Agentic RAG 项目工具总结

本文档总结了 `agentic_rag` 项目中使用的所有工具。

## 工具概览

项目实现了以下工具来支持智能问答系统：

### 1. **Vector Retriever (向量检索工具)** ✅ 已实现

- **位置**: `core/rag_components.py` → `search_vectors()`
- **技术栈**: 
  - 向量数据库: **Milvus**
  - 嵌入模型: `text-embedding-3-small` (默认)
  - 嵌入维度: 1536
- **功能**: 
  - 基于语义相似度的文档检索
  - 支持 top_k 参数（默认 15）
  - 返回文档内容、来源、相似度分数等元数据
- **使用场景**: 所有查询的基础检索工具

### 2. **Knowledge Graph Retriever (知识图谱检索工具)** ✅ 已实现

- **位置**: `core/rag_components.py` → `query_knowledge_graph()`
- **技术栈**: 
  - 图数据库: **Neo4j**
  - 查询语言: Cypher
- **功能**: 
  - 从知识图谱中提取相关实体和关系
  - 支持关键词匹配查询
  - 返回实体列表、关系列表和摘要
  - 默认最多返回 10 个实体
- **使用场景**: 
  - 作为基础工具，在每次查询中使用（如果启用）
  - 在答案生成时集成知识图谱信息以增强答案质量

### 3. **Reranker (重排序工具)** ✅ 已实现

- **位置**: `core/reranker.py` → `Reranker` 类
- **技术栈**: 
  - 模型: **qwen3-reranker-8b**
  - API: OpenAI 兼容接口
  - API 地址: `https://www.dmxapi.cn/v1` (默认)
- **功能**: 
  - 对向量检索结果进行重新排序
  - 基于查询-文档相关性评分
  - 返回 top_n 个最相关的文档（默认 5）
  - 为每个文档添加 `rerank_score` 字段
- **使用场景**: 
  - 在向量检索后使用（如果启用）
  - 提高检索结果的相关性排序

### 4. **Web Search (网络搜索工具)** ✅ 已实现

- **位置**: 
  - `tools/web_search.py` → `AcademicMultiSourceSearcher` + `WebSearchTool`
  - `core/agent_orchestrator.py` → `_register_tools()` + `_execute_tools()`
- **状态**: 已实现并接入编排器
- **技术栈**:
  - 学术多源：arXiv + PubMed + Semantic Scholar + CORE
  - 回退搜索：DuckDuckGo
- **策略**:
  - 支持三种模式：`academic` / `duckduckgo` / `both`
  - `both` 模式下默认“学术多源优先，空结果时回退 DuckDuckGo”
- **使用场景**: 
  - 用于查询最新信息（`latest` 类型查询）
  - 默认首轮即可使用（可通过 `WEB_SEARCH_START_ITERATION` 调整）

## 工具调用流程

### 工具选择策略 (`_select_tools()`)

根据查询类型和迭代次数选择工具：

```python
# 基础工具（总是使用）
tools = ['vector']

# 如果启用 KG，总是添加
if self.enable_kg:
    tools.append('kg')

# 如果启用重排序，添加
if config.enable_reranking:
    tools.append('reranker')

# 对于最新信息查询，默认首轮可触发（可配置起始轮次）
if query_type == 'latest' and iteration >= WEB_SEARCH_START_ITERATION and self.enable_web_search:
    tools.append('web')
```

### 工具执行顺序 (`_execute_tools()`)

1. **Vector Retrieval** - 执行向量检索
2. **KG Query** - 查询知识图谱（如果启用）
3. **Reranking** - 对向量检索结果重排序（如果启用）
4. **Web Search** - 网络搜索（如果启用，已实现）

## 工具配置

所有工具配置在 `config/config.py` 中管理，通过环境变量设置：

### Vector Retriever 配置
```bash
VECTOR_TOP_K=15
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
MILVUS_HOST=localhost
MILVUS_PORT=19531
MILVUS_COLLECTION=academic_papers
```

### Knowledge Graph 配置
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
KG_MAX_ENTITIES=10
```

### Reranker 配置
```bash
ENABLE_RERANKING=true
RERANKER_MODEL=qwen3-reranker-8b
RERANKER_TOP_N=5
RERANKER_API_KEY=your_api_key_here
RERANKER_BASE_URL=https://www.dmxapi.cn/v1
```

### Web Search 配置
```bash
ENABLE_WEB_SEARCH=false
WEB_SEARCH_RESULTS=5
WEB_SEARCH_ENGINE=duckduckgo
WEB_SEARCH_MODE=both
WEB_ENABLED_SOURCES=arxiv,pubmed,semantic_scholar,core
SEMANTIC_SCHOLAR_API_KEY=
CORE_API_KEY=
PUBMED_EMAIL=your-email@example.com
WEB_SEARCH_TIMEOUT=10
```

## 工具接口设计

所有工具遵循统一的接口设计（定义在 `tools/base_tool.py`）：

```python
class BaseTool(ABC):
    @abstractmethod
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """执行工具逻辑"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """获取工具描述"""
        pass
```

**说明**: 目前 `WebSearchTool` 已按 `BaseTool` 独立实现并注册；`Vector/KG/Reranker` 仍主要通过 `RAGComponents` 直接集成。

## 工具集成位置

- **主要集成点**: `core/agent_orchestrator.py`
  - `_select_tools()`: 工具选择逻辑
  - `_execute_tools()`: 工具执行逻辑
  - `_observe()`: 工具结果观察和组织

- **组件封装**: `core/rag_components.py`
  - `RAGComponents` 类封装了所有底层工具的实际实现
  - 集成了 RAGPhoto_5 项目的组件

## 总结

| 工具 | 状态 | 技术栈 | 用途 |
|------|------|--------|------|
| Vector Retriever | ✅ 已实现 | Milvus + Embeddings | 语义检索 |
| Knowledge Graph | ✅ 已实现 | Neo4j | 结构化知识提取 |
| Reranker | ✅ 已实现 | qwen3-reranker-8b | 结果重排序 |
| Web Search | ✅ 已实现 | arXiv + PubMed + Semantic Scholar + CORE + DuckDuckGo | 最新信息检索 |

## 相关文件

- `core/agent_orchestrator.py` - 工具编排和执行
- `core/rag_components.py` - 工具组件封装
- `core/reranker.py` - 重排序工具实现
- `tools/base_tool.py` - 工具基类定义
- `tools/web_search.py` - Web 搜索工具实现（多源 + 回退）
- `tools/__init__.py` - 工具导出
- `config/config.py` - 工具配置管理


