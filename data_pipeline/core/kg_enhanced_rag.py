"""
知识图谱增强的RAG系统
KG-Enhanced RAG System with Hybrid Retrieval
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
from openai import OpenAI

from config.config import config
from .hybrid_retriever import HybridRetriever
from .reranker import RerankerModel  # ✅ 导入Reranker


class KGEnhancedRAGSystem:
    """知识图谱增强的RAG系统"""
    
    def __init__(self, verbose: bool = False, use_reranking: bool = True):
        """
        初始化增强RAG系统
        
        Args:
            verbose: 是否打印详细信息
            use_reranking: 是否启用重排序（默认True）
        """
        self.verbose = verbose
        self.use_reranking = use_reranking
        
        # 初始化混合检索器
        self.retriever = HybridRetriever(verbose=verbose)
        
        # 初始化LLM客户端
        self.llm_client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        # ✅ 初始化Reranker
        if self.use_reranking:
            try:
                self.reranker = RerankerModel()
                if self.verbose:
                    print("✅ Reranker initialized successfully")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Reranker initialization failed: {e}")
                    print("   Continuing without reranking...")
                self.use_reranking = False
                self.reranker = None
        else:
            self.reranker = None
        
        if self.verbose:
            print("✅ KG-Enhanced RAG system initialized successfully")
    
    def _clean_latex_numbers(self, text: str) -> str:
        """
        清理LaTeX格式的数字，移除空格
        例如: $1 5 . 2 1 \\%$ → $15.21\\%$
        """
        import re
        
        # 匹配 $数字空格数字$ 的模式并移除空格
        def clean_number(match):
            full_match = match.group(0)
            cleaned = full_match.replace(' ', '')
            return cleaned
        
        # 匹配LaTeX数字格式
        pattern = r'\$\s*[\d\s\.]+\s*\\?%\s*'
        cleaned_text = re.sub(pattern, clean_number, text)
        
        return cleaned_text
    
    def build_enhanced_context(
        self, 
        vector_docs: List[Dict[str, Any]], 
        kg_knowledge: str = None
    ) -> str:
        """Build enhanced context - Documents first, then KG (to prioritize document content)"""
        context_parts = []
        
        # ✅ 1. Add document chunks FIRST (primary source of information)
        context_parts.append("=== PRIMARY SOURCE: Detailed Content from Literature ===\n")
        context_parts.append("**IMPORTANT**: Read these documents carefully. They contain the most detailed and accurate information.\n")
        for i, doc in enumerate(vector_docs[:5], 1):  # Limit to top 5
            # ✅ 修复: 使用'content'而不是'text'
            content = doc.get('content', '') or doc.get('text', '')
            source = doc.get('source', 'Unknown')
            
            # ✅ 清理LaTeX数字格式
            content = self._clean_latex_numbers(content)
            
            context_parts.append(f"[Document {i}] (Source: {source})")
            context_parts.append(content)
            context_parts.append("\n")
        
        # ✅ 2. Add structured knowledge from knowledge graph (if available) - SECONDARY source
        if kg_knowledge:
            context_parts.append("\n" + "="*50 + "\n")
            context_parts.append("=== SUPPLEMENTARY: Structured Knowledge from Knowledge Graph ===\n")
            context_parts.append("**NOTE**: This is supplementary information. If KG information conflicts with documents, trust documents.\n")
            context_parts.append(kg_knowledge)
        
        return "\n".join(context_parts)
    
    def _detect_question_type(self, query: str) -> str:
        """
        检测问题类型以选择最佳回答策略
        
        Returns:
            问题类型: 'numerical', 'comparison', 'conceptual', 'mechanistic', 'general'
        """
        query_lower = query.lower()
        
        # 数值查询问题
        numerical_keywords = [
            'pce', 'efficiency', 'percentage', '%', 'ratio',
            'how much', 'how many', 'what is the value', 'what number',
            'what were the', 'achieved', 'reported', 'measured',
            'power conversion', 'conversion efficiency'
        ]
        if any(kw in query_lower for kw in numerical_keywords):
            return "numerical"
        
        # 对比分析问题
        comparison_keywords = [
            'compare', 'comparison', 'difference', 'versus', 'vs', 'vs.',
            'better', 'worse', 'higher', 'lower', 'more', 'less',
            'how do', 'how does', 'differ', 'contrast'
        ]
        if any(kw in query_lower for kw in comparison_keywords):
            return "comparison"
        
        # 原因/机制问题
        mechanistic_keywords = [
            'why', 'how does', 'how did', 'mechanism', 'reason',
            'cause', 'explain how', 'what causes', 'how to',
            'process', 'method'
        ]
        if any(kw in query_lower for kw in mechanistic_keywords):
            return "mechanistic"
        
        # 概念定义问题
        conceptual_keywords = [
            'what is', 'what are', 'define', 'definition',
            'characteristics', 'properties', 'features',
            'what does', 'meaning of'
        ]
        if any(kw in query_lower for kw in conceptual_keywords):
            return "conceptual"
        
        # 默认为一般问题
        return "general"
    
    def generate_answer(
        self, 
        query: str, 
        context: str,
        has_kg_knowledge: bool = False,
        question_type: str = None
    ) -> str:
        """Generate answer using LLM with question-type-aware prompts"""
        
        # ✅ 统一使用增强提示词，无论是否有KG知识
        # 注意：has_kg_knowledge 可能为 False，但这是正常的（KG可能没有找到相关信息）
        # 即使没有KG知识，也使用相同的提示词策略，保持一致性
        
        # 检测问题类型（如果未提供）
        if question_type is None:
            question_type = self._detect_question_type(query)
        
        # 根据问题类型选择策略（统一使用增强提示词）
        if question_type == "numerical":
            # 数值问题：使用极其严格的提示词，防止幻觉
            strategy = """**Question Type: Numerical Query**

This question asks for specific numbers, measurements, or quantitative data.

CRITICAL RULES (MUST FOLLOW):
1. **ONLY report numbers that appear EXPLICITLY in the literature documents**
2. **DO NOT infer, estimate, or calculate new numbers**
3. **DO NOT use numbers from your training data or memory**
4. **Use Knowledge Graph for context, but extract numerical values from documents**
5. **If a requested number is not in the documents, clearly state "not specified in the documents"**

HOW TO EXTRACT NUMBERS:
- **LaTeX format with spaces**: $1 5 . 2 1 \\%$ means 15.21% (remove spaces between digits)
- **LaTeX format examples**: 
  * $1 6 . 5 5 \\%$ → 16.55%
  * $13 \\%$ → 13%
  * $1 4 - 1 7 \\%$ → range 14-17% (report as range, don't pick one end)
- Look for phrases like "PCE of", "efficiency reached", "efficiency of", "achieved", "reported"
- When you find a LaTeX formatted number, INTERPRET it (remove spaces) and report the clean value
- Cite which document contains each number

FORBIDDEN:
- ❌ Rounding or averaging numbers from the documents
- ❌ Combining multiple values into a summary value
- ❌ Using numbers from ranges (e.g., if documents say "14-17%", don't report "17%")
- ❌ Making up numbers based on similar examples"""

        elif question_type == "comparison":
            # 对比问题：强调文档详细内容
            strategy = """**Question Type: Comparison/Contrast**

This question asks to compare or contrast different entities, methods, or results.

STRATEGY:
- **READ the literature documents thoroughly** for comparative details
- **USE Knowledge Graph to understand relationships** between entities
- Look for explicit comparisons, phrases like "compared to", "higher/lower than", "better/worse"
- The KG shows relationships, and documents contain the detailed comparison data
- Provide specific comparative information and explain the differences
- Be thorough in your comparison"""

        elif question_type == "conceptual":
            # 概念问题：KG+文档结合
            strategy = """**Question Type: Conceptual/Definition**

This question asks about concepts, definitions, or characteristics.

STRATEGY:
- **USE both KG and documents together**
- Start with the KG to see the conceptual structure and relationships
- Then USE the documents to provide detailed explanations and context
- COMBINE: KG gives the framework, documents give the details
- Provide a clear, comprehensive explanation"""

        elif question_type == "mechanistic":
            # 机制/原因问题：KG+文档结合
            strategy = """**Question Type: Mechanism/Causation**

This question asks about how or why something works, mechanisms, or causes.

STRATEGY:
- **USE both KG and documents together**
- The KG shows causal relationships and process structures
- The documents provide detailed mechanistic explanations
- Look for causal relationships, processes, and step-by-step descriptions
- Explain the process or reason clearly and thoroughly
- Use specific information from both KG and documents"""

        else:  # general
            # 一般问题：平衡策略
            strategy = """**Question Type: General**

STRATEGY:
- **Use both KG and documents** as complementary sources
- KG provides conceptual overview and relationships
- Documents provide detailed information
- Read documents thoroughly for factual information"""

        # 构建完整的系统提示词（统一使用增强提示词）
        system_prompt = f"""You are a professional AI assistant specialized in photovoltaic domain.

{strategy}

CONTEXT STRUCTURE:
1. **Knowledge Graph (KG)**: Shows entities and relationships (e.g., "Device HAS_PROPERTY Metric")
   - Structural overview and relationships between concepts
   - Use KG to understand the conceptual framework
   - **NOTE**: KG may not contain all details; always check documents for specific information
2. **Literature Documents**: Contain actual content with data, measurements, and details
   - May include LaTeX formatted numbers
   - **PRIMARY SOURCE**: Documents contain the most detailed and accurate information
   - **CRITICAL**: Always read documents carefully before claiming information is missing

GENERAL GUIDELINES:
- **READ DOCUMENTS FIRST**: Documents are the primary source of factual information
- Keep answers professional, accurate, and concise
- Always base your answer on the provided context (both KG and documents)
- **PRIORITY**: Documents > KG (if there's a conflict, trust documents)
- **BEFORE SAYING "NOT FOUND"**: Carefully search through ALL document sections, including:
  * Figure captions (e.g., "Figure 2. Functionalization of CNT by ATR")
  * Text mentions (e.g., "using atom transfer polymerization (ATR)")
  * Method descriptions
- If information is truly insufficient after thorough search, state it clearly
- **DO NOT claim information is missing if it exists in the context**"""
        
        # Build user message
        user_message = f"""Context information:
{context}

Question: {query}

Please answer the question based on the above context."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(
        self, 
        query: str, 
        top_k: int = 10,
        use_kg: bool = True
    ) -> Dict[str, Any]:
        """
        执行知识图谱增强的RAG查询
        
        Args:
            query: 用户问题
            top_k: 向量检索返回的文档数量
            use_kg: 是否使用知识图谱增强
        
        Returns:
            包含答案、上下文和元数据的字典
        """
        # 检测问题类型（用于优化提示词，但不影响KG使用）
        question_type = self._detect_question_type(query)
        
        # ✅ 始终使用知识图谱，不自动禁用
        # 确保 use_kg 参数被尊重，不会因为问题类型而改变
        if self.verbose:
            print(f"\n🔍 查询: {query}")
            print(f"🎯 问题类型: {question_type}")
            print(f"🎛️  配置: top_k={top_k}, use_kg={use_kg} (知识图谱始终启用)")
        
        # 1. 混合检索
        vector_docs, kg_knowledge = self.retriever.hybrid_retrieve(
            query=query,
            top_k=top_k,
            use_kg=use_kg
        )
        
        # ✅ 1.5. 对数值问题应用Reranking
        reranked = False
        if question_type == "numerical" and self.use_reranking and self.reranker:
            try:
                if self.verbose:
                    print(f"🔄 对数值问题应用重排序，从 {len(vector_docs)} 个文档中精选...")
                
                # 准备文档内容
                doc_contents = [doc.get('content', '') for doc in vector_docs]
                
                # 调用Reranker
                reranked_scores = self.reranker.rerank(
                    query=query,
                    documents=doc_contents,
                    top_k=5  # 对数值问题只保留最相关的5个
                )
                
                # 根据Reranker结果重新排序文档
                reranked_docs = []
                for idx, score in reranked_scores:
                    if idx < len(vector_docs):
                        doc = vector_docs[idx].copy()
                        doc['rerank_score'] = score
                        reranked_docs.append(doc)
                
                vector_docs = reranked_docs
                reranked = True
                
                if self.verbose:
                    print(f"✅ 重排序完成，保留 {len(vector_docs)} 个最相关文档")
                    
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  重排序失败: {e}")
                    print(f"   继续使用原始检索结果...")
        
        # 2. 构建增强上下文
        context = self.build_enhanced_context(vector_docs, kg_knowledge)
        
        # 3. 生成答案
        answer = self.generate_answer(
            query=query,
            context=context,
            has_kg_knowledge=(kg_knowledge is not None),
            question_type=question_type  # ✅ 传递问题类型
        )
        
        # 4. 返回结果
        result = {
            'query': query,
            'answer': answer,
            'retrieved_docs': vector_docs,  # 添加实际的文档列表（用于评估）
            'vector_docs_count': len(vector_docs),
            'kg_enhanced': kg_knowledge is not None,
            'kg_knowledge': kg_knowledge,
            'reranked': reranked,  # ✅ 添加reranking标志
            'question_type': question_type,  # ✅ 添加问题类型
            'sources': [doc.get('source', 'Unknown') for doc in vector_docs[:5]],
            'total_time': 0  # 将在外部计算
        }
        
        if self.verbose:
            print(f"\n✅ Answer generated successfully")
            print(f"📊 Used {result['vector_docs_count']} documents")
            print(f"🕸️  KG enhancement: {'Yes' if result['kg_enhanced'] else 'No'}")
            print(f"🔄 Reranked: {'Yes' if reranked else 'No'}")
        
        return result
    
    def close(self):
        """关闭连接"""
        self.retriever.close()

