#!/usr/bin/env python3
"""
RAGPhoto_5 Knowledge Graph Enhanced RAG Query Tool
Interactive command-line interface for KG-enhanced RAG queries
Combines vector search with knowledge graph for better accuracy

Features:
- Interactive query mode
- Single query mode
- Comparison demo mode (KG vs Standard RAG)
- Batch evaluation mode (NEW)
- RAGAS evaluation metrics (NEW)
"""

import sys
import os
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after path setup
from config.config import config, load_env_file
from core.kg_enhanced_rag import KGEnhancedRAGSystem
from core.rag_system import RAGSystem
from core import QuestionLoader

# Try to import RAGAS evaluator
try:
    from evaluation.ragas_evaluation import RAGASEvaluator
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️  RAGAS评估模块未安装，批量评估功能将跳过RAGAS指标")

def print_header(title: str):
    """Print header"""
    print(f"\n{'='*80}")
    print(f"🎯 {title}")
    print(f"{'='*80}")

def print_step(step: int, title: str, description: str = ""):
    """打印步骤"""
    print(f"\n📍 步骤 {step}: {title}")
    if description:
        print(f"   {description}")
    print("-" * 60)

def compare_rag_systems():
    """Compare standard RAG vs KG-enhanced RAG"""
    print_header("KG-Enhanced RAG vs Standard RAG - Comparison Demo")

    print("\n🔧 Initializing KG-Enhanced RAG system...")
    kg_rag = KGEnhancedRAGSystem(verbose=True)
    print("\n🔧 Initializing Standard RAG system...")
    try:
        normal_rag = RAGSystem(verbose=True)
        normal_rag.initialize_vector_store()
        print("✅ Standard RAG system initialized")
    except Exception as e:
        print(f"⚠️  Standard RAG initialization failed: {e}")
        print("    Continuing with KG-Enhanced RAG only...")
        normal_rag = None

    test_queries = [
        "What is the efficiency of perovskite solar cells?",
        "Which materials can be fabricated by spin coating?",
        "What processes improve PCE?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}\n")
        print(f"📝 Test Question {i}/{len(test_queries)}")
        print(f"Question: {query}")
        print(f"\n{'='*80}\n")

        # KG-Enhanced RAG query
        print("\n--- KG-Enhanced RAG System ---")
        kg_result = kg_rag.query(query, top_k=5, use_kg=True)
        print(f"\n💡 Answer:\n{kg_result['answer']}")
        if kg_result.get('kg_knowledge'):
            print(f"\n🕸️  Additional KG Information:\n{kg_result['kg_knowledge']}")
        print(f"\n📚 Documents: {kg_result.get('vector_docs_count', 0)}")
        print(f"⏱️  Time: {kg_result.get('total_time', 0):.2f}s")

        # Standard RAG query
        if normal_rag:
            print("\n--- Standard RAG System ---")
            normal_result = normal_rag.rag_query(query, top_k=5)
            print(f"\n💡 Answer:\n{normal_result['answer']}")
            print(f"\n📚 Documents: {normal_result.get('retrieved_docs_count', 0)}")
            print(f"⏱️  Time: {normal_result.get('query_time', 0):.2f}s")

    print(f"\n{'='*80}\n")
    print("✅ Demo completed")

def interactive_query():
    """Interactive query mode"""
    print_header("KG-Enhanced RAG System - Interactive Mode")
    print("\n🔧 Initializing KG-Enhanced RAG system...")
    kg_rag = KGEnhancedRAGSystem(verbose=True)
    print("\nType 'exit' or 'quit' to exit.")

    while True:
        query = input("\n🔍 Your query (or 'exit'): ").strip()
        if query.lower() in ['exit', 'quit']:
            print("\n👋 Goodbye!")
            break
        if not query:
            continue

        print(f"\n🔍 Query: {query}")
        result = kg_rag.query(query, top_k=config.top_k_results, use_kg=True)

        print(f"\n💡 Answer:\n{result['answer']}")
        if result.get('kg_knowledge'):
            print(f"\n🕸️  Additional KG Information:\n{result['kg_knowledge']}")
        print(f"\n📚 Documents: {result.get('vector_docs_count', 0)}")
        print(f"⏱️  Time: {result.get('total_time', 0):.2f}s")

def single_query(query_text: str):
    """Execute single query"""
    print_header("KG-Enhanced RAG System - Single Query")
    print("\n🔧 Initializing KG-Enhanced RAG system...")
    kg_rag = KGEnhancedRAGSystem(verbose=True)

    print(f"\n🔍 Query: {query_text}")
    print(f"🎛️  Config: top_k={config.top_k_results}, use_kg=True")
    result = kg_rag.query(query_text, top_k=config.top_k_results, use_kg=True)

    print(f"\n{'='*80}")
    print(f"💡 Answer:\n{result['answer']}")
    if result.get('kg_knowledge'):
        print(f"\n🕸️  Additional Knowledge Graph Information:\n{result['kg_knowledge']}")
    print(f"\n{'='*80}")
    print(f"📊 Statistics:")
    print(f"  - Vector documents: {result.get('vector_docs_count', 0)}")
    print(f"  - KG enhancement: {'Yes' if result.get('kg_enhanced') else 'No'}")
    print(f"  - Total time: {result.get('total_time', 0):.2f}s")
    print(f"{'='*80}")

def load_evaluation_questions(sample_size: int = 0):
    """加载评估问题"""
    print_step(1, "加载评估问题", f"从Question.csv加载问题")
    
    try:
        question_loader = QuestionLoader("/data/wzfeng/RAGPhoto/Data/Question.csv")
        
        if sample_size > 0:
            questions = question_loader.get_sample_questions(n=sample_size)
            print(f"✓ 采样加载 {len(questions)} 个问题")
        else:
            questions = question_loader.load_all_questions()
            print(f"✓ 完整加载 {len(questions)} 个问题")
        
        # 显示问题统计
        if questions:
            avg_length = sum(len(q['question']) for q in questions) / len(questions)
            print(f"✓ 平均问题长度: {avg_length:.1f} 字符")
            
            # 统计有ground truth的问题数量
            valid_gt = sum(1 for q in questions if q.get('ground_truth', 'N/A') != 'N/A')
            print(f"✓ 有效Ground Truth: {valid_gt}/{len(questions)} ({valid_gt/len(questions)*100:.1f}%)")
            
            # 显示示例问题
            print(f"\n📝 示例问题:")
            for i, q in enumerate(questions[:3], 1):
                print(f"   {i}. {q['question'][:60]}...")
        
        return questions
        
    except Exception as e:
        print(f"❌ 问题加载失败: {e}")
        return []

def run_kg_rag_evaluation(questions: List[Dict], kg_rag: KGEnhancedRAGSystem):
    """运行KG增强RAG查询评估"""
    print_step(2, "KG-Enhanced RAG 系统评估", f"对 {len(questions)} 个问题进行KG增强查询")
    
    try:
        results = []
        start_time = time.time()
        
        for i, question_data in enumerate(questions, 1):
            question = question_data['question']
            ground_truth = question_data.get('expected_answer', question_data.get('ground_truth', ''))
            
            print(f"\r进度: {i}/{len(questions)} - {question[:40]}...", end='', flush=True)
            
            try:
                # 执行KG增强RAG查询（单独计时）
                query_start = time.time()
                result = kg_rag.query(question, top_k=config.top_k_results, use_kg=True)
                query_elapsed = time.time() - query_start
                
                if result and result.get('answer'):
                    # 提取上下文（向量 + KG）
                    contexts = []
                    
                    # 添加向量检索的上下文
                    vector_docs = result.get('retrieved_docs', [])
                    if not vector_docs:
                        # 如果没有retrieved_docs，尝试从其他字段获取
                        vector_docs = result.get('context_documents', [])
                    
                    for doc in vector_docs[:5]:
                        if isinstance(doc, dict):
                            # 尝试多种字段名
                            text = doc.get('content', '') or doc.get('text', '') or doc.get('chunk_text', '')
                            if text and text.strip():
                                contexts.append(text)
                        elif isinstance(doc, str):
                            # 如果doc是字符串，直接添加
                            if doc.strip():
                                contexts.append(doc)
                    
                    # 添加知识图谱信息作为独立上下文
                    kg_knowledge = result.get('kg_knowledge', '')
                    if kg_knowledge and kg_knowledge.strip():
                        contexts.append(f"Knowledge Graph Information:\n{kg_knowledge}")
                    
                    # 确保contexts不为空（至少有一个上下文）
                    if not contexts:
                        # 如果仍然没有上下文，尝试从result的其他字段获取
                        if result.get('contexts'):
                            contexts = result.get('contexts', [])
                        elif result.get('context'):
                            contexts = [result.get('context', '')]
                        else:
                            # 使用占位符，避免评估失败
                            contexts = ['No context available.']
                    
                    # 构建RAGAS评估所需的数据格式
                    eval_data = {
                        'question': question,
                        'answer': result['answer'],
                        'contexts': contexts,
                        'ground_truth': ground_truth,
                        'expected_answer': question_data.get('expected_answer', ''),
                        'retrieved_docs_count': len(vector_docs),
                        'question_id': question_data.get('question_id', f'Q{i}'),
                        'kg_enhanced': result.get('kg_enhanced', False),
                        'vector_docs_count': result.get('vector_docs_count', len(vector_docs)),
                        'query_time': query_elapsed  # 使用实际计时
                    }
                    results.append(eval_data)
                else:
                    print(f"\n⚠️  问题 {i} 查询失败: {question[:40]}...")
                    
            except Exception as e:
                print(f"\n❌ 问题 {i} 处理错误: {e}")
        
        query_time = time.time() - start_time
        print(f"\n✓ KG增强RAG查询完成，耗时 {query_time:.2f}秒")
        print(f"✓ 成功处理 {len(results)}/{len(questions)} 个问题")
        print(f"✓ 平均耗时 {query_time/len(questions):.2f}秒/问题")
        
        # 数据质量检查
        if results:
            valid_contexts = sum(1 for r in results if r.get('contexts') and len(r['contexts']) > 0)
            valid_answers = sum(1 for r in results if r.get('answer') and len(r['answer'].strip()) > 0)
            valid_gt = sum(1 for r in results if r.get('ground_truth') and len(r['ground_truth'].strip()) > 0)
            kg_enhanced_count = sum(1 for r in results if r.get('kg_enhanced', False))
            
            print(f"✓ 数据质量检查:")
            print(f"   有效上下文: {valid_contexts}/{len(results)} ({valid_contexts/len(results)*100:.1f}%)")
            print(f"   有效答案: {valid_answers}/{len(results)} ({valid_answers/len(results)*100:.1f}%)")
            print(f"   有效Ground Truth: {valid_gt}/{len(results)} ({valid_gt/len(results)*100:.1f}%)")
            print(f"   KG增强查询: {kg_enhanced_count}/{len(results)} ({kg_enhanced_count/len(results)*100:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"\n❌ KG-RAG评估失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_ragas_evaluation(eval_data: List[Dict]):
    """运行RAGAS评估"""
    if not RAGAS_AVAILABLE:
        print("⚠️  RAGAS评估模块不可用，跳过评估")
        return None
    
    print_step(3, "RAGAS指标评估", "计算6个完整的RAGAS指标")
    
    try:
        # 初始化RAGAS评估器
        csv_path = "/data/wzfeng/RAGPhoto/Data/Question.csv"
        evaluator = RAGASEvaluator(questions_csv_path=csv_path)
        
        print(f"✓ RAGAS评估器初始化完成")
        print(f"✓ 准备评估 {len(eval_data)} 个问题")
        
        # 运行评估
        start_time = time.time()
        metrics = evaluator.evaluate_dataset(eval_data)
        eval_time = time.time() - start_time
        
        if metrics:
            print(f"✓ RAGAS评估完成，耗时 {eval_time:.2f}秒")
            print(f"✓ 评估指标数量: {len(metrics)}")
            print(f"✓ 包含指标: {list(metrics.keys())}")
        else:
            print(f"⚠️  RAGAS评估完成但未返回指标")
        
        return metrics
        
    except Exception as e:
        print(f"❌ RAGAS评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_detailed_report(eval_data: List[Dict], metrics: Dict, output_file: str):
    """生成详细的评估报告"""
    print_step(4, "生成详细评估报告", f"保存到 {output_file}")
    
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✓ 创建输出目录: {output_dir}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAGPhoto_5 KG-Enhanced RAG 评估报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评估问题数: {len(eval_data)}\n")
            f.write(f"查询方式: 知识图谱增强RAG (向量检索 + 知识图谱)\n")
            f.write(f"评估框架: RAGAS (6个核心指标)\n")
            
            # 输出配置信息
            f.write(f"\n配置信息:\n")
            f.write(f"  - LLM模型: {config.llm_model}\n")
            f.write(f"  - 嵌入模型: {config.embedding_model}\n")
            f.write(f"  - 检索数量: top_k={config.top_k_results}\n")
            f.write(f"  - KG增强: 启用\n")
            f.write("\n")
            
            # 输出RAGAS完整评估指标
            if metrics:
                f.write("=" * 80 + "\n")
                f.write("RAGAS 完整评估指标 (6个核心指标)\n")
                f.write("=" * 80 + "\n")
                
                # 指标详细说明
                metric_details = {
                    'faithfulness': {
                        'name': '忠实性 (Faithfulness)',
                        'desc': '回答是否忠实基于检索到的文档内容',
                        'range': '0-1，越高越好'
                    },
                    'answer_relevancy': {
                        'name': '答案相关性 (Answer Relevancy)', 
                        'desc': '回答与用户问题的匹配程度',
                        'range': '0-1，越高越好'
                    },
                    'context_precision': {
                        'name': '上下文精确度 (Context Precision)',
                        'desc': '检索文档中有用信息的比例',
                        'range': '0-1，越高越好'
                    },
                    'context_recall': {
                        'name': '上下文召回率 (Context Recall)',
                        'desc': '检索文档包含答案所需信息的完整性',
                        'range': '0-1，越高越好'
                    },
                    'answer_correctness': {
                        'name': '答案正确性 (Answer Correctness)',
                        'desc': '答案的事实准确性和完整性',
                        'range': '0-1，越高越好'
                    },
                    'answer_similarity': {
                        'name': '答案相似性 (Answer Similarity)',
                        'desc': '生成答案与参考答案的语义相似度',
                        'range': '0-1，越高越好'
                    }
                }
                
                total_score = 0
                metric_count = 0
                
                for metric, value in metrics.items():
                    if metric in metric_details:
                        detail = metric_details[metric]
                        score_level = "优秀" if value >= 0.8 else "良好" if value >= 0.6 else "需改进"
                        score_emoji = "🟢" if value >= 0.8 else "🟡" if value >= 0.6 else "🔴"
                        
                        f.write(f"{score_emoji} {detail['name']}: {value:.4f} ({score_level})\n")
                        f.write(f"   说明: {detail['desc']}\n")
                        f.write(f"   范围: {detail['range']}\n\n")
                        
                        total_score += value
                        metric_count += 1
                
                if metric_count > 0:
                    avg_score = total_score / metric_count
                    avg_emoji = "🟢" if avg_score >= 0.8 else "🟡" if avg_score >= 0.6 else "🔴"
                    f.write(f"{avg_emoji} 整体评估得分: {avg_score:.4f}\n")
                    
                    if avg_score >= 0.8:
                        f.write(f"🎉 系统评价: 优秀 - KG增强RAG系统表现出色\n")
                    elif avg_score >= 0.6:
                        f.write(f"👍 系统评价: 良好 - KG增强RAG效果不错，有优化空间\n")
                    else:
                        f.write(f"⚠️ 系统评价: 需优化 - 建议调整检索策略或KG配置\n")
                f.write("\n")
            
            # 输出每个问题的详细结果
            f.write("=" * 80 + "\n")
            f.write("问题详细评估结果\n")
            f.write("=" * 80 + "\n")
            
            for i, data in enumerate(eval_data, 1):
                f.write("=" * 60 + "\n")
                f.write(f"问题 {i}:\n")
                f.write("=" * 60 + "\n")
                
                question = data['question']
                question_id = data.get('question_id', f'Q{i}')
                
                # 输出问题信息
                f.write(f"🆔 问题ID: {question_id}\n")
                f.write(f"🎯 完整问题:\n")
                f.write(f"Q: {question}\n\n")
                
                # 输出期望答案
                expected_answer = data.get('ground_truth', '')
                if expected_answer and expected_answer != 'N/A':
                    f.write(f"📖 期望答案:\n{expected_answer}\n")
                else:
                    f.write(f"📖 期望答案: N/A\n")
                
                # 输出KG增强RAG答案
                f.write(f"\n🤖 KG增强RAG答案:\n")
                f.write(f"{data.get('answer', 'N/A')}\n")
                
                # 输出KG增强状态
                kg_enhanced = data.get('kg_enhanced', False)
                f.write(f"\n🕸️  KG增强: {'是' if kg_enhanced else '否'}\n")
                
                # 输出上下文信息
                contexts = data.get('contexts', [])
                f.write(f"\n📊 检索信息:\n")
                f.write(f"上下文数量: {len(contexts)}\n")
                f.write(f"向量文档数: {data.get('vector_docs_count', 0)}\n")
                f.write(f"查询耗时: {data.get('query_time', 0):.2f}秒\n")
                
                if contexts:
                    f.write(f"\n📄 检索到的上下文片段:\n")
                    for j, context in enumerate(contexts[:3], 1):
                        preview = context[:200] + "..." if len(context) > 200 else context
                        f.write(f"  {j}. {preview}\n")
                    if len(contexts) > 3:
                        f.write(f"  ... 还有 {len(contexts) - 3} 个上下文片段\n")
                else:
                    f.write("⚠️ 未检索到有效上下文\n")
                
                f.write("\n")
            
            # 添加总结信息
            f.write("=" * 80 + "\n")
            f.write("KG增强RAG评估总结\n")
            f.write("=" * 80 + "\n")
            
            total_questions = len(eval_data)
            questions_with_context = sum(1 for r in eval_data if r.get('contexts', []))
            avg_context_count = sum(len(r.get('contexts', [])) for r in eval_data) / total_questions if total_questions > 0 else 0
            kg_enhanced_count = sum(1 for r in eval_data if r.get('kg_enhanced', False))
            avg_query_time = sum(r.get('query_time', 0) for r in eval_data) / total_questions if total_questions > 0 else 0
            
            f.write(f"📊 查询统计:\n")
            f.write(f"  总问题数: {total_questions}\n")
            f.write(f"  有上下文的问题数: {questions_with_context}\n")
            f.write(f"  平均上下文数量: {avg_context_count:.2f}\n")
            f.write(f"  上下文覆盖率: {questions_with_context/total_questions*100:.1f}%\n")
            f.write(f"  KG增强查询数: {kg_enhanced_count}\n")
            f.write(f"  KG增强率: {kg_enhanced_count/total_questions*100:.1f}%\n")
            f.write(f"  平均查询耗时: {avg_query_time:.2f}秒\n")
            
            if metrics and metric_count > 0:
                f.write(f"\n🎯 系统性能评价:\n")
                f.write(f"  评估指标数: {len(metrics)}个 (RAGAS完整指标)\n")
                avg_score = total_score / metric_count
                performance_level = "优秀" if avg_score >= 0.8 else "良好" if avg_score >= 0.6 else "需要优化"
                f.write(f"  整体得分: {avg_score:.4f} ({performance_level})\n")
                f.write(f"  KG增强RAG效果: {'显著' if kg_enhanced_count/total_questions >= 0.8 else '良好' if kg_enhanced_count/total_questions >= 0.5 else '有待提升'}\n")
        
        print(f"✓ 详细报告已保存到: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_metrics_summary(eval_data: List[Dict], metrics: Dict, output_file: str):
    """生成RAGAS指标汇总报告（仅包含6个核心指标）"""
    try:
        import os
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RAGPhoto_5 KG-Enhanced RAG RAGAS 评估指标汇总\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评估问题数: {len(eval_data)}\n")
            f.write(f"评估系统: KG增强RAG (向量检索 + 知识图谱)\n")
            f.write(f"评估框架: RAGAS\n\n")
            
            # RAGAS 6个核心指标汇总
            if metrics:
                f.write("=" * 80 + "\n")
                f.write("📊 RAGAS 6个核心指标 - 整体评分\n")
                f.write("=" * 80 + "\n\n")
                
                metric_details = {
                    'faithfulness': ('忠实性', '回答是否忠实基于检索文档'),
                    'answer_relevancy': ('答案相关性', '回答与问题的匹配度'),
                    'context_precision': ('上下文精确度', '检索文档的准确性'),
                    'context_recall': ('上下文召回率', '检索的完整性'),
                    'answer_correctness': ('答案正确性', '答案的事实准确性'),
                    'answer_similarity': ('答案相似性', '与标准答案的相似度')
                }
                
                total_score = 0
                metric_count = 0
                
                for metric, value in metrics.items():
                    if metric in metric_details:
                        name, desc = metric_details[metric]
                        score_emoji = "🟢" if value >= 0.8 else "🟡" if value >= 0.6 else "🔴"
                        score_level = "优秀" if value >= 0.8 else "良好" if value >= 0.6 else "需改进"
                        
                        f.write(f"{score_emoji} {name} ({metric}): {value:.4f} - {score_level}\n")
                        f.write(f"   说明: {desc}\n\n")
                        
                        total_score += value
                        metric_count += 1
                
                if metric_count > 0:
                    avg_score = total_score / metric_count
                    avg_emoji = "🟢" if avg_score >= 0.8 else "🟡" if avg_score >= 0.6 else "🔴"
                    f.write(f"\n{'='*80}\n")
                    f.write(f"{avg_emoji} 整体平均得分: {avg_score:.4f}\n")
                    f.write(f"{'='*80}\n\n")
                
                # KG增强统计
                kg_enhanced_count = sum(1 for r in eval_data if r.get('kg_enhanced', False))
                avg_query_time = sum(r.get('query_time', 0) for r in eval_data) / len(eval_data) if eval_data else 0
                
                f.write("=" * 80 + "\n")
                f.write("🕸️  KG增强效果统计\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"KG增强查询数: {kg_enhanced_count}/{len(eval_data)}\n")
                f.write(f"KG增强率: {kg_enhanced_count/len(eval_data)*100:.1f}%\n")
                f.write(f"平均查询耗时: {avg_query_time:.2f}秒\n\n")
                
                # 每个问题的指标评分
                f.write("=" * 80 + "\n")
                f.write("📋 每个问题的RAGAS指标评分\n")
                f.write("=" * 80 + "\n\n")
                
                for i, data in enumerate(eval_data, 1):
                    question_id = data.get('question_id', f'Q{i}')
                    question = data['question']
                    question_preview = question[:50] + "..." if len(question) > 50 else question
                    kg_enhanced = "✅" if data.get('kg_enhanced', False) else "❌"
                    
                    f.write(f"问题 {i} (ID: {question_id}) [KG增强: {kg_enhanced}]:\n")
                    f.write(f"Q: {question_preview}\n")
                    f.write(f"{'-'*60}\n")
                    
                    for metric in metric_details:
                        if metric in metrics:
                            f.write(f"  {metric}: {metrics[metric]:.4f}\n")
                    f.write(f"  查询耗时: {data.get('query_time', 0):.2f}秒\n")
                    f.write(f"\n")
                
                f.write("=" * 80 + "\n")
                f.write("评估完成\n")
                f.write("=" * 80 + "\n")
            else:
                f.write("⚠️  未进行RAGAS评估\n")
        
        print(f"✓ 指标汇总已保存到: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ 指标汇总生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_summary(metrics: Dict, eval_count: int, total_time: float):
    """打印评估总结"""
    print_header("评估总结")
    
    print(f"📋 评估统计:")
    print(f"   评估问题数: {eval_count}")
    print(f"   总耗时: {total_time:.2f}秒")
    print(f"   平均耗时: {total_time/eval_count:.2f}秒/问题")
    
    if metrics:
        print(f"\n📊 RAGAS指标结果:")
        
        # 指标解释
        explanations = {
            'faithfulness': '忠实性 - 回答是否基于检索文档',
            'answer_relevancy': '答案相关性 - 回答与问题的匹配度',
            'context_precision': '上下文精确度 - 检索文档的准确性',
            'context_recall': '上下文召回率 - 检索的完整性',
            'answer_correctness': '答案正确性 - 答案的事实准确性',
            'answer_similarity': '答案相似性 - 与标准答案的相似度'
        }
        
        for metric, value in metrics.items():
            if metric in explanations:
                score_emoji = "🟢" if value >= 0.8 else "🟡" if value >= 0.6 else "🔴"
                print(f"   {score_emoji} {explanations[metric]}: {value:.3f}")
        
        avg_score = sum(metrics.values()) / len(metrics)
        avg_emoji = "🟢" if avg_score >= 0.8 else "🟡" if avg_score >= 0.6 else "🔴"
        print(f"\n   {avg_emoji} 平均得分: {avg_score:.3f}")
        
        # 性能评价
        if avg_score >= 0.8:
            print(f"\n🎉 评估结果：优秀！KG增强RAG系统性能良好")
        elif avg_score >= 0.6:
            print(f"\n👍 评估结果：良好，有改进空间")
        else:
            print(f"\n⚠️  评估结果：需要优化")
    
    print(f"\n📋 下一步建议:")
    print("   1. 查看详细报告文件")
    print("   2. 运行更大样本评估: python run_kg_enhanced_query.py --evaluate --sample-size 50")
    print("   3. 对比标准RAG: python run_kg_enhanced_query.py --demo")
    print("   4. 运行交互查询: python run_kg_enhanced_query.py --interactive")

def batch_evaluation(sample_size: int = 10, output_file: str = None, skip_ragas: bool = False):
    """批量评估模式"""
    print_header("KG-Enhanced RAG 批量评估")
    print(f"🎯 评估模式: {'仅查询测试' if skip_ragas else 'RAGAS完整评估'}")
    print(f"📝 样本数量: {'全部' if sample_size == 0 else sample_size}")
    print(f"🕸️  KG增强: 启用")
    print(f"📊 评估指标: {'基础功能' if skip_ragas else '6个RAGAS完整指标'}")
    
    total_start_time = time.time()
    
    try:
        # 加载评估问题
        questions = load_evaluation_questions(sample_size)
        if not questions:
            print("❌ 无法加载评估问题")
            return
        
        # 初始化KG增强RAG系统
        print_step(2, "初始化KG增强RAG系统")
        kg_rag = KGEnhancedRAGSystem(verbose=False)
        print("✓ KG增强RAG系统初始化完成")
        
        # 运行KG-RAG评估
        eval_data = run_kg_rag_evaluation(questions, kg_rag)
        if not eval_data:
            print("❌ KG-RAG评估失败")
            return
        
        # 运行RAGAS评估（可选）
        metrics = None
        if not skip_ragas and RAGAS_AVAILABLE:
            metrics = run_ragas_evaluation(eval_data)
            if metrics:
                print(f"✓ 完成RAGAS 6个核心指标评估")
                print(f"✓ 评估指标: {list(metrics.keys())}")
        elif skip_ragas:
            print("⚠️  跳过RAGAS评估（仅测试查询功能）")
        elif not RAGAS_AVAILABLE:
            print("⚠️  RAGAS模块不可用，跳过评估")
        
        # 生成两个报告文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. RAGAS指标汇总报告
        metrics_file = f"outputs/evaluations/kg_enhanced_metrics_summary_{timestamp}.txt"
        metrics_success = generate_metrics_summary(eval_data, metrics, metrics_file)
        
        # 2. 完整详细报告
        if output_file is None:
            detailed_file = f"outputs/evaluations/kg_enhanced_detailed_{timestamp}.txt"
        else:
            detailed_file = output_file
        detailed_success = generate_detailed_report(eval_data, metrics, detailed_file)
        
        if metrics_success or detailed_success:
            total_time = time.time() - total_start_time
            print_summary(metrics, len(eval_data), total_time)
            
            print(f"\n📄 生成的报告文件:")
            if metrics_success:
                print(f"  ✅ 指标汇总: {metrics_file}")
            if detailed_success:
                print(f"  ✅ 详细报告: {detailed_file}")
        
    except KeyboardInterrupt:
        print("\n\n👋 评估被用户中断")
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    load_env_file()
    parser = argparse.ArgumentParser(
        description="RAGPhoto_5 Knowledge Graph Enhanced RAG Query Tool",
        epilog="""
Examples:
  # Interactive query mode
  python run_kg_enhanced_query.py --interactive
  
  # Single query
  python run_kg_enhanced_query.py "What is photovoltaic efficiency?"
  
  # Comparison demo
  python run_kg_enhanced_query.py --demo
  
  # Batch evaluation (NEW)
  python run_kg_enhanced_query.py --evaluate --sample-size 10
  
  # Batch evaluation without RAGAS
  python run_kg_enhanced_query.py --evaluate --sample-size 5 --skip-ragas
  
  # Full evaluation
  python run_kg_enhanced_query.py --evaluate --sample-size 0
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 原有参数
    parser.add_argument('query', nargs='?', help='Query string for single query')
    parser.add_argument('--demo', action='store_true', help='Run comparison demo mode')
    parser.add_argument('--interactive', action='store_true', help='Run interactive query mode')
    
    # 新增评估参数
    parser.add_argument('--evaluate', action='store_true', help='Run batch evaluation mode (NEW)')
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=10,
        help='Number of questions for evaluation (default: 10, 0 for all)'
    )
    parser.add_argument(
        '--output', 
        type=str,
        default=None,
        help='Output file path for evaluation report'
    )
    parser.add_argument(
        '--skip-ragas', 
        action='store_true',
        help='Skip RAGAS evaluation (only test query functionality)'
    )
    
    args = parser.parse_args()

    if args.evaluate:
        # 批量评估模式（新功能）
        batch_evaluation(
            sample_size=args.sample_size,
            output_file=args.output,
            skip_ragas=args.skip_ragas
        )
    elif args.demo:
        # 对比演示模式
        compare_rag_systems()
    elif args.interactive:
        # 交互式查询模式
        interactive_query()
    elif args.query:
        # 单次查询模式
        single_query(args.query)
    else:
        print("Please provide a query string, or use one of the available modes.")
        print("\n📖 Available Modes:")
        print("  --interactive   : Interactive query mode")
        print("  --demo          : Comparison demo (KG vs Standard RAG)")
        print("  --evaluate      : Batch evaluation with RAGAS metrics (NEW)")
        print("\n📝 Examples:")
        print('  python run_kg_enhanced_query.py "What is photovoltaic efficiency?"')
        print('  python run_kg_enhanced_query.py --interactive')
        print('  python run_kg_enhanced_query.py --demo')
        print('  python run_kg_enhanced_query.py --evaluate --sample-size 10')
        print("\n💡 Use --help for more information")
        parser.print_help()

if __name__ == "__main__":
    main()
