#!/usr/bin/env python3
"""
RAGPhoto_5 Standard RAG Query Tool
Interactive command-line interface for standard RAG queries (vector search only)
"""

import sys
import os
import time
from typing import Dict, Any, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import config, load_env_file
from core import RAGSystem

def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    RAGPhoto_5 智能查询系统                     ║
║                   光电科研文献专业问答助手                      ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_header(title: str):
    """Print header"""
    print(f"\n{'='*80}")
    print(f"🎯 {title}")
    print(f"{'='*80}")

def print_help():
    """打印帮助信息"""
    help_text = """
💡 使用指南:
────────────────────────────────────────
🔍 查询方式:
   • 直接输入问题进行查询
   • 输入 'help' 显示此帮助
   • 输入 'examples' 查看示例问题
   • 输入 'stats' 查看系统状态
   • 输入 'config' 查看配置信息
   • 输入 'exit' 或 'quit' 退出

🎯 查询技巧:
   • 使用专业术语获得更准确的结果
   • 可以询问效率、制备、表征等方面的问题
   • 支持中英文混合查询

📝 示例问题:
   • What is the efficiency of perovskite solar cells?
   • How are solar cell devices fabricated?
   • What characterization methods are used?
   • 钙钛矿太阳能电池的效率是多少？
────────────────────────────────────────
"""
    print(help_text)

def load_example_questions() -> List[str]:
    """加载示例问题"""
    try:
        question_loader = QuestionLoader("/home/wzfeng/RAGPhoto/Data/Question.csv")
        questions = question_loader.get_sample_questions(n=10)
        return [q['question'] for q in questions]
    except:
        # 如果加载失败，返回预设问题
        return [
            "What is the efficiency of perovskite solar cells?",
            "How are solar cell devices fabricated?", 
            "What characterization methods are used in solar cell research?",
            "What factors affect solar cell performance?",
            "How do you measure solar cell efficiency?",
            "What materials are used in perovskite solar cells?",
            "What are the challenges in solar cell development?",
            "How does temperature affect solar cell performance?",
            "What is the fill factor in solar cells?",
            "How do you improve solar cell stability?"
        ]

def show_examples():
    """显示示例问题"""
    print("\n📝 示例问题:")
    print("─" * 50)
    
    examples = load_example_questions()
    for i, question in enumerate(examples, 1):
        print(f"{i:2d}. {question}")
    
    print("\n💡 提示: 输入问题编号可以直接使用该问题查询")
    print("─" * 50)

def show_system_stats(rag_system: RAGSystem):
    """显示系统状态"""
    print("\n📊 系统状态:")
    print("─" * 50)
    
    try:
        # 检查向量数据状态
        is_built = rag_system.vector_manager.is_vector_data_built()
        print(f"向量数据状态: {'✅ 已构建' if is_built else '❌ 未构建'}")
        
        if is_built:
            # 读取元数据
            try:
                if os.path.exists(rag_system.vector_manager.metadata_file):
                    import json
                    with open(rag_system.vector_manager.metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    print(f"文档数量: {metadata.get('total_documents', 'N/A')}")
                    print(f"分块数量: {metadata.get('total_chunks', 'N/A')}")
                    print(f"构建时间: {metadata.get('build_time', 'N/A')[:19] if metadata.get('build_time') else 'N/A'}")
                    print(f"分块模式: {metadata.get('splitting_mode', 'N/A')}")
            except:
                print("元数据读取失败")
        
        # 检查配置
        print(f"LLM模型: {config.llm_model}")
        print(f"嵌入模型: {config.embedding_model}")
        print(f"重排序: {'✅ 启用' if config.enable_reranking else '❌ 禁用'}")
        print(f"检索数量: {config.top_k_retrieval}")
        print(f"最终数量: {config.top_k_results}")
        
    except Exception as e:
        print(f"状态获取失败: {e}")
    
    print("─" * 50)

def show_config():
    """显示配置信息"""
    print("\n⚙️ 系统配置:")
    print("─" * 50)
    
    configs = [
        ("文档路径", config.test_mds_path),
        ("向量维度", config.vector_dimension),
        ("集合名称", config.collection_name),
        ("LLM模型", config.llm_model),
        ("嵌入模型", config.embedding_model),
        ("检索数量", config.top_k_retrieval),
        ("最终数量", config.top_k_results),
        ("重排序", "启用" if config.enable_reranking else "禁用"),
        ("详细输出", "启用" if config.enable_verbose else "禁用")
    ]
    
    for name, value in configs:
        print(f"{name}: {value}")
    
    print("─" * 50)

def format_answer(result: Dict[str, Any]) -> str:
    """格式化查询结果"""
    if not result or not result.get('answer'):
        return "❌ 查询失败，未获得有效回答"
    
    answer = result['answer']
    docs = result.get('retrieved_docs', result.get('retrieved_documents', []))
    
    formatted = f"💡 回答:\n{answer}\n"
    
    if docs:
        formatted += f"\n📚 检索到的文档 ({len(docs)} 个):\n"
        for i, doc in enumerate(docs[:5], 1):  # 显示前5个文档
            content = doc.get('content', '')
            content_preview = content[:300].replace('\n', ' ').strip()  # 增加到300字符
            score = doc.get('score', 0)
            metadata = doc.get('metadata', {})
            section_name = metadata.get('section_name', '未知章节')
            source = metadata.get('source', '未知来源')
            formatted += f"\n   {i}. 📖 {section_name} (相似度: {score:.3f})\n"
            formatted += f"      来源: {source}\n"
            formatted += f"      内容: {content_preview}...\n"
    
    return formatted

def interactive_query():
    """交互式查询主循环"""
    print_banner()
    print("🚀 正在初始化RAG系统...")
    
    try:
        # 初始化RAG系统
        rag = RAGSystem(verbose=False)
        rag.initialize_vector_store()
        print("✅ RAG系统初始化完成\n")
        
        # 检查向量数据
        if not rag.vector_manager.is_vector_data_built():
            print("⚠️  向量数据未构建！")
            print("💡 请先运行: python run_pipeline.py --vector")
            print("   或使用检查脚本: python check_vector_db.py\n")
            return
        
        print_help()
        
        # 加载示例问题用于快速选择
        examples = load_example_questions()
        query_count = 0
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n🔍 请输入您的问题 (或输入help查看帮助): ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 感谢使用RAGPhoto_5智能查询系统！")
                    break
                
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                
                elif user_input.lower() == 'examples':
                    show_examples()
                    continue
                
                elif user_input.lower() == 'stats':
                    show_system_stats(rag)
                    continue
                
                elif user_input.lower() == 'config':
                    show_config()
                    continue
                
                # 检查是否是示例问题编号
                if user_input.isdigit():
                    question_idx = int(user_input) - 1
                    if 0 <= question_idx < len(examples):
                        query = examples[question_idx]
                        print(f"📝 选择问题: {query}")
                    else:
                        print(f"❌ 无效的问题编号，请输入1-{len(examples)}之间的数字")
                        continue
                else:
                    query = user_input
                
                # 执行查询
                print("\n🔄 正在查询中...")
                start_time = time.time()
                
                try:
                    result = rag.rag_query(query, top_k=config.top_k_results)
                    query_time = time.time() - start_time
                    query_count += 1
                    
                    print(f"\n{'='*60}")
                    print(f"📋 查询结果 #{query_count}")
                    print(f"{'='*60}")
                    print(f"❓ 问题: {query}")
                    print(f"⏱️  耗时: {query_time:.2f}秒")
                    print("─" * 60)
                    
                    # 显示格式化的回答
                    formatted_answer = format_answer(result)
                    print(formatted_answer)
                    
                except Exception as e:
                    print(f"\n❌ 查询失败: {e}")
                
            except KeyboardInterrupt:
                print("\n\n👋 用户中断查询")
                break
            except EOFError:
                print("\n\n👋 退出系统")
                break
    
    except Exception as e:
        print(f"\n❌ 系统初始化失败: {e}")
        print("\n🔧 故障排除:")
        print("   1. 检查.env配置文件")
        print("   2. 确保Milvus服务运行: docker compose -f config/docker-compose.yml up -d")
        print("   3. 检查向量数据: python check_vector_db.py")

def batch_query():
    """批量查询模式"""
    print("📝 批量查询模式")
    print("─" * 50)
    
    try:
        # 初始化RAG系统
        rag = RAGSystem(verbose=False)
        rag.initialize_vector_store()
        
        # 加载示例问题
        examples = load_example_questions()
        
        print(f"🔍 将对 {len(examples)} 个问题进行批量查询")
        choice = input("是否继续? (y/N): ").strip().lower()
        
        if choice != 'y':
            return
        
        print("\n🚀 开始批量查询...")
        start_time = time.time()
        
        results = []
        for i, question in enumerate(examples, 1):
            print(f"\n进度: {i}/{len(examples)} - {question[:50]}...")
            
            try:
                result = rag.rag_query(question, top_k=3)
                results.append({
                    'question': question,
                    'answer': result.get('answer', ''),
                    'success': bool(result.get('answer'))
                })
            except Exception as e:
                results.append({
                    'question': question,
                    'answer': f'错误: {e}',
                    'success': False
                })
        
        # 统计结果
        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r['success'])
        
        print(f"\n📊 批量查询完成:")
        print(f"   总问题数: {len(results)}")
        print(f"   成功查询: {success_count}")
        print(f"   成功率: {success_count/len(results)*100:.1f}%")
        print(f"   总耗时: {total_time:.2f}秒")
        print(f"   平均耗时: {total_time/len(results):.2f}秒/问题")
        
        # 保存结果
        import os
        os.makedirs("outputs/queries", exist_ok=True)
        output_file = f"outputs/queries/batch_query_results_{int(time.time())}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RAGPhoto_5 批量查询结果\n")
            f.write("=" * 60 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"问题 {i}: {result['question']}\n")
                f.write(f"回答: {result['answer']}\n")
                f.write(f"状态: {'成功' if result['success'] else '失败'}\n")
                f.write("-" * 60 + "\n\n")
        
        print(f"✅ 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 批量查询失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAGPhoto_5 交互式查询系统")
    parser.add_argument("query", nargs="?", help="查询问题（可选，不提供则进入交互模式）")
    parser.add_argument("--batch", action="store_true", help="批量查询模式")
    
    args = parser.parse_args()
    
    # 加载配置
    load_env_file('.env')
    
    if args.batch:
        batch_query()
    elif args.query:
        # Single query mode
        print_header("RAGPhoto_5 Standard RAG Query")
        try:
            print("\n🔧 Initializing system...")
            rag = RAGSystem()
            rag.initialize_vector_store()
            print("✅ System initialized\n")
            
            print(f"🔍 Query: {args.query}")
            print("="*80)
            
            result = rag.rag_query(args.query)
            
            if result.get('answer'):
                print(f"\n💡 Answer:\n{result['answer']}")
                print("\n" + "="*80)
                # 获取检索到的文档数量
                docs = result.get('retrieved_docs', [])
                docs_count = len(docs) if isinstance(docs, list) else result.get('retrieved_docs_count', 0)
                print(f"📚 Retrieved documents: {docs_count}")
                print(f"⏱️  Time: {result.get('total_time', result.get('query_time', 0)):.2f} seconds")
                
                # 显示检索到的文档详情（用于诊断）
                if docs:
                    print(f"\n📄 检索到的文档详情:")
                    print("="*80)
                    for i, doc in enumerate(docs[:5], 1):
                        metadata = doc.get('metadata', {})
                        source = metadata.get('source', '未知来源')
                        section = metadata.get('section_name', '未知章节')
                        score = doc.get('score', 0)
                        content = doc.get('content', '')
                        preview = content[:200].replace('\n', ' ').strip()
                        print(f"\n文档 {i}:")
                        print(f"  来源: {source}")
                        print(f"  章节: {section}")
                        print(f"  相似度: {score:.4f}")
                        print(f"  内容预览: {preview}...")
            else:
                print("❌ Query failed")
        except Exception as e:
            print(f"❌ Query error: {e}")
            print("\n💡 Tip: Run 'python run_pipeline.py --vector' to build the vector database first")
    else:
        interactive_query()

if __name__ == "__main__":
    main() 