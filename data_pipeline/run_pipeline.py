#!/usr/bin/env python3
"""
RAGPhoto_5 完整数据处理流水线
从文档读取 → 章节分块 → 向量化 → 构建数据库 的完整流程
"""

import sys
import os
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import config, load_env_file
from core import (
    DocumentLoader,
    PhotovoltaicTextSplitter,
    EmbeddingGenerator,
    VectorManager,
    RAGSystem,
    PhotovoltaicKnowledgeGraph,
    Neo4jManager
)
# from utils.milvus_compose_manager import DockerComposeMilvusManager  # 模块不存在
import subprocess

def print_header(title: str):
    """打印标题"""
    print(f"\n{'='*80}")
    print(f"🎯 {title}")
    print(f"{'='*80}")

def print_step(step: int, title: str, description: str = ""):
    """打印步骤"""
    print(f"\n📍 步骤 {step}: {title}")
    if description:
        print(f"   {description}")
    print("-" * 60)

def check_prerequisites():
    """检查运行前提条件"""
    print_step(1, "检查系统环境")
    
    # 检查配置文件
    if not os.path.exists('.env'):
        print("❌ .env配置文件不存在")
        print("💡 请创建.env文件，参考config/env_example.txt")
        return False
    
    # 加载配置
    load_env_file('.env')
    print("✓ 配置文件加载成功")
    
    # 检查文档目录
    if not os.path.exists(config.test_mds_path):
        print(f"❌ 文档目录不存在: {config.test_mds_path}")
        return False
    
    print(f"✓ 文档目录存在: {config.test_mds_path}")
    
    # 检查Milvus服务
    try:
        # 使用 docker ps 检查服务
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, check=False)
        if 'milvus' in result.stdout.lower():
            print("✓ Milvus服务运行正常")
        else:
            print("⚠️  Milvus服务未运行")
            print("💡 请手动启动 Milvus 服务:")
            print("   方式1: cd config && docker-compose up -d")
            print("   方式2: cd milvus_standalone && docker-compose up -d")
            # 不强制要求，继续执行
            print("   提示: 继续执行，假设服务将自动启动...")
    except Exception as e:
        print(f"⚠️  无法检查Milvus服务状态: {e}")
        print("   继续执行，假设服务已运行...")
    
    return True

def load_documents() -> List[Dict[str, Any]]:
    """加载文档"""
    print_step(2, "加载学术文档", f"从 {config.test_mds_path} 加载文档")
    
    try:
        # 初始化文档加载器
        loader = DocumentLoader(config.test_mds_path)
        
        # 加载所有文档
        start_time = time.time()
        documents = loader.load_all_documents()
        load_time = time.time() - start_time
        
        if not documents:
            print("❌ 没有找到任何文档")
            return []
        
        print(f"✓ 成功加载 {len(documents)} 个文档")
        print(f"✓ 加载耗时: {load_time:.2f}秒")
        
        # 显示文档统计
        total_chars = sum(len(doc['content']) for doc in documents)
        avg_size = total_chars / len(documents) if documents else 0
        
        print(f"📊 文档统计:")
        print(f"   总字符数: {total_chars:,}")
        print(f"   平均大小: {avg_size:.0f} 字符/文档")
        
        # 显示文档列表
        print(f"\n📄 文档列表:")
        for i, doc in enumerate(documents[:10], 1):  # 最多显示10个
            filename = doc['metadata'].get('filename', f'文档{i}')
            size = len(doc['content'])
            encoding = doc['metadata'].get('encoding', 'unknown')
            print(f"   {i:2d}. {filename} ({size:,} 字符, {encoding})")
        
        if len(documents) > 10:
            print(f"   ... 还有 {len(documents) - 10} 个文档")
        
        return documents
        
    except Exception as e:
        print(f"❌ 文档加载失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def split_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """智能分层分块处理"""
    print_step(3, "智能分层分块", "使用光电文献智能分层分块器")
    
    try:
        # 初始化分块器
        splitter = PhotovoltaicTextSplitter()
        print("✓ 智能分层分块器初始化完成")
        print("🎯 分块策略: 章节识别 + 章节内分块")
        
        # 执行分块
        start_time = time.time()
        chunks = splitter.split_documents(documents)
        split_time = time.time() - start_time
        
        if not chunks:
            print("❌ 文档分块失败")
            return []
        
        print(f"✓ 智能分层分块完成，耗时 {split_time:.2f}秒")
        print(f"✓ 生成 {len(chunks)} 个智能分块")
        
        # 显示分块统计
        stats = splitter.get_stats(chunks)
        print(f"\n📊 分块统计:")
        print(f"   总分块数: {stats['total_chunks']}")
        print(f"   总字符数: {stats['total_characters']:,}")
        print(f"   平均大小: {stats['avg_chunk_size']:.1f} 字符")
        print(f"   平均Token: {stats['avg_tokens_per_chunk']:.1f}")
        print(f"   分块模式: {stats['splitting_mode']}")
        print(f"   语义评分: {stats['avg_semantic_score']:.3f}")
        print(f"   完整章节数: {stats['complete_sections']}")
        
        # 显示章节分布
        print(f"\n📈 章节类型分布:")
        for section_type, count in stats['section_distribution'].items():
            section_info = splitter.section_info.get(section_type, splitter.section_info['default'])
            section_name = section_info.name
            print(f"   📖 {section_name}: {count} 个分块")
        
        # 显示质量验证
        print(f"\n✅ 质量验证:")
        print(f"   使用的分块大小: {stats['chunk_size_used']} 字符")
        print(f"   使用的重叠大小: {stats['chunk_overlap_used']} 字符")
        print(f"   平均语义评分: {stats['avg_semantic_score']:.3f}/1.0")
        
        return chunks
        
    except Exception as e:
        print(f"❌ 智能分层分块失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def build_knowledge_graph(documents: List[Dict[str, Any]]) -> bool:
    """构建知识图谱"""
    print_step(6, "构建知识图谱", "提取实体和关系并存储到Neo4j")
    
    try:
        # 初始化Neo4j管理器并清空数据
        neo4j_manager = Neo4jManager()
        print("🧹 正在清空旧的知识图谱数据...")
        neo4j_manager.clear_database()
        print("✓ 旧数据已清空")

        # 初始化知识图谱构建器
        kg_builder = PhotovoltaicKnowledgeGraph(
            use_traceable_extractor=True,
            enable_traceability=True,
            use_enhanced_extractor=True
        )
        
        # 从文档构建知识图谱
        stats = kg_builder.build_kg_from_documents(documents)
        
        # 将提取的实体和关系存入Neo4j
        entities = kg_builder.get_extracted_entities()
        relations = kg_builder.get_extracted_relations()
        
        print(f"💾 正在将 {len(entities)} 个实体和 {len(relations)} 个关系存入Neo4j...")
        
        # ✅ 保存实体并检查结果
        if entities:
            entity_count = neo4j_manager.add_entities_batch(entities)
            if entity_count != len(entities):
                print(f"⚠️  警告: 只保存了 {entity_count}/{len(entities)} 个实体")
            else:
                print(f"✓ 已成功保存 {entity_count} 个实体到Neo4j")
        else:
            print("⚠️  没有实体需要保存")
        
        # ✅ 保存关系并检查结果
        if relations:
            relation_count = neo4j_manager.add_relations_batch(relations)
            if relation_count != len(relations):
                print(f"⚠️  警告: 只保存了 {relation_count}/{len(relations)} 个关系")
            else:
                print(f"✓ 已成功保存 {relation_count} 个关系到Neo4j")
        else:
            print("⚠️  没有关系需要保存")
        
        # ✅ 验证保存结果
        print(f"\n🔍 验证Neo4j数据库...")
        verification_stats = neo4j_manager.get_statistics()
        db_entities = verification_stats.get('total_entities', 0)
        db_relations = verification_stats.get('total_relations', 0)
        
        print(f"  Neo4j中的实体数: {db_entities}")
        print(f"  Neo4j中的关系数: {db_relations}")
        
        if db_entities == 0 and len(entities) > 0:
            print(f"  ⚠️  警告: 提取了 {len(entities)} 个实体，但Neo4j中为0，可能保存失败")
        if db_relations == 0 and len(relations) > 0:
            print(f"  ⚠️  警告: 提取了 {len(relations)} 个关系，但Neo4j中为0，可能保存失败")
        
        # 关闭连接
        neo4j_manager.close()
        
        if db_entities > 0 or db_relations > 0:
            print("✅ 知识图谱数据成功存入Neo4j")
        else:
            print("⚠️  知识图谱数据可能未成功保存（Neo4j中无数据）")
        
        print(f"📊 提取统计: {stats['entities']} 实体, {stats['relations']} 关系")
        print(f"📊 保存统计: {db_entities} 实体, {db_relations} 关系")
        
        return True

    except Exception as e:
        print(f"❌ 构建知识图谱失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def vectorize_chunks(chunks: List[Dict[str, Any]], force_rebuild: bool = False) -> bool:
    """向量化和存储"""
    print_step(4, "向量化处理", "生成嵌入向量并存储到Milvus")
    
    try:
        # 初始化向量管理器
        vector_manager = VectorManager()
        
        rebuild_needed = False
        if force_rebuild:
            print("🔧 --force-rebuild 参数已设置，强制重建")
            rebuild_needed = True
        elif vector_manager.is_vector_data_built():
            # 检查数据库是否真的有数据
            num_entities = vector_manager.vector_store.get_collection_num_entities()
            if num_entities == 0:
                print(f"⚠️  元数据存在但数据库为空 (0 vectors)，强制重建")
                rebuild_needed = True
            else:
                print(f"⚠️  向量数据已存在 ({num_entities} vectors)")
                try:
                    choice = input("是否重建向量数据? (y/N): ").strip().lower()
                    if choice == 'y':
                        rebuild_needed = True
                except EOFError:
                    # 在非交互模式下，默认不重建
                    print("💡 (非交互模式，已跳过提示)")
        else:
            # 元数据不存在，需要初次构建
            print("🔧 未找到现有数据，开始初次构建")
            rebuild_needed = True

        if not rebuild_needed:
            print("✓ 保持现有向量数据")
            return True

        print("🔄 准备构建/重建向量数据...")
        vector_manager.clear_vector_data()
        
        # 构建向量数据
        print("🚀 开始向量化处理...")
        start_time = time.time()
        
        # 使用正确的方法名
        success = vector_manager.build_vector_database(force_rebuild=True)
        
        if success:
            build_time = time.time() - start_time
            print(f"✓ 向量化完成，耗时 {build_time:.2f}秒")
            
            # ✅ 保存元数据（包含所有必需字段）
            metadata = {
                'vector_built': True,  # ✅ 关键字段：标记向量数据已构建
                'total_documents': len(set(c['metadata'].get('filename', '') for c in chunks)),
                'total_chunks': len(chunks),
                'total_vectors': len(chunks),  # ✅ 添加向量数量
                'build_time': datetime.now().isoformat(),
                'build_duration_seconds': build_time,
                'chunk_stats': vector_manager.text_splitter.get_stats(chunks),
                'splitting_mode': 'intelligent_layered',
                # ✅ 添加指纹信息（用于变化检测）
                'document_fingerprint': vector_manager._get_document_fingerprint(),
                'config_fingerprint': vector_manager._get_config_fingerprint(),
                'chunk_size': config.chunk_size,
                'chunk_overlap': config.chunk_overlap,
                'vector_dimension': config.vector_dimension
            }
            
            # 保存元数据到文件
            with open(vector_manager.metadata_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print("✓ 元数据保存完成（包含指纹信息）")
            
            return True
        else:
            print("❌ 向量化失败")
            return False
            
    except Exception as e:
        print(f"❌ 向量化处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_system():
    """测试RAG系统"""
    print_step(5, "RAG系统测试", "验证完整的问答流程")
    
    try:
        # 初始化RAG系统
        rag = RAGSystem(verbose=False)
        rag.initialize_vector_store()
        print("✓ RAG系统初始化完成")
        
        # 测试查询
        test_queries = [
            "What is the efficiency of perovskite solar cells?",
            "How are solar cell devices fabricated?",
            "What characterization methods are used?"
        ]
        
        print(f"\n🔍 运行 {len(test_queries)} 个测试查询:")
        
        all_queries_succeeded = True
        for i, query in enumerate(test_queries, 1):
            print(f"\n查询 {i}: {query}")
            
            try:
                start_time = time.time()
                result = rag.rag_query(query, top_k=3)
                query_time = time.time() - start_time
                
                if result.get('answer'):
                    answer_preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
                    print(f"✓ 回答: {answer_preview}")
                    print(f"✓ 检索文档: {result.get('retrieved_docs', 0)} 个")
                    print(f"✓ 排序方法: {result.get('ranking_method', 'unknown')}")
                    print(f"✓ 查询耗时: {query_time:.2f}秒")
                else:
                    print("❌ 查询失败")
                    all_queries_succeeded = False
                    
            except Exception as e:
                print(f"❌ 查询错误: {e}")
                all_queries_succeeded = False
        
        if all_queries_succeeded:
            print("\n✅ RAG系统测试完成")
            return True
        else:
            print("\n❌ RAG系统部分查询失败")
            return False
        
    except Exception as e:
        print(f"❌ RAG系统测试失败: {e}")
        return False

def print_summary(success_steps: List[bool]):
    """打印总结"""
    print_header("流水线执行总结")
    
    steps = [
        "环境检查",
        "文档加载", 
        "章节分块",
        "向量化处理",
        "RAG系统测试"
    ]
    
    print("📋 执行结果:")
    for i, (step, success) in enumerate(zip(steps, success_steps), 1):
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {i}. {step}: {status}")
    
    total_success = sum(success_steps)
    print(f"\n📊 成功率: {total_success}/{len(steps)} ({total_success/len(steps)*100:.1f}%)")
    
    if all(success_steps):
        print("\n🎉 流水线执行完成！")
        print("\n📋 下一步建议:")
        print("   1. 运行完整演示: python scripts/run_section_demo.py")
        print("   2. 运行RAG演示: python scripts/run_demo.py")
        print("   3. 运行评估: python run_evaluation.py --sample-size 3")
        print("   4. 查看文档: docs/COMPLETE_WORKFLOW.md")
    else:
        print("\n⚠️  流水线部分失败，请检查失败步骤")
        print("\n🔧 故障排除:")
        print("   1. 检查.env配置文件")
        print("   2. 确保Milvus服务运行: docker compose -f config/docker-compose.yml up -d")
        print("   3. 检查文档目录: ls -la", config.test_mds_path)
        print("   4. 重新运行: python run_pipeline.py")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAGPhoto_5 统一数据处理流水线。默认操作是构建向量数据库。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--vector',
        action='store_true',
        help='构建或更新向量数据库 (如果未指定任何操作，则为默认操作)'
    )
    parser.add_argument(
        '--kg',
        action='store_true',
        help='构建或更新知识图谱'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='与 --vector 一同使用，强制重建向量数据库'
    )
    args = parser.parse_args()

    # 如果没有指定任何操作，则默认为构建向量数据库
    if not args.vector and not args.kg:
        print("💡 未指定操作，将执行默认操作：构建向量数据库。")
        args.vector = True

    print_header("RAGPhoto_5 统一数据处理流水线")
    
    start_time = time.time()
    
    # 步骤1 & 2: 公共操作 (环境检查和文档加载)
    if not check_prerequisites():
        return
    
    documents = load_documents()
    if not documents:
        return

    # 步骤3-5: 构建向量数据库
    if args.vector:
        print_header("向量数据库流水线")
        chunks = split_documents(documents)
        if chunks:
            if vectorize_chunks(chunks, force_rebuild=args.force_rebuild):
                test_rag_system()
    
    # 步骤6: 构建知识图谱
    if args.kg:
        print_header("知识图谱构建流水线")
        build_knowledge_graph(documents)

    total_time = time.time() - start_time
    print_header("流水线执行完毕")
    print(f"⏱️  总耗时: {total_time:.2f}秒")
    print("\n✅ 操作完成。")
    print("\n📋 下一步建议:")
    print("   - 使用 --vector 或 --kg 选项来分别构建数据库。")
    print("   - 运行查询: python run_query.py \"您的查询问题\"")
    print("   - 查看知识图谱: python scripts/knowledge_graph/visualize_kg_graph.py")

if __name__ == "__main__":
    main() 