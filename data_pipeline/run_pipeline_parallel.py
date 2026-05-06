#!/usr/bin/env python3
"""
RAGPhoto_5 完整数据处理流水线（并行版本）
从文档读取 → 章节分块 → 向量化 → 构建数据库 的完整流程
并行版本：使用多线程加速知识图谱提取
"""

import sys
import os
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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

# 全局锁用于线程安全的打印和统计
print_lock = Lock()
stats_lock = Lock()

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

def safe_print(message: str):
    """线程安全的打印"""
    with print_lock:
        print(message)

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

def process_chunk_batch(chunk_batch: List[Dict[str, Any]], batch_id: int, total_batches: int, 
                       kg_builder: PhotovoltaicKnowledgeGraph) -> tuple:
    """
    并行处理一个chunk批次，提取实体和关系
    
    Args:
        chunk_batch: 一批chunks
        batch_id: 批次ID
        total_batches: 总批次数
        kg_builder: 知识图谱构建器实例
        
    Returns:
        (entities, relations, batch_id)
    """
    try:
        safe_print(f"  🔄 批次 {batch_id}/{total_batches}: 处理 {len(chunk_batch)} 个chunks...")
        
        # 准备文本chunks和元数据
        text_chunks = []
        paper_ids = []
        metadata_list = []
        
        for chunk in chunk_batch:
            text_chunks.append(chunk.get('content', ''))
            paper_id = chunk.get('metadata', {}).get('source', f'paper_{batch_id}')
            paper_ids.append(paper_id)
            metadata_list.append(chunk.get('metadata', {}))
        
        # 提取实体和关系
        if kg_builder.use_traceable_extractor:
            entities, relations = kg_builder.llm_extractor.extract_entities_and_relations_batch(
                text_chunks=text_chunks,
                paper_id=f"batch_{batch_id}",
                metadata_list=metadata_list
            )
        else:
            entities, relations = kg_builder.llm_extractor.extract_entities_and_relations_batch(
                text_chunks=text_chunks,
                paper_ids=paper_ids
            )
        
        safe_print(f"  ✓ 批次 {batch_id}/{total_batches}: 提取了 {len(entities)} 个实体, {len(relations)} 个关系")
        
        return (entities, relations, batch_id)
        
    except Exception as e:
        safe_print(f"  ❌ 批次 {batch_id}/{total_batches} 处理失败: {e}")
        return ([], [], batch_id)

def build_knowledge_graph_parallel(documents: List[Dict[str, Any]], num_workers: int = 10, batch_size: int = 50) -> bool:
    """
    并行构建知识图谱
    
    Args:
        documents: 文档列表
        num_workers: 并行工作线程数（默认10）
        batch_size: 每个批次处理的chunk数量（默认50）
    """
    print_step(6, "构建知识图谱（并行版本）", 
              f"提取实体和关系并存储到Neo4j (并行度: {num_workers}, 批次大小: {batch_size})")
    
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
        print("✓ 知识图谱构建器初始化完成")
        
        # 使用智能分块器处理文档
        from core import PhotovoltaicTextSplitter
        splitter = PhotovoltaicTextSplitter()
        
        # 准备所有chunks
        all_chunks = []
        for i, doc in enumerate(documents):
            content = doc.get('content', '')
            if len(content.strip()) < 50:
                continue
                
            paper_id = doc.get('metadata', {}).get('source', f'paper_{i}')
            doc_metadata = doc.get('metadata', {})
            
            print(f"  📄 处理文档 {i+1}/{len(documents)}: {paper_id} ({len(content)} 字符)")
            
            # 智能分块
            chunks = splitter.split_text(content)
            print(f"     → 分为 {len(chunks)} 个智能chunks")
            
            # 为每个chunk准备元数据
            for chunk_idx, chunk in enumerate(chunks):
                chunk_content = chunk.get('content', '')
                chunk_metadata = chunk.get('metadata', {})
                
                combined_metadata = {
                    **doc_metadata,
                    **chunk_metadata,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                }
                
                all_chunks.append({
                    'content': chunk_content,
                    'metadata': combined_metadata
                })
        
        total_chunks = len(all_chunks)
        print(f"\n📊 总共 {total_chunks} 个chunks需要处理")
        
        # 将chunks分成批次
        batches = []
        for i in range(0, total_chunks, batch_size):
            batch = all_chunks[i:i + batch_size]
            batches.append(batch)
        
        total_batches = len(batches)
        print(f"📦 分为 {total_batches} 个批次，每批 {batch_size} 个chunks")
        print(f"🚀 使用 {num_workers} 个工作线程并行处理...\n")
        
        # 并行处理所有批次
        all_entities = []
        all_relations = []
        
        # 增量保存配置
        # 策略：每完成约10%的批次保存一次，但至少每5批保存一次，最多每20批保存一次
        # 这样可以确保：小规模任务频繁保存，大规模任务不会太频繁
        base_interval = max(1, total_batches // 10)  # 每完成10%保存一次
        incremental_save_interval = max(5, min(base_interval, 20))  # 限制在5-20之间
        
        saved_entities_count = 0
        saved_relations_count = 0
        
        start_time = time.time()
        
        save_percentage = (incremental_save_interval / total_batches * 100) if total_batches > 0 else 0
        print(f"💡 增量保存配置: 每完成 {incremental_save_interval} 个批次（约 {save_percentage:.1f}%）自动保存到Neo4j")
        print(f"💡 您可以在处理过程中查看Neo4j数据库了解提取进度")
        print(f"💡 实时监控命令: python scripts/knowledge_graph/view_kg_status_realtime.py\n")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(process_chunk_batch, batch, batch_id + 1, total_batches, kg_builder): batch_id
                for batch_id, batch in enumerate(batches)
            }
            
            # 收集结果
            completed = 0
            pending_entities = []
            pending_relations = []
            
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    entities, relations, _ = future.result()
                    all_entities.extend(entities)
                    all_relations.extend(relations)
                    pending_entities.extend(entities)
                    pending_relations.extend(relations)
                    completed += 1
                    
                    # 显示进度
                    progress = (completed / total_batches) * 100
                    safe_print(f"📊 进度: {completed}/{total_batches} 批次完成 ({progress:.1f}%)")
                    safe_print(f"   📈 累计提取: {len(all_entities)} 个实体, {len(all_relations)} 个关系")
                    
                    # 增量保存：每完成一定数量的批次就保存一次
                    # 条件说明：
                    # 1. completed % incremental_save_interval == 0: 每完成N个批次时保存（N=incremental_save_interval）
                    # 2. completed == total_batches: 完成所有批次时也保存（确保最后一次，无论剩余多少批次）
                    if completed % incremental_save_interval == 0 or completed == total_batches:
                        if pending_entities or pending_relations:
                            safe_print(f"\n💾 增量保存中... (批次 {completed}/{total_batches})")
                            
                            # 去重待保存的数据
                            unique_pending_entities = kg_builder._deduplicate_entities(pending_entities)
                            unique_pending_relations = kg_builder._deduplicate_relations(pending_relations)
                            
                            # 保存到Neo4j
                            if unique_pending_entities:
                                saved = neo4j_manager.add_entities_batch(unique_pending_entities)
                                saved_entities_count += saved
                                safe_print(f"   ✓ 已保存 {saved} 个实体 (累计: {saved_entities_count})")
                            
                            if unique_pending_relations:
                                saved = neo4j_manager.add_relations_batch(unique_pending_relations)
                                saved_relations_count += saved
                                safe_print(f"   ✓ 已保存 {saved} 个关系 (累计: {saved_relations_count})")
                            
                            # 查询当前Neo4j中的统计信息
                            try:
                                stats = neo4j_manager.get_statistics()
                                db_entities = stats.get('total_entities', 0)
                                db_relations = stats.get('total_relations', 0)
                                safe_print(f"   📊 Neo4j当前状态: {db_entities} 实体, {db_relations} 关系")
                                safe_print(f"   💡 提示: 可以在另一个终端运行查询命令查看知识图谱")
                            except Exception as e:
                                safe_print(f"   ⚠️  无法查询统计信息: {e}")
                            
                            # 清空待保存列表
                            pending_entities = []
                            pending_relations = []
                            
                            safe_print("")  # 空行分隔
                    
                except Exception as e:
                    safe_print(f"❌ 批次 {batch_id + 1} 处理异常: {e}")
        
        extraction_time = time.time() - start_time
        print(f"\n✓ 并行提取完成，耗时 {extraction_time:.2f}秒")
        print(f"✓ 提取了 {len(all_entities)} 个实体和 {len(all_relations)} 个关系")
        
        # 保存剩余的待保存数据
        if pending_entities or pending_relations:
            print(f"\n💾 保存最后一批数据...")
            unique_pending_entities = kg_builder._deduplicate_entities(pending_entities)
            unique_pending_relations = kg_builder._deduplicate_relations(pending_relations)
            
            if unique_pending_entities:
                saved = neo4j_manager.add_entities_batch(unique_pending_entities)
                saved_entities_count += saved
                print(f"   ✓ 已保存 {saved} 个实体")
            
            if unique_pending_relations:
                saved = neo4j_manager.add_relations_batch(unique_pending_relations)
                saved_relations_count += saved
                print(f"   ✓ 已保存 {saved} 个关系")
        
        # 去重和合并（最终统计）
        print("\n🔄 正在去重和合并所有实体/关系...")
        unique_entities = kg_builder._deduplicate_entities(all_entities)
        unique_relations = kg_builder._deduplicate_relations(all_relations)
        
        print(f"✓ 去重后: {len(unique_entities)} 个实体, {len(unique_relations)} 个关系")
        
        # 检查是否有未保存的数据
        if saved_entities_count < len(unique_entities) or saved_relations_count < len(unique_relations):
            print(f"\n💾 保存剩余数据到Neo4j...")
            
            # 获取已保存的实体ID（避免重复保存）
            try:
                existing_stats = neo4j_manager.get_statistics()
                existing_entity_count = existing_stats.get('total_entities', 0)
                existing_relation_count = existing_stats.get('total_relations', 0)
                
                # 如果已保存的数量少于去重后的数量，需要保存剩余部分
                if existing_entity_count < len(unique_entities):
                    # 这里简化处理：由于增量保存可能已经保存了大部分，只保存差异部分
                    # 实际实现中，增量保存应该已经处理了所有数据
                    print(f"   ℹ️  已通过增量保存保存了大部分数据")
            except:
                pass
        
        # ✅ 验证保存结果
        print(f"\n🔍 验证Neo4j数据库...")
        verification_stats = neo4j_manager.get_statistics()
        db_entities = verification_stats.get('total_entities', 0)
        db_relations = verification_stats.get('total_relations', 0)
        
        print(f"  Neo4j中的实体数: {db_entities}")
        print(f"  Neo4j中的关系数: {db_relations}")
        
        # 显示实体类型分布
        entity_types = verification_stats.get('entity_types', {})
        if entity_types and isinstance(entity_types, dict):
            print(f"\n  📊 实体类型分布:")
            for etype, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"     - {etype}: {count}")
        
        # 显示关系类型分布
        relation_types = verification_stats.get('relation_types', {})
        if relation_types and isinstance(relation_types, dict):
            print(f"\n  📊 关系类型分布:")
            for rtype, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"     - {rtype}: {count}")
        
        if db_entities == 0 and len(unique_entities) > 0:
            print(f"  ⚠️  警告: 提取了 {len(unique_entities)} 个实体，但Neo4j中为0，可能保存失败")
        if db_relations == 0 and len(unique_relations) > 0:
            print(f"  ⚠️  警告: 提取了 {len(unique_relations)} 个关系，但Neo4j中为0，可能保存失败")
        
        # 关闭连接
        neo4j_manager.close()
        
        if db_entities > 0 or db_relations > 0:
            print("\n✅ 知识图谱数据成功存入Neo4j")
        else:
            print("\n⚠️  知识图谱数据可能未成功保存（Neo4j中无数据）")
        
        print(f"\n📊 最终统计:")
        print(f"   提取统计: {len(unique_entities)} 实体, {len(unique_relations)} 关系")
        print(f"   保存统计: {db_entities} 实体, {db_relations} 关系")
        print(f"   总耗时: {extraction_time:.2f}秒")
        
        # 计算加速比（如果有原始版本的时间数据）
        print(f"\n💡 并行处理优势:")
        print(f"   - 使用 {num_workers} 个工作线程")
        print(f"   - 批次大小: {batch_size} chunks/批次")
        print(f"   - 预计加速比: ~{num_workers}x (取决于API响应时间)")
        
        print(f"\n💡 查看知识图谱:")
        print(f"   - Neo4j Browser: http://localhost:7475")
        print(f"   - 可视化脚本: python scripts/knowledge_graph/visualize_kg_graph.py")
        print(f"   - 查询示例: python -c \"from core.neo4j_manager import Neo4jManager; nm = Neo4jManager(); print(nm.get_statistics())\"")
        
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAGPhoto_5 统一数据处理流水线（并行版本）。默认操作是构建向量数据库。",
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
        help='构建或更新知识图谱（使用并行处理）'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='与 --vector 一同使用，强制重建向量数据库'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=256,
        help='并行工作线程数（默认: 10，建议范围: 2-16）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='每个批次处理的chunk数量（默认: 50，建议范围: 20-100）'
    )
    args = parser.parse_args()

    # 如果没有指定任何操作，则默认为构建向量数据库
    if not args.vector and not args.kg:
        print("💡 未指定操作，将执行默认操作：构建向量数据库。")
        args.vector = True

    print_header("RAGPhoto_5 统一数据处理流水线（并行版本）")
    print(f"🚀 并行配置: {args.num_workers} 个工作线程, 批次大小: {args.batch_size}")
    
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
    
    # 步骤6: 构建知识图谱（并行版本）
    if args.kg:
        print_header("知识图谱构建流水线（并行版本）")
        build_knowledge_graph_parallel(
            documents, 
            num_workers=args.num_workers,
            batch_size=args.batch_size
        )

    total_time = time.time() - start_time
    print_header("流水线执行完毕")
    print(f"⏱️  总耗时: {total_time:.2f}秒")
    print("\n✅ 操作完成。")
    print("\n📋 下一步建议:")
    print("   - 使用 --vector 或 --kg 选项来分别构建数据库。")
    print("   - 运行查询: python run_query.py \"您的查询问题\"")
    print("   - 查看知识图谱: python scripts/knowledge_graph/visualize_kg_graph.py")
    print("\n💡 并行版本优势:")
    print(f"   - 使用 {args.num_workers} 个工作线程并行处理")
    print(f"   - 预计加速比: ~{args.num_workers}x（取决于API响应时间）")
    print("   - 适合处理大量文档的知识图谱提取")

if __name__ == "__main__":
    main()


# # 使用默认配置（10个工作线程，每批50个chunks）
# python run_pipeline_parallel.py --kg

# # 如果需要调整，可以手动指定
# python run_pipeline_parallel.py --kg --num-workers 8  # 降低到8个
# python run_pipeline_parallel.py --kg --num-workers 16  # 提升到16个
