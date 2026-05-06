"""
向量数据管理器
支持一次性构建、存储和管理向量数据，智能检测变化避免重复构建
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from config.config import config
from .document_loader import DocumentLoader
from .photovoltaic_text_splitter import PhotovoltaicTextSplitter
from .embeddings import EmbeddingGenerator
from .vector_store import MilvusVectorStore
import time

class VectorManager:
    """向量数据管理器（智能变化检测版）"""
    
    def __init__(self):
        """初始化向量管理器"""
        print("初始化向量数据管理器:")
        print(f"  文档路径: {config.test_mds_path}")
        print(f"  向量维度: {config.vector_dimension}")
        print(f"  分块大小: {config.chunk_size}")
        
        # 元数据文件路径
        self.metadata_file = os.path.join(
            config.project_root, 
            "milvus_data", 
            "vector_metadata.json"
        )
        print(f"  元数据文件: {self.metadata_file}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        
        # 初始化组件
        self.document_loader = DocumentLoader(config.test_mds_path)
        self.text_splitter = PhotovoltaicTextSplitter(
            default_chunk_size=config.chunk_size,
            default_chunk_overlap=config.chunk_overlap
        )
        self.embedding_generator = EmbeddingGenerator()
    
    def _get_document_fingerprint(self) -> str:
        """
        获取文档目录的指纹（用于检测文档变化）
        
        Returns:
            文档目录的MD5指纹
        """
        try:
            if not os.path.exists(config.test_mds_path):
                return ""
            
            # 收集所有文档文件的信息
            file_info = []
            for root, dirs, files in os.walk(config.test_mds_path):
                for file in files:
                    if file.endswith('.md'):
                        file_path = os.path.join(root, file)
                        try:
                            stat = os.stat(file_path)
                            file_info.append(f"{file}:{stat.st_size}:{stat.st_mtime}")
                        except Exception:
                            continue
            
            # 生成指纹
            content = "|".join(sorted(file_info))
            return hashlib.md5(content.encode()).hexdigest()
            
        except Exception as e:
            print(f"⚠️ 无法生成文档指纹: {e}")
            return ""
    
    def _get_config_fingerprint(self) -> str:
        """
        获取配置的指纹（用于检测配置变化）
        
        Returns:
            配置的MD5指纹
        """
        try:
            config_info = [
                f"chunk_size:{config.chunk_size}",
                f"chunk_overlap:{config.chunk_overlap}",
                f"vector_dimension:{config.vector_dimension}",
                f"embedding_model:{getattr(config, 'embedding_model', 'text-embedding-3-small')}",
                f"splitting_mode:intelligent_layered"
            ]
            content = "|".join(config_info)
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            print(f"⚠️ 无法生成配置指纹: {e}")
            return ""
    
    def _needs_rebuild(self) -> tuple[bool, str]:
        """
        检查是否需要重建向量数据库
        
        Returns:
            (是否需要重建, 原因说明)
        """
        # 如果元数据文件不存在，需要构建
        if not os.path.exists(self.metadata_file):
            return True, "向量数据不存在"
        
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            return True, f"元数据文件损坏: {e}"
        
        # 检查基本构建状态
        if not metadata.get('vector_built', False):
            return True, "向量数据未完成构建"
        
        # 检查文档变化
        current_doc_fingerprint = self._get_document_fingerprint()
        stored_doc_fingerprint = metadata.get('document_fingerprint', '')
        if current_doc_fingerprint != stored_doc_fingerprint:
            return True, "文档内容已发生变化"
        
        # 检查配置变化
        current_config_fingerprint = self._get_config_fingerprint()
        stored_config_fingerprint = metadata.get('config_fingerprint', '')
        if current_config_fingerprint != stored_config_fingerprint:
            return True, "分块配置已发生变化"
        
        # 检查集合是否存在
        try:
            vector_store = MilvusVectorStore()
            if not vector_store.collection_exists():
                return True, "Milvus集合不存在"
        except Exception as e:
            return True, f"无法连接Milvus: {e}"
        
        return False, "向量数据为最新状态"

    def is_vector_data_built(self) -> bool:
        """
        检查向量数据是否已构建且为最新
        
        Returns:
            是否已构建向量数据且无需重建
        """
        needs_rebuild, reason = self._needs_rebuild()
        if needs_rebuild:
            print(f"📋 检测结果: {reason}")
            return False
        else:
            print(f"✅ 检测结果: {reason}")
            return True
    
    def build_vector_database(self, force_rebuild: bool = False) -> bool:
        """
        构建向量数据库
        
        Args:
            force_rebuild: 是否强制重建
            
        Returns:
            构建是否成功
        """
        # 智能检测是否需要重建
        if not force_rebuild:
            needs_rebuild, reason = self._needs_rebuild()
            if not needs_rebuild:
                print(f"✓ 跳过构建: {reason}")
                return True
            else:
                print(f"🔄 需要重建: {reason}")
        else:
            print("🔄 强制重建向量数据库")
        
        print("\n🔨 开始构建向量数据库...")
        start_time = datetime.now()
        
        try:
            # 1. 加载文档
            print("1. 加载文档...")
            documents = self.document_loader.load_all_documents()
            
            if not documents:
                print("✗ 没有找到文档")
                return False
            
            # 2. 分割文档
            print("2. 分割文档...")
            chunks = self.text_splitter.split_documents(documents)
            print(f"✓ 生成了 {len(chunks)} 个文档分块")
            
            # 2.5. 增强文档块元数据（语义标注）
            print("2.5. 增强文档块元数据...")
            try:
                from core.chunk_metadata_enricher import ChunkMetadataEnricher
                enricher = ChunkMetadataEnricher()
                
                enriched_count = 0
                for chunk in chunks:
                    semantic_metadata = enricher.enrich_chunk(chunk['content'])
                    chunk['metadata'].update(semantic_metadata)
                    enriched_count += 1
                
                print(f"✓ 已增强 {enriched_count} 个文档块的元数据")
                
                # 统计元数据
                has_numerical = sum(1 for c in chunks if c['metadata'].get('has_numerical_data', False))
                has_methods = sum(1 for c in chunks if c['metadata'].get('has_methods', False))
                print(f"   - 包含数值数据: {has_numerical} 块")
                print(f"   - 包含方法描述: {has_methods} 块")
            except Exception as e:
                print(f"⚠️  元数据增强失败: {e}")
                print("   继续处理，但不包含语义标签")
            
            # 3. 生成嵌入向量（分批处理以避免内存问题）
            print("3. 生成嵌入向量...")
            start_embedding_time = time.time()
            
            chunk_contents = [chunk['content'] for chunk in chunks]
            
            # 智能分批处理 - 根据配置调整批次大小
            if config.enable_parallel_processing:
                # 并行处理时使用更大的批次
                batch_size = min(2000, len(chunk_contents))
            else:
                # 串行处理时使用中等批次
                batch_size = 1000
                
            valid_chunks = []  # 只保存成功生成embedding的chunks
            embeddings = []
            
            print(f"   优化批处理: 批次大小={batch_size}, 并行处理={'启用' if config.enable_parallel_processing else '禁用'}")
            
            # 预过滤无效内容
            valid_chunk_data = []
            for i, (chunk, content) in enumerate(zip(chunks, chunk_contents)):
                if not content or len(content.strip()) < config.min_chunk_length:
                    continue
                valid_chunk_data.append((chunk, content))
            
            print(f"   预过滤结果: {len(valid_chunk_data)}/{len(chunks)} 个有效分块")
            
            if not valid_chunk_data:
                print("⚠️ 没有有效的文档分块")
                return False
            
            # 提取有效内容进行批处理
            valid_chunks_list = [item[0] for item in valid_chunk_data]
            valid_contents_list = [item[1] for item in valid_chunk_data]
            
            for i in range(0, len(valid_contents_list), batch_size):
                batch_contents = valid_contents_list[i:i + batch_size]
                batch_chunks_slice = valid_chunks_list[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(valid_contents_list) + batch_size - 1)//batch_size
                
                print(f"   处理批次 {batch_num}/{total_batches} ({len(batch_contents)} 个分块)")
                batch_start_time = time.time()
                
                try:
                    batch_embeddings = self.embedding_generator.generate_embeddings_batch(batch_contents)
                    
                    if len(batch_embeddings) != len(batch_chunks_slice):
                        print(f"   ⚠️ 批次结果数量不匹配: embeddings={len(batch_embeddings)}, chunks={len(batch_chunks_slice)}")
                        # 逐个处理这个批次
                        for j, (chunk, content) in enumerate(zip(batch_chunks_slice, batch_contents)):
                            if j < len(batch_embeddings):
                                embeddings.append(batch_embeddings[j])
                                valid_chunks.append(chunk)
                    else:
                        embeddings.extend(batch_embeddings)
                        valid_chunks.extend(batch_chunks_slice)
                    
                    batch_elapsed = time.time() - batch_start_time
                    print(f"   ✓ 批次 {batch_num} 完成，用时 {batch_elapsed:.2f}秒")
                    
                except Exception as e:
                    print(f"   ⚠️ 批次处理失败，尝试单个处理: {e}")
                    # 如果批次失败，逐个处理
                    for j, (chunk, content) in enumerate(zip(batch_chunks_slice, batch_contents)):
                        if not content or not content.strip():
                            continue
                        try:
                            embedding = self.embedding_generator.generate_embedding(content)
                            embeddings.append(embedding)
                            valid_chunks.append(chunk)
                        except Exception as se:
                            print(f"   ✗ 跳过失败的文档 {j}: {se}")
                            continue
                
                print(f"   ✓ 已完成 {len(embeddings)}/{len(valid_contents_list)} 个嵌入向量")
            
            # 更新chunks为只包含有效的chunks
            chunks = valid_chunks
            
            # 确保数据一致性
            if len(chunks) != len(embeddings):
                raise ValueError(f"数据不一致: chunks({len(chunks)}) != embeddings({len(embeddings)})")
            
            embedding_elapsed = time.time() - start_embedding_time
            print(f"✓ 生成了 {len(embeddings)} 个嵌入向量，总用时 {embedding_elapsed:.2f}秒")
            
            # 显示缓存统计
            cache_stats = self.embedding_generator.get_cache_stats()
            print(f"✓ 缓存统计: {cache_stats['cache_size']} 个已缓存项")
            
            # 4. 初始化向量存储并插入数据
            print("4. 存储向量数据...")
            insert_start_time = time.time()
            
            vector_store = MilvusVectorStore()
            vector_store.create_collection()
            vector_store.create_index()
            
            # 优化插入批次大小
            insert_batch_size = min(500, len(chunks))  # 减小插入批次以提高稳定性
            total_inserted = 0
            
            for i in range(0, len(chunks), insert_batch_size):
                batch_chunks = chunks[i:i + insert_batch_size]
                batch_embeddings = embeddings[i:i + insert_batch_size]
                
                # 再次确认批次数据一致性
                if len(batch_chunks) != len(batch_embeddings):
                    print(f"   ⚠️ 批次数据不一致: chunks({len(batch_chunks)}) != embeddings({len(batch_embeddings)})")
                    continue
                
                insert_batch_num = i//insert_batch_size + 1
                total_insert_batches = (len(chunks) + insert_batch_size - 1)//insert_batch_size
                print(f"   插入批次 {insert_batch_num}/{total_insert_batches}")
                
                try:
                    vector_store.insert_documents(batch_chunks, batch_embeddings)
                    total_inserted += len(batch_chunks)
                    print(f"   ✓ 已插入 {total_inserted}/{len(chunks)} 个文档")
                except Exception as e:
                    print(f"   ⚠️ 批次插入失败: {e}")
                    # 尝试更小的批次
                    mini_batch_size = 100
                    for j in range(0, len(batch_chunks), mini_batch_size):
                        mini_batch_chunks = batch_chunks[j:j + mini_batch_size]
                        mini_batch_embeddings = batch_embeddings[j:j + mini_batch_size]
                        
                        if len(mini_batch_chunks) != len(mini_batch_embeddings):
                            print(f"   ⚠️ 小批次数据不一致: chunks({len(mini_batch_chunks)}) != embeddings({len(mini_batch_embeddings)})")
                            continue
                            
                        try:
                            vector_store.insert_documents(mini_batch_chunks, mini_batch_embeddings)
                            total_inserted += len(mini_batch_chunks)
                        except Exception as ee:
                            print(f"   ✗ 小批次也失败: {ee}")
                            continue
            
            insert_elapsed = time.time() - insert_start_time
            print(f"✓ 数据插入完成，用时 {insert_elapsed:.2f}秒")
            
            # 5. 加载集合
            vector_store.load_collection()
            
            # 构建完成
            end_time = datetime.now()
            build_duration = (end_time - start_time).total_seconds()
            
            print(f"✅ 向量数据库构建完成!")
            print(f"   构建时长: {build_duration:.2f}秒")
            print(f"   文档数量: {len(documents)}")
            print(f"   分块数量: {len(chunks)}")
            print(f"   向量数量: {len(embeddings)}")
            
            # 6. 保存元数据（包含指纹信息）
            metadata = {
                'vector_built': True,
                'build_time': end_time.isoformat(),
                'build_duration_seconds': build_duration,
                'total_documents': len(documents),
                'total_chunks': len(chunks),
                'total_vectors': len(embeddings),
                'document_fingerprint': self._get_document_fingerprint(),
                'config_fingerprint': self._get_config_fingerprint(),
                'chunk_size': config.chunk_size,
                'chunk_overlap': config.chunk_overlap,
                'vector_dimension': config.vector_dimension,
                'splitting_mode': 'intelligent_layered'
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print("✓ 元数据已保存")
            return True
            
        except Exception as e:
            print(f"❌ 向量数据库构建失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量管理器统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'vector_built': False,
            'build_time': None,
            'total_documents': 0,
            'total_chunks': 0,
            'total_vectors': 0
        }
        
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                stats.update(metadata)
            except Exception:
                pass
        
        return stats
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        
        print("\n📊 向量数据库状态:")
        print("=" * 50)
        
        if stats['vector_built']:
            print("状态: ✓ 已构建")
            print(f"构建时间: {stats.get('build_time', 'N/A')}")
            print(f"文档数量: {stats.get('total_documents', 0)}")
            print(f"分块数量: {stats.get('total_chunks', 0)}")
            print(f"向量数量: {stats.get('total_vectors', 0)}")
            print(f"分块大小: {stats.get('chunk_size', 'N/A')}")
            print(f"重叠大小: {stats.get('chunk_overlap', 'N/A')}")
        else:
            print("状态: ✗ 未构建")
    
    def rebuild_database(self) -> bool:
        """
        重建数据库
        
        Returns:
            重建是否成功
        """
        return self.build_vector_database(force_rebuild=True)
    
    def clear_vector_data(self):
        """清除向量数据"""
        try:
            # 删除Milvus集合
            vector_store = MilvusVectorStore()
            if vector_store.collection_exists():
                vector_store.drop_collection()
                print("✓ 已删除Milvus集合")
            
            # 删除元数据文件
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
                print("✓ 已删除元数据文件")
                
        except Exception as e:
            print(f"⚠️ 清除数据时出错: {e}")

    def ensure_vector_data(self) -> bool:
        """
        确保向量数据存在且为最新，智能检测变化
        
        Returns:
            向量数据是否可用
        """
        print("\n🔍 检查向量数据状态...")
        
        needs_rebuild, reason = self._needs_rebuild()
        
        if needs_rebuild:
            print(f"📋 {reason}，开始构建...")
            return self.build_vector_database()
        else:
            print(f"✅ {reason}，跳过构建")
            return True 