#!/usr/bin/env python3
"""
知识图谱状态查看工具
在并行处理过程中查看已提取的知识图谱统计信息
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.neo4j_manager import Neo4jManager

def print_kg_status():
    """打印知识图谱当前状态"""
    try:
        print("=" * 80)
        print("📊 知识图谱当前状态")
        print("=" * 80)
        
        neo4j_manager = Neo4jManager()
        
        # 获取统计信息
        stats = neo4j_manager.get_statistics()
        
        total_entities = stats.get('total_entities', 0)
        total_relations = stats.get('total_relations', 0)
        
        print(f"\n📈 总体统计:")
        print(f"   实体总数: {total_entities}")
        print(f"   关系总数: {total_relations}")
        
        # 实体类型分布
        entity_types = stats.get('entity_types', {})
        if entity_types and isinstance(entity_types, dict):
            print(f"\n📊 实体类型分布:")
            sorted_types = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)
            for etype, count in sorted_types[:15]:
                percentage = (count / total_entities * 100) if total_entities > 0 else 0
                print(f"   - {etype}: {count} ({percentage:.1f}%)")
        
        # 关系类型
        relation_types = stats.get('relation_types', {})
        if relation_types and isinstance(relation_types, dict):
            print(f"\n🔗 关系类型分布:")
            sorted_rels = sorted(relation_types.items(), key=lambda x: x[1], reverse=True)
            for rtype, count in sorted_rels[:15]:
                percentage = (count / total_relations * 100) if total_relations > 0 else 0
                print(f"   - {rtype}: {count} ({percentage:.1f}%)")
        
        # 查询一些示例实体
        print(f"\n🔍 示例实体 (前10个):")
        sample_entities = neo4j_manager.search_entities(limit=10)
        if sample_entities:
            for i, entity in enumerate(sample_entities, 1):
                name = entity.get('name', 'Unknown')
                etype = entity.get('type', 'Unknown')
                print(f"   {i}. {name} ({etype})")
        else:
            print("   (暂无实体)")
        
        neo4j_manager.close()
        
        print("\n" + "=" * 80)
        print("💡 提示:")
        print("   - 在并行处理过程中，数据会增量保存到Neo4j")
        print("   - 可以随时运行此脚本查看当前状态")
        print("   - Neo4j Browser: http://localhost:7475")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print_kg_status()

