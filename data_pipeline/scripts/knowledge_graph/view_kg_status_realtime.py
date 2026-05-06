#!/usr/bin/env python3
"""
知识图谱实时状态监控工具
在并行处理过程中实时查看已提取的知识图谱统计信息（自动刷新）
"""

import sys
import os
import time
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.neo4j_manager import Neo4jManager

def clear_screen():
    """清屏（跨平台）"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_kg_status(clear: bool = False, show_timestamp: bool = True):
    """打印知识图谱当前状态"""
    try:
        if clear:
            clear_screen()
        
        print("=" * 80)
        print("📊 知识图谱实时状态监控")
        if show_timestamp:
            print(f"⏰ 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        neo4j_manager = Neo4jManager()
        
        # 获取统计信息
        stats = neo4j_manager.get_statistics()
        
        total_entities = stats.get('total_entities', 0)
        total_relations = stats.get('total_relations', 0)
        
        print(f"\n📈 总体统计:")
        print(f"   实体总数: {total_entities:,}")
        print(f"   关系总数: {total_relations:,}")
        
        # 实体类型分布
        entity_types = stats.get('entity_types', {})
        if entity_types and isinstance(entity_types, dict):
            print(f"\n📊 实体类型分布 (Top 10):")
            sorted_types = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)
            for etype, count in sorted_types[:10]:
                percentage = (count / total_entities * 100) if total_entities > 0 else 0
                print(f"   - {etype}: {count:,} ({percentage:.1f}%)")
        
        # 关系类型分布
        relation_types = stats.get('relation_types', {})
        if relation_types and isinstance(relation_types, dict):
            print(f"\n🔗 关系类型分布 (Top 10):")
            sorted_rels = sorted(relation_types.items(), key=lambda x: x[1], reverse=True)
            for rtype, count in sorted_rels[:10]:
                percentage = (count / total_relations * 100) if total_relations > 0 else 0
                print(f"   - {rtype}: {count:,} ({percentage:.1f}%)")
        
        # 查询一些示例实体
        print(f"\n🔍 示例实体 (前5个):")
        sample_entities = neo4j_manager.search_entities(limit=5)
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
        print("   - 数据会定期自动刷新")
        print("   - 按 Ctrl+C 退出监控")
        print("   - Neo4j Browser: http://localhost:7475")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitor_realtime(interval: int = 5, clear_screen: bool = True):
    """
    实时监控知识图谱状态
    
    Args:
        interval: 刷新间隔（秒），默认5秒
        clear_screen: 是否清屏，默认True
    """
    print("🚀 启动实时监控...")
    print(f"📊 刷新间隔: {interval} 秒")
    print(f"🔄 清屏模式: {'开启' if clear_screen else '关闭'}")
    print("💡 按 Ctrl+C 退出监控\n")
    
    time.sleep(2)  # 等待2秒后开始
    
    try:
        last_entities = 0
        last_relations = 0
        
        while True:
            # 打印状态
            success = print_kg_status(clear=clear_screen, show_timestamp=True)
            
            if success:
                # 获取当前统计（用于显示变化）
                try:
                    neo4j_manager = Neo4jManager()
                    stats = neo4j_manager.get_statistics()
                    current_entities = stats.get('total_entities', 0)
                    current_relations = stats.get('total_relations', 0)
                    neo4j_manager.close()
                    
                    # 显示变化
                    if last_entities > 0 or last_relations > 0:
                        entity_diff = current_entities - last_entities
                        relation_diff = current_relations - last_relations
                        
                        if entity_diff > 0 or relation_diff > 0:
                            print(f"\n📈 变化: +{entity_diff:,} 实体, +{relation_diff:,} 关系")
                        elif entity_diff < 0 or relation_diff < 0:
                            print(f"\n📉 变化: {entity_diff:,} 实体, {relation_diff:,} 关系")
                        else:
                            print(f"\n➡️  无变化")
                    
                    last_entities = current_entities
                    last_relations = current_relations
                except:
                    pass
            
            # 等待下次刷新
            print(f"\n⏳ {interval} 秒后自动刷新... (按 Ctrl+C 退出)")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ 监控已停止")
        print("💡 提示: 可以运行 'python scripts/knowledge_graph/view_kg_status.py' 查看最终状态")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="知识图谱实时状态监控工具",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='刷新间隔（秒），默认5秒'
    )
    parser.add_argument(
        '--no-clear',
        action='store_true',
        help='不清屏（保留历史记录）'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='只查看一次，不持续监控'
    )
    
    args = parser.parse_args()
    
    if args.once:
        # 只查看一次
        print_kg_status(clear=False, show_timestamp=True)
    else:
        # 实时监控
        monitor_realtime(interval=args.interval, clear_screen=not args.no_clear)

if __name__ == "__main__":
    main()


# # 实时监控，每5秒自动刷新（默认）
# python scripts/knowledge_graph/view_kg_status_realtime.py

# # 自定义刷新间隔（每3秒刷新一次）
# python scripts/knowledge_graph/view_kg_status_realtime.py --interval 3

# # 不清屏模式（保留历史记录，方便对比）
# python scripts/knowledge_graph/view_kg_status_realtime.py --no-clear

# # 只查看一次（不持续监控）
# python scripts/knowledge_graph/view_kg_status_realtime.py --once
