#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Graph Graphical Visualization
Generate visual graph representations of the knowledge graph
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
# 先加载环境变量和配置
from config.config import load_env_file

# 再导入其他模块
from core.neo4j_manager import Neo4jManager
from core.llm_extractor import LLMEntityRelationExtractor

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib import font_manager
import numpy as np

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def clean_label_text(text: str, max_length: int = 30) -> str:
    """
    清理标签文本，移除LaTeX数学公式符号，避免matplotlib解析错误
    
    Args:
        text: 原始文本
        max_length: 最大长度，超过则截断
    
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    import re
    
    # 保存原始文本用于后续处理
    original_text = text
    
    # 方法1: 移除LaTeX数学公式符号 $...$ 或 $$...$$
    # 匹配 $...$ 或 $$...$$，包括嵌套的情况
    text = re.sub(r'\$\$?[^$]*\$\$?', '', text)
    
    # 方法2: 如果文本仍然包含 $，尝试提取 $ 之间的内容并简化
    if '$' in text:
        # 移除所有 $ 符号
        text = text.replace('$', '')
    
    # 移除LaTeX命令（如 \boldsymbol, \mathsf, \mathrm, \frac 等）
    # 匹配 \command{...} 或 \command 格式
    text = re.sub(r'\\[a-zA-Z]+\s*\{[^}]*\}', '', text)  # \command{content}
    text = re.sub(r'\\[a-zA-Z]+\s*', '', text)  # \command
    
    # 移除LaTeX特殊字符和符号
    text = re.sub(r'[{}_]', ' ', text)
    
    # 移除LaTeX数学符号（如 \times, \div, \pm 等）
    text = re.sub(r'\\[^a-zA-Z\s]', '', text)
    
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 如果清理后为空或太短，尝试保留一些原始内容
    if not text or len(text) < 3:
        # 尝试从原始文本中提取纯文本部分
        # 移除所有 $ 和 \ 开头的命令
        fallback = original_text
        fallback = re.sub(r'\$[^$]*\$', '', fallback)  # 移除 $...$
        fallback = re.sub(r'\\[a-zA-Z]+\s*\{[^}]*\}', '', fallback)  # 移除 \command{...}
        fallback = re.sub(r'\\[a-zA-Z]+', '', fallback)  # 移除 \command
        fallback = re.sub(r'[{}_$\\]', ' ', fallback)  # 移除特殊字符
        fallback = re.sub(r'\s+', ' ', fallback).strip()
        
        if fallback and len(fallback) > len(text):
            text = fallback
    
    # 如果文本太长，截断并添加省略号
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    
    # 如果清理后仍为空，使用占位符
    if not text:
        # 尝试使用原始文本的前几个字符
        if original_text:
            # 只保留字母、数字和常见标点
            simple_text = re.sub(r'[^a-zA-Z0-9\s\-.,;:()]', '', original_text)
            if simple_text.strip():
                text = simple_text.strip()[:max_length]
            else:
                text = "[Math]"
        else:
            text = "[Empty]"
    
    return text

def visualize_knowledge_graph(output_path: str = "outputs/visualizations", limit: int = 200):
    """
    Fetch the knowledge graph from Neo4j and generate visual representations.
    """
    print("🎨 Starting knowledge graph visualization...")

    # Build absolute path for output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, output_path)
    os.makedirs(output_dir, exist_ok=True)

    neo4j_manager = None
    try:
        # Step 1: Fetch data from Neo4j
        print(f"📊 Step 1: Fetching graph data from Neo4j (limit: {limit} nodes)...")
        neo4j_manager = Neo4jManager()
        entities, relations = neo4j_manager.get_full_graph(limit=limit)

        if not entities:
            print("❌ No data found in Neo4j database. Cannot generate visualization.")
            print("💡 Tip: Run the knowledge graph extraction pipeline first:")
            print("   python scripts/knowledge_graph/extract_knowledge_graph.py")
            return

        print(f"✅ Fetched {len(entities)} entities and {len(relations)} relations.")

        # Step 2: Create NetworkX graph
        print("\n📈 Step 2: Building networkx graph object...")
        G = nx.DiGraph()
        
        # Define colors for different entity types
        entity_colors = {
            'Material': '#FF6B6B', 'Device': '#4ECDC4', 'Metric': '#45B7D1',
            'Process': '#FFA07A', 'Measurement': '#98D8C8', 'Author': '#F7DC6F',
            'Institution': '#BB8FCE', 'Concept': '#85C1E9', 'Parameter': '#FFD166'
        }
        
        # Add nodes (entities)
        # 创建节点名称到清理后标签的映射
        node_labels = {}
        for entity in entities:
            entity_name = entity['name']
            G.add_node(entity_name, type=entity['type'], properties=entity.get('properties', {}))
            # 为可视化创建清理后的标签
            node_labels[entity_name] = clean_label_text(entity_name)
        
        # Add edges (relations)
        for rel in relations:
            # Ensure both source and target nodes are in the graph before adding edge
            if G.has_node(rel['source_name']) and G.has_node(rel['target_name']):
                G.add_edge(rel['source_name'], rel['target_name'], type=rel['type'])

        print(f"✅ Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Step 3: Generate visualizations
        print("\n🎨 Step 3: Generating visualization images...")
        
        layouts = {
            "network": (nx.spring_layout, {'k': 0.8, 'iterations': 50, 'seed': 42}),
            "circular": (nx.circular_layout, {}),
            "hierarchical": (nx.kamada_kawai_layout, {})
        }

        for layout_name, (layout_func, params) in layouts.items():
            print(f"  - Generating {layout_name} layout...")
            plt.figure(figsize=(24, 18))
            pos = layout_func(G, **params)
            
            # Draw nodes
            node_colors = [entity_colors.get(data.get('type', 'Concept'), '#EAEAEA') for _, data in G.nodes(data=True)]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8, edgecolors='black')
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, width=1.5, alpha=0.7, connectionstyle="arc3,rad=0.1")
            
            # Draw labels with cleaned text (avoid LaTeX parsing errors)
            # 使用清理后的标签，避免matplotlib解析LaTeX数学公式时出错
            labels_dict = {}
            for node in G.nodes():
                try:
                    cleaned_label = node_labels.get(node, clean_label_text(node))
                    # 确保标签不为空且不包含特殊字符
                    if cleaned_label and len(cleaned_label) > 0:
                        labels_dict[node] = cleaned_label
                    else:
                        labels_dict[node] = f"Node_{hash(node) % 10000}"
                except Exception as e:
                    # 如果清理失败，使用简化标签
                    labels_dict[node] = f"Node_{hash(node) % 10000}"
            
            # 禁用matplotlib的数学模式，避免解析LaTeX公式
            plt.rcParams['text.usetex'] = False
            
            try:
                nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=8, font_weight='bold', 
                                       font_family='sans-serif', bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', edgecolor='none', alpha=0.7))
            except Exception as e:
                print(f"  ⚠️  绘制标签时出错: {e}")
                print(f"  ℹ️  尝试使用简化标签...")
                # 如果仍然出错，使用节点索引作为标签
                simple_labels = {node: f"E{i}" for i, node in enumerate(G.nodes())}
                nx.draw_networkx_labels(G, pos, labels=simple_labels, font_size=8, font_weight='bold')
            
            plt.title(f"Photovoltaic Knowledge Graph ({layout_name.capitalize()} Layout)", fontsize=20, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            file_path = os.path.join(output_dir, f"knowledge_graph_{layout_name}.png")
            plt.savefig(file_path, format="PNG", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Visualization saved to: {file_path}")

        print("\n🎉 Knowledge graph visualizations created successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Visualization interrupted by user")
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保连接被关闭
        if neo4j_manager is not None:
            try:
                neo4j_manager.close()
                print("✓ Neo4j connection closed")
            except Exception:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations from Neo4j Knowledge Graph.")
    parser.add_argument('--limit', type=int, default=200, help='Maximum number of nodes to fetch from the graph.')
    args = parser.parse_args()

    try:
        visualize_knowledge_graph(limit=args.limit)
    except Exception as e:
        print(f"❌ Main execution failed: {e}")
        import traceback
        traceback.print_exc() 