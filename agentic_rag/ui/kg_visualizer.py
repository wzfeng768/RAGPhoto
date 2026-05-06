"""
Knowledge Graph Visualizer for Agentic RAG v2
Enhanced visualization with real Neo4j data
"""

import plotly.graph_objects as go
import math
from typing import Dict, List, Any, Tuple
import networkx as nx
import re


def clean_latex_for_display(text: str) -> str:
    """
    Clean LaTeX expressions for display in Plotly (which doesn't support LaTeX)
    Converts LaTeX to more readable plain text
    
    Args:
        text: Text potentially containing LaTeX
        
    Returns:
        Cleaned text for display
    """
    if not text:
        return text
    
    # Remove LaTeX delimiters but keep content
    text = re.sub(r'\\\[(.*?)\\\]', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', r'\1', text, flags=re.DOTALL)
    
    # Clean common LaTeX commands
    text = re.sub(r'\\mathbf\{(.*?)\}', r'\1', text)  # Bold
    text = re.sub(r'\\mathrm\{(.*?)\}', r'\1', text)  # Roman
    text = re.sub(r'\\text\{(.*?)\}', r'\1', text)    # Text
    text = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', r'(\1)/(\2)', text)  # Fractions
    text = re.sub(r'\\times', '×', text)  # Times symbol
    text = re.sub(r'\\cdot', '·', text)   # Dot product
    text = re.sub(r'\\_', '_', text)      # Underscore
    text = re.sub(r'\\', '', text)        # Remove remaining backslashes
    
    return text


def get_full_kg_overview(neo4j_manager, sample_size: int = 100) -> Dict[str, Any]:
    """
    Get overview of the full knowledge graph (sampled)
    
    Args:
        neo4j_manager: Neo4jManager instance
        sample_size: Number of nodes to sample
        
    Returns:
        Dictionary with nodes and edges data
    """
    try:
        # First check if there's any data
        with neo4j_manager.driver.session(database=neo4j_manager.database) as session:
            count_result = session.run('MATCH (e:Entity) RETURN count(e) as count')
            total_count = count_result.single()['count']
            
            if total_count == 0:
                print("No entities found in knowledge graph")
                return {"nodes": [], "edges": [], "total_entities": 0}
        
        # Query with proper error handling
        cypher_query = f"""
        MATCH (e:Entity)
        WITH e
        ORDER BY rand()
        LIMIT {sample_size}
        OPTIONAL MATCH (e)-[r:RELATES]->(e2:Entity)
        RETURN 
            e.name as entity_name,
            e.type as entity_type,
            coalesce(e.description, '') as entity_desc,
            toString(id(e)) as entity_id,
            e2.name as target_name,
            e2.type as target_type,
            coalesce(e2.description, '') as target_desc,
            toString(id(e2)) as target_id,
            type(r) as rel_type,
            coalesce(r.type, '') as rel_subtype
        """
        
        nodes = []
        edges = []
        node_ids = set()
        
        with neo4j_manager.driver.session(database=neo4j_manager.database) as session:
            result = session.run(cypher_query)
            
            for record in result:
                # Add source entity
                entity_id = record.get('entity_id')
                entity_name = record.get('entity_name')
                
                if entity_id and entity_name and entity_id not in node_ids:
                    nodes.append({
                        'id': entity_id,
                        'name': entity_name,
                        'type': record.get('entity_type', 'Unknown'),
                        'description': record.get('entity_desc', '')
                    })
                    node_ids.add(entity_id)
                
                # Add target entity (if exists)
                target_id = record.get('target_id')
                target_name = record.get('target_name')
                
                if target_id and target_name and target_id not in node_ids:
                    nodes.append({
                        'id': target_id,
                        'name': target_name,
                        'type': record.get('target_type', 'Unknown'),
                        'description': record.get('target_desc', '')
                    })
                    node_ids.add(target_id)
                
                # Add relationship (if exists)
                if entity_id and target_id and target_name:
                    rel_type = record.get('rel_subtype') or record.get('rel_type', 'RELATES')
                    if rel_type != 'RELATES':  # Skip empty relationships
                        edges.append({
                            'source': entity_id,
                            'target': target_id,
                            'type': rel_type
                        })
        
        print(f"Loaded {len(nodes)} nodes and {len(edges)} edges from knowledge graph")
        return {"nodes": nodes, "edges": edges, "total_entities": total_count}
        
    except Exception as e:
        print(f"Error getting full KG overview: {e}")
        import traceback
        traceback.print_exc()
        return {"nodes": [], "edges": [], "error": str(e)}


def query_kg_data(neo4j_manager, query: str, max_nodes: int = 20) -> Dict[str, Any]:
    """
    Query Neo4j for entities and relationships related to the query
    
    Args:
        neo4j_manager: Neo4jManager instance
        query: User query string
        max_nodes: Maximum number of nodes to retrieve
        
    Returns:
        Dictionary with nodes and edges data
    """
    try:
        # Extract keywords from query
        keywords = extract_query_keywords(query)
        
        if not keywords:
            return {"nodes": [], "edges": []}
        
        # Build cypher query to find related entities
        keyword_pattern = " OR ".join([f"toLower(e.name) CONTAINS toLower('{kw}')" for kw in keywords[:3]])
        
        cypher_query = f"""
        MATCH (e:Entity)
        WHERE {keyword_pattern}
        WITH e LIMIT {max_nodes // 2}
        OPTIONAL MATCH (e)-[r:RELATES]->(e2:Entity)
        RETURN 
            e.name as entity_name,
            e.type as entity_type,
            coalesce(e.description, '') as entity_desc,
            toString(id(e)) as entity_id,
            e2.name as target_name,
            e2.type as target_type,
            coalesce(e2.description, '') as target_desc,
            toString(id(e2)) as target_id,
            type(r) as rel_type,
            r.type as rel_subtype
        """
        
        # Execute query using Neo4j driver session
        nodes = []
        edges = []
        node_ids = set()
        
        with neo4j_manager.driver.session(database=neo4j_manager.database) as session:
            result = session.run(cypher_query)
            
            for record in result:
                # Add source entity
                entity_id = record.get('entity_id')
                if entity_id and entity_id not in node_ids:
                    nodes.append({
                        'id': entity_id,
                        'name': record.get('entity_name', 'Unknown'),
                        'type': record.get('entity_type', 'Unknown'),
                        'description': record.get('entity_desc', '')
                    })
                    node_ids.add(entity_id)
                
                # Add target entity (if exists)
                target_id = record.get('target_id')
                if target_id and target_id not in node_ids:
                    nodes.append({
                        'id': target_id,
                        'name': record.get('target_name', 'Unknown'),
                        'type': record.get('target_type', 'Unknown'),
                        'description': record.get('target_desc', '')
                    })
                    node_ids.add(target_id)
                
                # Add relationship (if exists)
                if entity_id and target_id:
                    edges.append({
                        'source': entity_id,
                        'target': target_id,
                        'type': record.get('rel_subtype') or record.get('rel_type', 'RELATES')
                    })
        
        return {"nodes": nodes[:max_nodes], "edges": edges}
        
    except Exception as e:
        print(f"Error querying KG: {e}")
        import traceback
        traceback.print_exc()
        return {"nodes": [], "edges": []}


def extract_query_keywords(query: str, max_keywords: int = 5) -> List[str]:
    """Extract important keywords from query"""
    import re
    
    stop_words = {
        'what', 'how', 'why', 'when', 'where', 'which', 'who',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'from', 'as', 'can', 'could'
    }
    
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    return keywords[:max_keywords]


def create_overview_visualization(
    nodes: List[Dict],
    edges: List[Dict],
    highlighted_nodes: List[int] = None
) -> go.Figure:
    """
    Create interactive overview visualization of the full knowledge graph
    with zoom and drag capabilities
    
    Args:
        nodes: List of all nodes
        edges: List of all edges
        highlighted_nodes: List of node IDs to highlight
        
    Returns:
        Plotly figure object
    """
    if not nodes:
        fig = go.Figure()
        fig.add_annotation(
            text="No knowledge graph data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#64748b")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#f8fafc',
            height=400
        )
        return fig
    
    # Use networkx for layout
    G = nx.Graph()
    
    # Add nodes with attributes
    for node in nodes:
        node_id = node.get('id')
        G.add_node(node_id, **node)
    
    # Add edges
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source in G and target in G:
            G.add_edge(source, target)
    
    # Calculate layout
    try:
        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#cbd5e1'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Create node traces with hover info
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []
    node_hover = []
    
    highlighted_set = set(highlighted_nodes) if highlighted_nodes else set()
    
    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node_id]
        node_name = node_data.get('name', str(node_id))
        node_type = node_data.get('type', 'Unknown')
        
        # Clean LaTeX for display in Plotly
        display_name = clean_latex_for_display(node_name)
        
        # Highlight relevant nodes
        if node_id in highlighted_set:
            node_color.append('#ef4444')  # Red for highlighted
            node_size.append(12)
        else:
            node_color.append('#94a3b8')  # Gray for others
            node_size.append(6)
        
        # Add hover info with cleaned text
        node_hover.append(f"<b>{display_name}</b><br>Type: {node_type}<br>Connections: {G.degree(node_id)}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_hover,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='white')
        ),
        showlegend=False
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text='🌐 Full Knowledge Graph Overview (Drag to Pan, Scroll to Zoom)',
                font=dict(size=11, color='#64748b'),
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#f8fafc',
            paper_bgcolor='white',
            height=400,
            # Enable drag and zoom
            dragmode='pan',
            modebar=dict(
                orientation='h',
                bgcolor='rgba(255,255,255,0.8)',
                color='#64748b',
                activecolor='#3b82f6'
            )
        )
    )
    
    # Configure interactivity
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    
    return fig


def create_kg_visualization(
    nodes: List[Dict], 
    edges: List[Dict],
    title: str = "Knowledge Graph: Query-Related Concepts"
) -> go.Figure:
    """
    Create interactive knowledge graph visualization with edge labels
    
    Args:
        nodes: List of node dictionaries with id, name, type
        edges: List of edge dictionaries with source, target, type
        title: Graph title
        
    Returns:
        Plotly figure object
    """
    if not nodes:
        # Return empty graph with message
        fig = go.Figure()
        fig.add_annotation(
            text="No knowledge graph data available for this query",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#f8fafc',
            height=500,
            dragmode='pan'
        )
        return fig
    
    # Use networkx for better layout
    G = nx.Graph()
    
    # Add nodes
    node_map = {}  # Map node id to index
    for i, node in enumerate(nodes):
        node_id = node.get('id', i)
        G.add_node(node_id, **node)
        node_map[node_id] = i
    
    # Add edges with relationship type
    edge_labels = {}
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source in node_map and target in node_map:
            rel_type = edge.get('type', 'RELATES')
            G.add_edge(source, target, relation=rel_type)
            edge_labels[(source, target)] = rel_type
    
    # Calculate layout with more space
    try:
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#94a3b8'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Create edge label traces (relationship types)
    edge_label_x = []
    edge_label_y = []
    edge_label_text = []
    
    for (source, target), rel_type in edge_labels.items():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        # Position label at midpoint
        edge_label_x.append((x0 + x1) / 2)
        edge_label_y.append((y0 + y1) / 2)
        edge_label_text.append(rel_type)
    
    edge_label_trace = go.Scatter(
        x=edge_label_x,
        y=edge_label_y,
        mode='text',
        text=edge_label_text,
        textfont=dict(size=8, color='#475569', family='Arial'),
        textposition='middle center',
        hoverinfo='text',
        hovertext=edge_label_text,
        showlegend=False
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_color = []
    node_size = []
    
    # Color map for different entity types
    color_map = {
        "Material": "#10b981",
        "Device": "#3b82f6", 
        "Process": "#8b5cf6",
        "Property": "#f59e0b",
        "Structure": "#ec4899",
        "Concept": "#06b6d4",
        "Metric": "#f97316",
        "Technology": "#6366f1",
        "Algorithm": "#14b8a6"
    }
    
    for node_id in G.nodes():
        x, y = pos[node_id]
        node_data = G.nodes[node_id]
        
        node_x.append(x)
        node_y.append(y)
        
        name = node_data.get('name', str(node_id))
        description = node_data.get('description', '')
        
        # Clean LaTeX for display in Plotly
        display_name = clean_latex_for_display(name)
        display_desc = clean_latex_for_display(description)
        
        node_text.append(display_name[:25])  # Display name (truncated)
        
        # Hover info with cleaned text
        node_type = node_data.get('type', 'Unknown')
        degree = G.degree(node_id)
        hover_text = f"<b>{display_name}</b><br>Type: {node_type}<br>Connections: {degree}"
        if display_desc:
            hover_text += f"<br>Description: {display_desc[:150]}..."
        node_hover.append(hover_text)
        
        # Color by type
        node_color.append(color_map.get(node_type, "#64748b"))
        
        # Size by degree
        node_size.append(min(20 + degree * 5, 50))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        hovertext=node_hover,
        hoverinfo='text',
        textposition="top center",
        textfont=dict(size=10, color='#1e293b', family='Arial', weight='bold'),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white'),
            opacity=0.9
        ),
        showlegend=False
    )
    
    # Create figure with all traces
    fig = go.Figure(
        data=[edge_trace, edge_label_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=title + ' (Drag to Pan, Scroll to Zoom)',
                font=dict(size=14, color='#1e40af', family='Arial'),
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#f8fafc',
            paper_bgcolor='white',
            height=500,
            dragmode='pan',
            modebar=dict(
                orientation='h',
                bgcolor='rgba(255,255,255,0.8)',
                color='#64748b',
                activecolor='#3b82f6'
            )
        )
    )
    
    # Enable zoom and pan
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    
    return fig


def get_kg_statistics(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
    """Get statistics about the knowledge graph"""
    if not nodes:
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "node_types": {},
            "edge_types": {}
        }
    
    # Count node types
    node_types = {}
    for node in nodes:
        node_type = node.get('type', 'Unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    # Count edge types
    edge_types = {}
    for edge in edges:
        edge_type = edge.get('type', 'Unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "node_types": node_types,
        "edge_types": edge_types
    }
