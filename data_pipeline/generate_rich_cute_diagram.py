import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
import matplotlib.patheffects as path_effects

def create_rich_cute_diagram():
    # Setup canvas
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # --- Color Palette (Pastel/Cute) ---
    colors = {
        'paper': '#FFFDF5',
        'paper_shadow': '#E0E0E0',
        'ink': '#2D3436',
        'highlight': '#FF7675', # Pink/Red
        'process_blue': '#74B9FF',
        'process_green': '#55EFC4',
        'process_purple': '#A29BFE',
        'process_yellow': '#FFEAA7',
        'milvus': '#0984E3',
        'neo4j': '#00B894',
        'solar': '#FDCB6E', # Sun color
        'text': '#2D3436'
    }

    # --- Helper Functions for Complex Icons ---

    def draw_hand_drawn_rect(x, y, w, h, color, edgecolor='black', lw=2, label=None, rotate=0):
        # Add slight randomness to mimic hand-drawn
        rect = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=f"round,pad={0.05*w},rounding_size={0.1*min(w,h)}",
            facecolor=color, edgecolor=edgecolor, linewidth=lw,
            transform=patches.Affine2D().rotate_deg_around(x, y, rotate) + ax.transData
        )
        # Shadow
        shadow = patches.FancyBboxPatch(
            (x - w/2 + 0.3, y - h/2 - 0.3), w, h,
            boxstyle=f"round,pad={0.05*w},rounding_size={0.1*min(w,h)}",
            facecolor='#000000', alpha=0.1, zorder=rect.zorder-1,
            transform=patches.Affine2D().rotate_deg_around(x, y, rotate) + ax.transData
        )
        ax.add_patch(shadow)
        ax.add_patch(rect)
        if label:
            ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color=colors['text'])

    def draw_doc_with_details(x, y, scale=1.0, rotate=0, content_type="text"):
        w, h = 5*scale, 7*scale
        # Paper
        draw_hand_drawn_rect(x, y, w, h, colors['paper'], rotate=rotate)
        
        # Content details
        t = patches.Affine2D().rotate_deg_around(x, y, rotate) + ax.transData
        
        if content_type == "formula":
            ax.text(x, y+1*scale, r"$E=mc^2$", ha='center', va='center', fontsize=6*scale, transform=t, alpha=0.6)
            ax.text(x, y-1*scale, r"$\eta > 20\%$", ha='center', va='center', fontsize=6*scale, transform=t, alpha=0.6)
        elif content_type == "chart":
            # Draw mini bar chart
            ax.bar([x-1*scale, x, x+1*scale], [1*scale, 2*scale, 1.5*scale], 
                   width=0.6*scale, bottom=y-2*scale, color=colors['process_blue'], alpha=0.6, transform=t)
        else: # text lines
            for i in range(4):
                line_y = y + 1.5*scale - i*1.0*scale
                ax.plot([x-1.5*scale, x+1.5*scale], [line_y, line_y], 
                        color='#B2BEC3', lw=1.5*scale, transform=t)

    def draw_cute_sun(x, y, scale=1.0):
        # Sun body
        sun = patches.Circle((x, y), 3*scale, facecolor=colors['solar'], edgecolor=colors['ink'], lw=2)
        ax.add_patch(sun)
        # Rays
        for i in range(8):
            angle = i * (360/8)
            rad = np.radians(angle)
            x1 = x + 3.5*scale * np.cos(rad)
            y1 = y + 3.5*scale * np.sin(rad)
            x2 = x + 5*scale * np.cos(rad)
            y2 = y + 5*scale * np.sin(rad)
            ax.plot([x1, x2], [y1, y2], color=colors['solar'], lw=3*scale)
        # Sunglasses on sun (RAGPhoto vibes)
        ax.add_patch(patches.Wedge((x-1*scale, y+0.5*scale), 1*scale, 180, 360, facecolor='black'))
        ax.add_patch(patches.Wedge((x+1*scale, y+0.5*scale), 1*scale, 180, 360, facecolor='black'))
        ax.plot([x-0.2*scale, x+0.2*scale], [y+0.5*scale, y+0.5*scale], color='black', lw=1)

    def draw_smart_scissors(x, y, scale=1.0):
        # Scissors body
        draw_hand_drawn_rect(x, y, 4*scale, 4*scale, colors['process_blue'], rotate=45)
        ax.text(x, y, "✂️", fontsize=25*scale, ha='center', va='center', rotation=90)
        
        # Ruler showing precision
        ax.plot([x-3*scale, x-3*scale], [y-3*scale, y+3*scale], color=colors['ink'], lw=2)
        for i in range(6):
            tick_y = y - 3*scale + i*1.2*scale
            ax.plot([x-3*scale, x-2.5*scale], [tick_y, tick_y], color=colors['ink'], lw=1)
        ax.text(x-3.5*scale, y, "Ruler", rotation=90, ha='center', va='center', fontsize=8*scale)

    def draw_ai_brain_bot(x, y, scale=1.0, type='vector'):
        # Head
        draw_hand_drawn_rect(x, y, 5*scale, 4.5*scale, colors['process_purple'])
        # Eyes
        ax.add_patch(patches.Circle((x-1*scale, y+0.5*scale), 0.8*scale, facecolor='white', edgecolor='black'))
        ax.add_patch(patches.Circle((x+1*scale, y+0.5*scale), 0.8*scale, facecolor='white', edgecolor='black'))
        # Pupils
        ax.add_patch(patches.Circle((x-1*scale, y+0.5*scale), 0.3*scale, facecolor='black'))
        ax.add_patch(patches.Circle((x+1*scale, y+0.5*scale), 0.3*scale, facecolor='black'))
        
        if type == 'vector':
            # Binary code thought bubble
            ax.text(x+3*scale, y+3*scale, "0101\n1010", fontsize=8*scale, 
                    bbox=dict(boxstyle="circle", facecolor="white", edgecolor=colors['ink']))
            label = "Embedding\nModel"
        else:
            # Magnifying glass
            ax.text(x+3*scale, y+2*scale, "🔍", fontsize=15*scale)
            label = "KG LLM\nExtractor"
            
        ax.text(x, y-4*scale, label, ha='center', va='top', fontsize=10*scale, fontweight='bold')

    def draw_vector_cube_storage(x, y, scale=1.0):
        # Draw a "transparent" container
        draw_hand_drawn_rect(x, y, 7*scale, 9*scale, colors['milvus'], rotate=0)
        # Draw cubes inside
        positions = [(-1.5, 2), (1.5, 2), (-1.5, -1), (1.5, -1)]
        for dx, dy in positions:
            # Cube face
            r = patches.Rectangle((x+dx*scale-1*scale, y+dy*scale-1*scale), 2*scale, 2*scale, 
                                  facecolor='white', edgecolor='white', alpha=0.3)
            ax.add_patch(r)
            ax.text(x+dx*scale, y+dy*scale, "[v]", ha='center', va='center', fontsize=6*scale, color='white')
        
        ax.text(x, y-6*scale, "Milvus DB\n(Vectors)", ha='center', va='top', fontsize=11*scale, fontweight='bold')

    def draw_graph_network_storage(x, y, scale=1.0):
        # Draw container
        draw_hand_drawn_rect(x, y, 7*scale, 9*scale, colors['neo4j'], rotate=0)
        # Draw nodes and edges
        nodes = [(0, 2), (-2, -1), (2, -1), (0, -3)]
        for i, (nx, ny) in enumerate(nodes):
            ax.add_patch(patches.Circle((x+nx*scale, y+ny*scale), 0.8*scale, facecolor='white', edgecolor='white'))
            # Edges to next node
            if i < len(nodes)-1:
                next_x, next_y = nodes[i+1]
                ax.plot([x+nx*scale, x+next_x*scale], [y+ny*scale, y+next_y*scale], color='white', lw=1.5)
        
        # Cross edge
        ax.plot([x+nodes[1][0]*scale, x+nodes[2][0]*scale], [y+nodes[1][1]*scale, y+nodes[2][1]*scale], color='white', lw=1.5)
        
        ax.text(x, y-6*scale, "Neo4j DB\n(Graph)", ha='center', va='top', fontsize=11*scale, fontweight='bold')

    def draw_curved_arrow_path(x1, y1, x2, y2, style='simple', color='#555'):
        # Custom hand-drawn style arrow
        if style == 'wavy':
            mid_x = (x1 + x2) / 2
            # Bezier curve mimic
            path_data = [
                (Path.MOVETO, (x1, y1)),
                (Path.CURVE3, (mid_x, y1)),
                (Path.CURVE3, (mid_x, (y1+y2)/2)),
                (Path.CURVE3, (mid_x, y2)),
                (Path.LINETO, (x2, y2))
            ]
        else:
             path_data = [
                (Path.MOVETO, (x1, y1)),
                (Path.LINETO, (x2, y2))
            ]
            
        # Draw arrow head separately for cuteness
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=2, head_length=2, fc=color, ec=color, length_includes_head=True, lw=2, zorder=0)

    # --- SCENE COMPOSITION ---

    # 1. Background & Decor
    draw_cute_sun(90, 90, scale=1.5)
    ax.text(50, 95, "RAGPhoto Data Factory", fontsize=24, fontweight='bold', ha='center', color=colors['ink'])
    
    # 2. Input Section (Left)
    ax.text(10, 60, "Input Source", fontsize=14, fontweight='bold', color=colors['ink'], ha='center')
    draw_doc_with_details(10, 50, scale=1.2, rotate=-5, content_type="formula")
    draw_doc_with_details(12, 48, scale=1.2, rotate=5, content_type="chart")
    draw_doc_with_details(8, 45, scale=1.2, rotate=-2, content_type="text")
    
    # 3. Processing Center (Intelligent Splitting)
    draw_curved_arrow_path(18, 50, 28, 50)
    
    # "Cleaning & Sectioning" Box
    draw_hand_drawn_rect(35, 50, 12, 10, colors['process_yellow'])
    ax.text(35, 53, "Cleaner &\nSectioning", ha='center', va='center', fontweight='bold')
    # Add icon for regex/cleaning
    ax.text(35, 47, "✨", ha='center', va='center', fontsize=20)
    
    # Arrow to Splitter
    draw_curved_arrow_path(41, 50, 48, 50)
    
    # "Intelligent Splitter"
    draw_smart_scissors(52, 50, scale=1.2)
    ax.text(52, 40, "Intelligent\nLayered Splitter", ha='center', va='top', fontweight='bold', fontsize=10)
    
    # --- Branching Paths ---
    
    # Path A: Vector (Up)
    draw_curved_arrow_path(55, 55, 65, 75)
    
    # Strategy Bubbles
    ax.text(60, 65, "Abstract\n(Small)", fontsize=8, ha='center', 
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=colors['ink'], alpha=0.8))
    ax.text(62, 70, "Results\n(Large)", fontsize=8, ha='center', 
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=colors['ink'], alpha=0.8))
            
    # AI Model (Embedding)
    draw_ai_brain_bot(70, 75, scale=1.2, type='vector')
    
    draw_curved_arrow_path(76, 75, 85, 75)
    
    # Milvus DB
    draw_vector_cube_storage(90, 75, scale=1.2)
    
    # Path B: KG (Down)
    draw_curved_arrow_path(55, 45, 65, 25)
    
    # Strategy Bubbles for KG
    ax.text(60, 35, "Methodology\n(Process)", fontsize=8, ha='center', 
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=colors['ink'], alpha=0.8))
            
    # AI Model (Extractor)
    draw_ai_brain_bot(70, 25, scale=1.2, type='kg')
    
    draw_curved_arrow_path(76, 25, 85, 25)
    
    # Neo4j DB
    draw_graph_network_storage(90, 25, scale=1.2)
    
    # --- Connecting the dots (Logic Flow Annotations) ---
    
    # Fingerprint check annotation
    ax.annotate("MD5 Check", xy=(35, 55), xytext=(35, 65),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='#888'),
                ha='center', fontsize=9, bbox=dict(boxstyle="round", fc="#EEE"))

    # Output annotation
    ax.text(90, 50, "Dual-Path\nRetrieval Ready!", ha='center', va='center', 
            fontsize=12, fontweight='bold', color=colors['highlight'], rotation=-10)
    ax.annotate("", xy=(90, 65), xytext=(90, 55), arrowprops=dict(arrowstyle="->", color=colors['highlight'], ls='dashed'))
    ax.annotate("", xy=(90, 35), xytext=(90, 45), arrowprops=dict(arrowstyle="->", color=colors['highlight'], ls='dashed'))

    # Save
    plt.tight_layout()
    plt.savefig('rag_data_pipeline_rich_cute.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Rich cute diagram saved to rag_data_pipeline_rich_cute.png")

if __name__ == "__main__":
    create_rich_cute_diagram()
