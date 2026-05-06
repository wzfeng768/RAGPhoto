import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

def create_cute_diagram():
    # Use a style that mimics hand-drawn if xkcd is available, otherwise just standard with customizations
    try:
        plt.xkcd()
    except:
        pass # Fallback to standard if xkcd font/style not available

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Colors
    c_paper = '#FFFFFF'
    c_paper_line = '#333333'
    c_scissors = '#FF6B6B'
    c_vector = '#4ECDC4'
    c_graph = '#FFD93D'
    c_db_vec = '#45B7D1'
    c_db_graph = '#96CEB4'
    c_robot = '#A8D8EA'

    # --- Helper Functions for "Cute" Icons ---

    def draw_cute_doc(x, y, scale=1.0, label=""):
        # Paper body
        w, h = 6*scale, 8*scale
        rect = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1", 
                                      facecolor=c_paper, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        # Lines representing text
        for i in range(3):
            ax.plot([x-w/3, x+w/3], [y+h/4 - i*h/4, y+h/4 - i*h/4], color='#888888', lw=2)
        # Label
        if label:
            ax.text(x, y-h/2-2, label, ha='center', va='top', fontsize=12, fontweight='bold')

    def draw_stack_docs(x, y, scale=1.0, label="Docs"):
        # Draw a few documents behind
        draw_cute_doc(x+1*scale, y+1*scale, scale)
        draw_cute_doc(x-1*scale, y-0.5*scale, scale)
        draw_cute_doc(x, y, scale, label)

    def draw_cute_scissors(x, y, scale=1.0, label=""):
        # Handles (circles)
        c1 = patches.Circle((x-2*scale, y-2*scale), 1*scale, facecolor=c_scissors, edgecolor='black', lw=2)
        c2 = patches.Circle((x+2*scale, y-2*scale), 1*scale, facecolor=c_scissors, edgecolor='black', lw=2)
        ax.add_patch(c1)
        ax.add_patch(c2)
        # Blades
        path1 = Path([(x-2*scale, y-1*scale), (x+3*scale, y+4*scale)], [Path.MOVETO, Path.LINETO])
        patch1 = patches.PathPatch(path1, edgecolor='silver', lw=4*scale)
        path2 = Path([(x+2*scale, y-1*scale), (x-3*scale, y+4*scale)], [Path.MOVETO, Path.LINETO])
        patch2 = patches.PathPatch(path2, edgecolor='silver', lw=4*scale)
        ax.add_patch(patch1)
        ax.add_patch(patch2)
        # Pivot
        pivot = patches.Circle((x, y+0.5*scale), 0.3*scale, facecolor='black')
        ax.add_patch(pivot)
        if label:
            ax.text(x, y-4*scale, label, ha='center', va='top', fontsize=12, fontweight='bold')

    def draw_cute_robot(x, y, scale=1.0, label=""):
        # Head
        w, h = 6*scale, 5*scale
        head = patches.Rectangle((x-w/2, y-h/2), w, h, facecolor=c_robot, edgecolor='black', lw=2, joinstyle='round')
        ax.add_patch(head)
        # Eyes
        e1 = patches.Circle((x-1.5*scale, y+0.5*scale), 0.8*scale, facecolor='white', edgecolor='black', lw=1.5)
        e2 = patches.Circle((x+1.5*scale, y+0.5*scale), 0.8*scale, facecolor='white', edgecolor='black', lw=1.5)
        ax.add_patch(e1)
        ax.add_patch(e2)
        # Pupils
        p1 = patches.Circle((x-1.5*scale, y+0.5*scale), 0.3*scale, facecolor='black')
        p2 = patches.Circle((x+1.5*scale, y+0.5*scale), 0.3*scale, facecolor='black')
        ax.add_patch(p1)
        ax.add_patch(p2)
        # Antenna
        ax.plot([x, x], [y+h/2, y+h/2+2*scale], color='black', lw=2)
        ax.add_patch(patches.Circle((x, y+h/2+2*scale), 0.5*scale, color='red'))
        if label:
            ax.text(x, y-h/2-2, label, ha='center', va='top', fontsize=12, fontweight='bold')

    def draw_cute_vector_grid(x, y, scale=1.0, label=""):
        # A grid representing vectors
        w, h = 6*scale, 6*scale
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, facecolor=c_vector, edgecolor='black', lw=2)
        ax.add_patch(rect)
        # Grid lines
        for i in range(1, 3):
            ax.plot([x-w/2 + i*w/3, x-w/2 + i*w/3], [y-h/2, y+h/2], color='white', lw=1.5)
            ax.plot([x-w/2, x+w/2], [y-h/2 + i*h/3, y-h/2 + i*h/3], color='white', lw=1.5)
        # Numbers
        ax.text(x-w/6, y+h/6, "0.1", ha='center', va='center', fontsize=6*scale, color='white')
        ax.text(x+w/6, y-h/6, "0.9", ha='center', va='center', fontsize=6*scale, color='white')
        if label:
            ax.text(x, y-h/2-2, label, ha='center', va='top', fontsize=12, fontweight='bold')

    def draw_cute_graph_nodes(x, y, scale=1.0, label=""):
        # Nodes
        pos = [(0, 2), (-2, -1), (2, -1)]
        for dx, dy in pos:
            c = patches.Circle((x+dx*scale, y+dy*scale), 1.2*scale, facecolor=c_graph, edgecolor='black', lw=2)
            ax.add_patch(c)
        # Edges
        ax.plot([x, x-2*scale], [y+2*scale, y-1*scale], color='black', lw=2, zorder=0)
        ax.plot([x, x+2*scale], [y+2*scale, y-1*scale], color='black', lw=2, zorder=0)
        ax.plot([x-2*scale, x+2*scale], [y-1*scale, y-1*scale], color='black', lw=2, zorder=0)
        if label:
            ax.text(x, y-3*scale, label, ha='center', va='top', fontsize=12, fontweight='bold')

    def draw_cute_db_cylinder(x, y, color, scale=1.0, label=""):
        w, h = 6*scale, 7*scale
        # Body
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, facecolor=color, edgecolor='black', lw=2)
        ax.add_patch(rect)
        # Top oval
        top = patches.Ellipse((x, y+h/2), w, 2*scale, facecolor=color, edgecolor='black', lw=2)
        ax.add_patch(top)
        # Bottom oval (half visible)
        bottom = patches.Arc((x, y-h/2), w, 2*scale, theta1=180, theta2=360, edgecolor='black', lw=2)
        ax.add_patch(bottom)
        if label:
            ax.text(x, y-h/2-3, label, ha='center', va='top', fontsize=12, fontweight='bold')

    def draw_arrow_curve(x1, y1, x2, y2, rad=0.2):
        style = f"Simple,tail_width=0.5,head_width=4,head_length=8"
        kw = dict(arrowstyle=style, color="gray")
        patch = patches.FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}", **kw)
        ax.add_patch(patch)

    # --- Drawing the Scene ---

    # 1. Input: Docs
    draw_stack_docs(15, 50, scale=1.5, label="Input Papers\n(PDF/MD)")

    # Arrow to Splitter
    draw_arrow_curve(22, 50, 32, 50, rad=0)

    # 2. Process: Scissors (Splitter)
    draw_cute_scissors(38, 50, scale=1.2, label="Intelligent\nSplitter")
    
    # Text bubbles for splitting strategies
    ax.text(38, 62, "Abstract\n(Small)", ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='#FFF', edgecolor='#CCC'))
    ax.text(38, 38, "Results\n(Large)", ha='center', va='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='#FFF', edgecolor='#CCC'))

    # Split to two paths
    
    # Path Top: Vector
    draw_arrow_curve(44, 55, 55, 75, rad=-0.2)
    
    draw_cute_robot(60, 75, scale=1.2, label="AI Embedding")
    
    draw_arrow_curve(66, 75, 78, 75, rad=0)
    
    draw_cute_vector_grid(84, 75, scale=1.2, label="Vector Data")
    
    draw_arrow_curve(90, 75, 90, 65, rad=-0.3) # Down to DB
    
    # Path Bottom: KG
    draw_arrow_curve(44, 45, 55, 25, rad=0.2)
    
    draw_cute_robot(60, 25, scale=1.2, label="LLM Extraction")
    # Change robot color for KG to distinguish
    # (Re-drawing robot head with different color manually or just rely on label)
    
    draw_arrow_curve(66, 25, 78, 25, rad=0)
    
    draw_cute_graph_nodes(84, 25, scale=1.2, label="Knowledge\nGraph")
    
    draw_arrow_curve(90, 25, 90, 35, rad=0.3) # Up to DB

    # 3. Storage: Two DBs side by side or merged? Let's show two distinct stores
    # Since layout is Top/Bottom, let's put DBs at the end of their lines
    
    # Redraw arrows to point to specific DBs
    
    # Top DB (Vector)
    draw_cute_db_cylinder(85, 65, c_db_vec, scale=1.3, label="Milvus\n(Vectors)")
    # Re-route arrow
    
    # Bottom DB (Graph)
    draw_cute_db_cylinder(85, 35, c_db_graph, scale=1.3, label="Neo4j\n(Relations)")

    # Title
    ax.text(50, 92, "RAGPhoto Data Pipeline", ha='center', va='center', fontsize=20, fontweight='bold', fontname='sans-serif')
    ax.text(50, 88, "(Cute Schematic)", ha='center', va='center', fontsize=12, color='gray')

    # Decorative elements
    # Sparkles
    for _ in range(5):
        sx, sy = np.random.uniform(10, 90), np.random.uniform(10, 90)
        ax.text(sx, sy, "✨", fontsize=14, alpha=0.6)

    plt.tight_layout()
    plt.savefig('rag_data_pipeline_cute.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Cute diagram saved to rag_data_pipeline_cute.png")

if __name__ == "__main__":
    create_cute_diagram()
