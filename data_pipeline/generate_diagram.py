import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def create_rag_pipeline_diagram():
    # Setup figure
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Style configurations
    box_props = dict(boxstyle='round,pad=1', facecolor='white', edgecolor='#333333', linewidth=2)
    process_props = dict(boxstyle='round,pad=0.8', facecolor='#E6F3FF', edgecolor='#0066CC', linewidth=2)
    storage_props = dict(boxstyle='round,pad=0.8', facecolor='#FFF2CC', edgecolor='#D6B656', linewidth=2)
    db_props = dict(boxstyle='cylinder,pad=0.5', facecolor='#E2F0CB', edgecolor='#82B366', linewidth=2) # Matplotlib doesn't have cylinder, using round
    
    # Helper to draw box
    def draw_box(x, y, width, height, text, style='process', fontsize=10):
        if style == 'input':
            color = '#F5F5F5'
            edge = '#666666'
        elif style == 'process':
            color = '#E6F3FF' # Light Blue
            edge = '#0066CC'
        elif style == 'algo':
            color = '#DAE8FC' # Lighter Blue
            edge = '#6C8EBF'
        elif style == 'db':
            color = '#D5E8D4' # Light Green
            edge = '#82B366'
        elif style == 'model':
            color = '#FFE6CC' # Light Orange
            edge = '#D79B00'
        
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2", 
                                      linewidth=2, edgecolor=edge, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='#333333')
        return x + width/2, y + height/2, x + width/2, y, x + width, y + height/2, x, y + height/2

    # Helper to draw arrow
    def draw_arrow(x1, y1, x2, y2, text=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#555555', lw=2, connectionstyle="arc3,rad=0"))
        if text:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y + 1, text, ha='center', va='center', fontsize=9, backgroundcolor='white')

    # --- Layout ---

    # 1. Input Section
    draw_box(40, 90, 20, 6, "Academic Papers\n(PDF/MD/TXT)", 'input', 12)
    
    # Arrow down
    draw_arrow(50, 90, 50, 84)
    
    # 2. Preprocessing
    draw_box(35, 76, 30, 8, "Document Loader\n& Preprocessing\n(Cleaning/LaTeX Norm)", 'process')
    
    # Arrow down
    draw_arrow(50, 76, 50, 70)
    
    # 3. Intelligent Splitter (Key Component)
    # Drawing a container box for the splitter
    splitter_rect = patches.FancyBboxPatch((20, 45), 60, 25, boxstyle="round,pad=0.5", 
                                          linewidth=2, edgecolor='#9673A6', facecolor='#E1D5E7', alpha=0.3)
    ax.add_patch(splitter_rect)
    ax.text(50, 68, "Photovoltaic Text Splitter\n(Intelligent Layered)", ha='center', va='center', fontsize=11, fontweight='bold', color='#444444')
    
    # Section Recognition
    draw_box(40, 60, 20, 5, "Section Recognition\n(Regex Patterns)", 'algo')
    
    # Arrow from Preprocessing to Splitter
    draw_arrow(50, 70, 50, 65) # Adjusted to hit container top
    
    # Arrows to diff strategies
    draw_arrow(50, 60, 30, 55)
    draw_arrow(50, 60, 50, 55)
    draw_arrow(50, 60, 70, 55)
    
    # Differential Strategies
    draw_box(22, 48, 16, 5, "Abstract/Conclusion\n(Small Chunk)", 'algo', 8)
    draw_box(42, 48, 16, 5, "Methodology/Fab\n(Medium Chunk)", 'algo', 8)
    draw_box(62, 48, 16, 5, "Results/Tables\n(Large Chunk)", 'algo', 8)
    
    # 4. Split Point
    # Output of splitter goes to two paths
    
    # Path A: Vector DB (Left side)
    draw_arrow(30, 48, 25, 40)
    draw_arrow(50, 48, 25, 40) # Converge
    
    draw_box(10, 32, 30, 6, "Chunk Metadata Enricher\n(Semantic Tagging)", 'process')
    draw_arrow(25, 32, 25, 26)
    
    draw_box(10, 18, 30, 6, "Embedding Generator\n(OpenAI/HuggingFace)", 'model')
    draw_arrow(25, 18, 25, 12)
    
    draw_box(10, 4, 30, 6, "Milvus Vector DB\n(Vector Storage)", 'db')
    
    # Path B: Knowledge Graph (Right side)
    draw_arrow(50, 48, 75, 40)
    draw_arrow(70, 48, 75, 40) # Converge
    
    draw_box(60, 32, 30, 6, "KG Extractor (LLM)\n(Entity/Relation Extraction)", 'model')
    draw_arrow(75, 32, 75, 26)
    
    draw_box(60, 18, 30, 6, "Entity Normalizer\n(Deduplication/Merging)", 'process')
    draw_arrow(75, 18, 75, 12)
    
    draw_box(60, 4, 30, 6, "Neo4j Graph DB\n(Knowledge Storage)", 'db')

    # Connection lines for context
    # Maybe add a "Change Detection" note near Document Loader or Vector Manager
    ax.text(85, 76, "Change Detection:\nMD5 Fingerprints", ha='left', va='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='#FFF2CC'))
    draw_arrow(84, 76, 65, 78, "") # Pointing roughly to loader area

    # Title
    plt.title("RAGPhoto Agentic Data Pipeline Architecture", fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        patches.Patch(facecolor='#E6F3FF', edgecolor='#0066CC', label='Process'),
        patches.Patch(facecolor='#DAE8FC', edgecolor='#6C8EBF', label='Algorithm/Logic'),
        patches.Patch(facecolor='#FFE6CC', edgecolor='#D79B00', label='Model/LLM'),
        patches.Patch(facecolor='#D5E8D4', edgecolor='#82B366', label='Database'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig('rag_data_pipeline_schematic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Diagram saved to rag_data_pipeline_schematic.png")

if __name__ == "__main__":
    create_rag_pipeline_diagram()
