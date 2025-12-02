import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

def draw_professional_framework():
    # Setup figure with the reference image's background color
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Color Palette based on the reference image
    bg_color = '#EEF5F9'  # Very light blue background
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    box_fill = '#FFFFFF'
    box_edge = '#B0BEC5' # Blue-grey border
    header_bg = '#E1F5FE' # Light blue header background for emphasis
    text_main = '#263238' # Dark blue-grey text
    text_sub = '#455A64'  # Lighter grey text
    accent_blue = '#0277BD' # Stronger blue for titles

    # Main Title
    plt.text(6, 7.5, 'Evaluation Framework', ha='center', va='center', fontsize=24, fontweight='bold', color=text_main)

    # Helper function to draw a styled module
    def draw_module(x, y, width, height, title, color_code, items, is_wide=False):
        # Shadow effect (simple offset)
        shadow = patches.FancyBboxPatch((x+0.05, y-0.05), width, height, boxstyle="round,pad=0.1", 
                                      linewidth=0, facecolor='#CFD8DC', zorder=1)
        ax.add_patch(shadow)
        
        # Main Box
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                    linewidth=1.5, edgecolor=box_edge, facecolor=box_fill, zorder=2)
        ax.add_patch(rect)
        
        # Decorative Icon/Shape (Circle)
        # Position circle to the left of the title
        circle = patches.Circle((x + width/2 - 2.2, y + height - 0.5), radius=0.15, color=color_code, zorder=3)
        # Adjust position based on title length is hard, so let's center the title and put a colored bar above it or a dot next to it.
        # Let's try a simple colored bar on the left side of the box for a "card" look
        left_bar = patches.Rectangle((x, y), 0.15, height, linewidth=0, facecolor=color_code, zorder=4)
        # Clip the bar to the rounded box is hard with simple patches.
        # Let's stick to a colored dot next to the title.
        
        # Re-calculate title position to center it with the dot
        # We'll just put the dot and title centered together roughly
        
        ax.add_patch(patches.Circle((x + 0.5, y + height - 0.5), radius=0.12, color=color_code, zorder=3))
        
        # Title
        plt.text(x + 0.8, y + height - 0.5, title, ha='left', va='center', 
                 fontsize=14, fontweight='bold', color=accent_blue, zorder=3)
        
        # Divider
        plt.plot([x + 0.2, x + width - 0.2], [y + height - 0.8, y + height - 0.8], 
                 color='#ECEFF1', linewidth=2, zorder=3)
        
        # Content
        text_start_y = y + height - 1.2
        line_height = 0.4
        
        if is_wide:
            content_text = ",  ".join(items)
            import textwrap
            wrapped_lines = textwrap.wrap(content_text, width=60)
            for i, line in enumerate(wrapped_lines):
                plt.text(x + width/2, text_start_y - i*line_height, line, ha='center', va='top', 
                         fontsize=12, color=text_sub, zorder=3)
        else:
            for i, item in enumerate(items):
                plt.text(x + 0.3, text_start_y - i*line_height, f"• {item}", ha='left', va='top', 
                         fontsize=11, color=text_sub, zorder=3)

    # --- Layout Configuration ---
    
    # Top Row: 3 Metrics Columns
    col_width = 3.5
    col_height = 4.0
    gap = 0.4
    start_x = (12 - (3 * col_width + 2 * gap)) / 2
    y_pos = 2.8

    # 1. Relevance Metrics (Red/Orange accent)
    draw_module(start_x, y_pos, col_width, col_height, 
                "Relevance Metrics", "#FF7043", 
                ["Recall@k", "MRR", "nDCG"])

    # 2. Reasoning Metrics (Blue accent)
    draw_module(start_x + col_width + gap, y_pos, col_width, col_height, 
                "Reasoning Metrics", "#42A5F5", 
                ["Multi-Hop Coverage (MHC)", "Evidence Completeness (ECS)", "Reasoning Accuracy (RTA)"])

    # 3. Safety Metrics (Green accent)
    draw_module(start_x + 2 * (col_width + gap), y_pos, col_width, col_height, 
                "Safety Metrics", "#66BB6A", 
                ["Relevance", "Factuality", "Evidence Sufficiency", "Safety Compliance"])

    # Bottom Row: Baselines (Purple accent)
    wide_width = 3 * col_width + 2 * gap
    wide_height = 1.8
    draw_module(start_x, 0.5, wide_width, wide_height, 
                "Baseline Comparisons", "#AB47BC", 
                ["BM25", "BioBERT", "PubMedBERT", "DPR", "TART", "Self-RAG"], is_wide=True)

    # Save
    plt.tight_layout()
    plt.savefig('evaluation_framework_v2.png', dpi=300, bbox_inches='tight', facecolor=bg_color)
    print("Image saved as evaluation_framework_v2.png")

if __name__ == "__main__":
    draw_professional_framework()
