import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

# Canvas
FW, FH = 14, 8
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
fig.patch.set_facecolor('white')

# Colors
C_SKIP = '#d4edda'; C_SKIP_E = '#1a7a34'  # Green
C_DEC  = '#cce0ff'; C_DEC_E  = '#1a4a8a'  # Blue
C_PROC = '#f8f9fa'; C_PROC_E = '#343a40'  # Light Gray
C_ATTN = '#e0fbff'; C_ATTN_E = '#008b8b'  # Cyan/Teal
C_MAP  = '#fff3cd'; C_MAP_E  = '#856404'  # Gold/Yellow

def rbox(ax, x, y, w, h, fc, ec, text, fs=13, bold=False):
    p = FancyBboxPatch((x-w/2, y-h/2), w, h,
                       boxstyle="round,pad=0,rounding_size=0.15",
                       linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            color='#111111', fontweight='bold' if bold else 'normal', zorder=4)

def circle_op(ax, x, y, symbol, fc='#ffffff', ec='#333333', size=0.32):
    c = Circle((x, y), size, linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, symbol, ha='center', va='center', fontsize=18, fontweight='bold', zorder=6)

def arr(ax, x1, y1, x2, y2, c='#333333', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->,head_width=0.3,head_length=0.2',
                                color=c, lw=lw), zorder=4)

# Layout Variables
Y_TOP = 6.5
Y_MID = 3.5

# 1. Inputs (Left)
rbox(ax, 1.5, Y_TOP, 1.8, 0.8, C_SKIP, C_SKIP_E, 'Skip features\n$f_l$', bold=True)
rbox(ax, 1.5, Y_MID, 1.8, 0.8, C_DEC,  C_DEC_E,  'Gating signal\n$g$', bold=True)

# 2. Branching and Processing
# Branching point for f_l
bx = 3.2
ax.plot([2.4, bx], [Y_TOP, Y_TOP], color='#333333', lw=1.5, zorder=2)
# Branch 1: to Multiplier (horizontal)
# Branch 2: to W_f (down)
ax.plot([bx, bx, 4.0], [Y_TOP, Y_TOP-1.2, Y_TOP-1.2], color='#333333', lw=1.5, zorder=2)
arr(ax, bx, Y_TOP-1.2, 4.0, Y_TOP-1.2)
rbox(ax, 4.8, Y_TOP-1.2, 1.5, 0.6, C_PROC, C_PROC_E, '$W_f$ (1×1)')

# Gating signal processing
rbox(ax, 4.8, Y_MID, 1.5, 0.6, C_PROC, C_PROC_E, '$W_g$ (1×1)')
arr(ax, 2.4, Y_MID, 4.05, Y_MID)

# 3. Addition
circle_op(ax, 6.8, Y_MID, '+')
arr(ax, 5.55, Y_MID, 6.48, Y_MID)
# From W_f down to +
ax.plot([5.55, 6.8, 6.8], [Y_TOP-1.2, Y_TOP-1.2, Y_MID+0.32], color='#333333', lw=1.5, zorder=2)
arr(ax, 6.8, Y_TOP-1.2, 6.8, Y_MID+0.32)

# 4. Activation and Map
rbox(ax, 8.8, Y_MID, 1.3, 0.6, C_PROC, C_PROC_E, 'ReLU')
arr(ax, 7.12, Y_MID, 8.15, Y_MID)

rbox(ax, 11.2, Y_MID, 2.0, 0.8, C_MAP, C_MAP_E, '$W_\psi$ (1×1)\nSigmoid ($\\psi$)')
arr(ax, 9.45, Y_MID, 10.2, Y_MID)

# 5. Multiplication
circle_op(ax, 11.2, Y_TOP, '×')
arr(ax, 11.2, Y_MID+0.4, 11.2, Y_TOP-0.32)

# Connection from branching bx to multiply
arr(ax, bx, Y_TOP, 10.88, Y_TOP)

# 6. Final Output
rbox(ax, 13.0, Y_TOP, 1.5, 0.8, C_ATTN, C_ATTN_E, 'Refined\n$\\hat{f}_l$', bold=True)
arr(ax, 11.52, Y_TOP, 12.25, Y_TOP)

# Title and Formula
ax.text(FW/2, 7.6, 'Soft Spatial Attention Mechanism (SSA)', 
        ha='center', fontsize=18, fontweight='bold')
ax.text(FW/2, 0.6, '$\\hat{f}_l = f_l \\cdot \\sigma(W_\\psi^T (\\sigma_{\\mathrm{ReLU}}(W_f^T f_l + W_g^T g)))$', 
        ha='center', fontsize=16, color='#222222', math_fontfamily='cm')

plt.savefig(r'd:\project_saoudia_work\realfake2017_Lora\ssa_mechanism.png', 
            dpi=250, bbox_inches='tight', facecolor='white')
print("SSA Figure saved!")
plt.close()
