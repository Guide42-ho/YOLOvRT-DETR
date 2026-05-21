import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ─── Load data ───────────────────────────────────────────────────────────────
df_yolo   = pd.read_csv(os.path.join(BASE, 'cm_yolo.csv'))
df_rtdetr = pd.read_csv(os.path.join(BASE, 'cm_rtdetr.csv'))

# CSV: rows = Ground Truth, cols = Predicted
# Goal: X-axis = Ground Truth, Y-axis = Predicted
# imshow[row, col] = imshow[y, x]
# → need matrix[pred_i, gt_j]  = original[gt_j, pred_i]  → use .T

labels = ['label_damage', 'label_normal', 'background']

def load_matrix(df):
    return df[labels].values.astype(int).T   # [pred, gt]

cm_yolo   = load_matrix(df_yolo)
cm_rtdetr = load_matrix(df_rtdetr)

# ─── Colormap ────────────────────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    'dark_blue',
    ['#ffffff', '#cce0f5', '#6aaed6', '#2171b5', '#08306b'],
    N=256
)

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('white')

configs = [('YOLO', cm_yolo, '(a)'), ('RT-DETR', cm_rtdetr, '(b)')]

for ax, (title, cm, panel) in zip(axes, configs):
    ax.set_facecolor('white')
    vmax = cm.max()
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)   # X = Ground Truth
    ax.set_yticklabels(labels, fontsize=10)   # Y = Predicted

    for pred_i in range(len(labels)):
        for gt_j in range(len(labels)):
            val = cm[pred_i, gt_j]
            bg_norm = val / (vmax if vmax > 0 else 1)
            color = 'white' if bg_norm > 0.4 else '#1a1a2e'
            ax.text(gt_j, pred_i, str(val),
                    ha='center', va='center',
                    fontsize=13, fontweight='bold', color=color)

    ax.set_xlabel('Ground Truth', fontsize=11, labelpad=8)
    ax.set_ylabel('Predicted',    fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.text(0.5, -0.18, panel, transform=ax.transAxes,
            ha='center', va='top', fontsize=12)

plt.tight_layout(pad=2.5)
out_path = os.path.join(BASE, 'confusion_matrices.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
plt.close()