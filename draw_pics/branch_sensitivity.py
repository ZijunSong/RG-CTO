import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
iters = ['iter0', 'iter1', 'iter2']
methods = ['Neg-only', 'Global-Neg', 'RSE-64', 'RSE-32', 'Pos-only', 'CTO-base']
colors = ['#1f77b4', '#2ca02c', '#8c564b', '#d62728', '#ff7f0e', '#9467bd']

data = np.array([
    [43.2, 43.2, 43.2, 43.2, 43.2, 43.2],   # iter0
    [43.0, 58.5, 58.0, 58.3, 59.8, 60.0],   # iter1
    [44.7, 55.0, 58.6, 59.1, 61.3, 62.2],   # iter2
])

# -----------------------------
# Plot
# -----------------------------
n_iters, n_methods = data.shape
x = np.arange(n_iters)
bar_width = 0.12
offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_width

fig, ax = plt.subplots(figsize=(8.8, 4.8))

for i, (method, color) in enumerate(zip(methods, colors)):
    xs = x + offsets[i]
    bars = ax.bar(xs, data[:, i], width=bar_width * 0.92, color=color, label=method, zorder=3)
    for bar, val in zip(bars, data[:, i]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.15,
            f'{val:.1f}',
            ha='center',
            va='center',
            fontsize=11,
            rotation=30,
        )

ax.set_xlabel('Iteration', fontsize=14)
ax.set_ylabel('Pass@1 (%)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(iters, fontsize=13)
ax.tick_params(axis='both', labelsize=13)
ax.set_ylim(38, 67)
ax.set_yticks(np.arange(40, 66, 5))
ax.yaxis.grid(True, linestyle='--', color='0.75', zorder=0)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

ax.legend(
    loc='upper left',
    ncol=2,
    fontsize=11,
    frameon=True,
    edgecolor='0.8',
)

plt.tight_layout()
plt.savefig('branch_sensitivity.png', dpi=300, bbox_inches='tight')
plt.savefig('branch_sensitivity.pdf', bbox_inches='tight')
print('Saved: branch_sensitivity.png / branch_sensitivity.pdf')
