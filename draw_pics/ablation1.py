import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
iters = ['Iter0', 'Iter1', 'Iter2']
x = np.arange(len(iters))

ablation_data = {
    r'CTO: $\alpha_0$': {
        'params': ['0.3', '0.55', '0.8'],
        'values': [
            [43.3, 57.0, 57.9],
            [43.3, 60.2, 60.3],       # default ablation curve
            [43.3, 58.5, 58.8],  
        ],
        'default_idx': 1,
        'main_mean': [43.2, 60.0, 62.2],
        'main_std':  [0.0,  3.1,  3.2],
    },
    r'CTO: $K_{\mathrm{CTO}}$': {
        'params': ['4', '8', '16'],
        'values': [
            [43.3, 62.7, 61.6],
            [43.3, 60.2, 60.3],       # default ablation curve
            [43.3, 53.5, 56.3],
        ],
        'default_idx': 1,
        'main_mean': [43.2, 60.0, 62.2],
        'main_std':  [0.0,  3.1,  3.2],
    },
    r'RG-CTO: $\tau_{\mathrm{match}}$': {
        'params': ['0.7', '0.8', '0.9'],
        'values': [
            [43.3, 57.9, 59.6],
            [43.3, 61.3, 62.7],       # default ablation curve
            [43.3, 58.0, 56.5],
        ],
        'default_idx': 1,
        'main_mean': [43.2, 60.5, 61.9],
        'main_std':  [0.0,  1.1,  1.2],
    },
    r'RG-CTO: $\lambda_u$': {
        'params': ['0.25', '0.5', '0.75'],
        'values': [
            [43.3, 52.1, 55.1],
            [43.3, 61.3, 62.7],       # default ablation curve
            [43.3, 57.4, 57.6],
        ],
        'default_idx': 1,
        'main_mean': [43.2, 60.5, 61.9],
        'main_std':  [0.0,  1.1,  1.2],
    },
    r'RG-CTO: $\lambda_l$': {
        'params': ['0.25', '0.5', '0.75'],
        'values': [
            [43.3, 56.2, 61.0],
            [43.3, 61.3, 62.7],       # default ablation curve
            [43.3, 57.8, 58.5],
        ],
        'default_idx': 1,
        'main_mean': [43.2, 60.5, 61.9],
        'main_std':  [0.0,  1.1,  1.2],
    },
    r'RG-CTO: $\delta$': {
        'params': ['0.2', '0.4', '0.6'],
        'values': [
            [43.3, 63.6, 62.3],
            [43.3, 61.3, 62.7],       # default ablation curve
            [43.3, 57.9, 56.9],
        ],
        'default_idx': 1,
        'main_mean': [43.2, 60.5, 61.9],
        'main_std':  [0.0,  1.1,  1.2],
    },
}

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
axes = axes.flatten()

for ax, (title, d) in zip(axes, ablation_data.items()):
    params = d['params']
    values = d['values']
    default_idx = d['default_idx']

    lines = []

    # Plot ablation curves
    for idx, (p, y) in enumerate(zip(params, values)):
        lw = 2.8 if idx == default_idx else 1.8
        ms = 7 if idx == default_idx else 5

        line, = ax.plot(
            x, y,
            marker='o',
            linewidth=lw,
            markersize=ms,
            label=f'{p}' + (' (default)' if idx == default_idx else '')
        )
        lines.append(line)

    # Main-table reference band: mean ± std for default setting
    main_mean = np.array(d['main_mean'], dtype=float)
    main_std = np.array(d['main_std'], dtype=float)
    lower = main_mean - main_std
    upper = main_mean + main_std

    default_color = lines[default_idx].get_color()

    ax.fill_between(
        x, lower, upper,
        color=default_color,
        alpha=0.15,
        label=r'default $\pm$ 1 std',
        zorder=0
    )

    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(iters, fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_ylabel('Pass@1 (%)', fontsize=14)
    ax.set_ylim(40, 66)
    ax.grid(True, linestyle='--', alpha=0.35)

    ax.legend(
        loc='lower right',
        fontsize=11,
        frameon=True
    )

plt.tight_layout()
plt.savefig('ablation_iter_xaxis_with_band.png', dpi=300, bbox_inches='tight')
plt.savefig('ablation_iter_xaxis_with_band.pdf', bbox_inches='tight')
plt.show()