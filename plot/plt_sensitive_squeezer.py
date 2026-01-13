import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import os
import matplotlib.patheffects as patheffects

edge_width = 0.1
plt.rc('font', family=plt.rcParams["font.family"][0], weight='medium')
plt.rcParams['hatch.linewidth'] = edge_width

colors = ['Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'Wistia']
hatches = ['x', 'o', '*', '//', '\\\\', '+']
line_styles = ['-', '-.', ':', '--']

def get_sorted_folders(path):

    folder_names = [
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]

    sorted_folders = sorted(folder_names)
    return sorted_folders

def get_max_length(file, bs):
    cur_idx = 0
    iter_lengths = []
    _lengths = []
    for line in open(file):
        data = json.loads(line)
        _lengths.extend(data['response_lengths'])
        cur_idx += 1
        if cur_idx % bs == 0:
            iter_lengths.append(np.max(_lengths))
            _lengths = []
    return iter_lengths

context_lengths = [8]
batch_sizes = [64]
base_n = 8

configs=[
    (1.0, 8),
    (1.0, 9),
    (1.0, 10),
    (1.0, 12),
    (1.125, 8),
    (1.25, 8),
    (1.5, 8),
    (1.25, 10)
]

widths = [0.3, 0.3, 0.3]
width_ratios = [4, 4, 1]

config_names = [
    "Fixed P", 
    "Fixed N",
    "Ours"
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={'width_ratios': width_ratios})

for global_idx, length in enumerate(context_lengths):
    batch_size = batch_sizes[global_idx]
    source_folder = f"../output/profiler/prompt_squeezer_ultra_context{length}k"
    exp_runs = get_sorted_folders(source_folder)
    if len(exp_runs) > 8:
        exp_runs.remove(exp_runs[-1]) # redundant run


    avg_iter_times = []

    for i, (p, n) in enumerate(configs):
        if i not in range(len(exp_runs)):
            continue
        short_iters_per_round = int(1 / (p - 1)) if not p == 1 else 1
        long_iters_per_round = 1 if not p == 1 else 0

        metric_file = f"{source_folder}/{exp_runs[i]}/get_batch/time-iterations.jsonl"
        iteration_times = []
        with open(metric_file, 'r', encoding='utf-8') as f:
            for line in f:
                log_data = json.loads(line)
                iteration_times.append(log_data['duration'])

        max_runs = len(iteration_times)

        full_iters = int(max_runs / (short_iters_per_round + long_iters_per_round))

        total_runs = full_iters * (short_iters_per_round + long_iters_per_round)

        total_shorts = []
        total_longs = []

        for _iter in range(total_runs):
            if long_iters_per_round > 0 and (_iter + 1) % (short_iters_per_round + long_iters_per_round) == 0:
                total_longs.append(iteration_times[_iter])
            else:
                total_shorts.append(iteration_times[_iter])

        average_iter_time = None
        if long_iters_per_round > 0:
            average_iter_time = (np.max(total_shorts) * short_iters_per_round + np.max(total_longs) * long_iters_per_round) / (short_iters_per_round + long_iters_per_round)
        else:
            average_iter_time = np.max(total_shorts)
    
        avg_iter_times.append(average_iter_time)

    normalized_iter_times = [_t * 100 / np.max(avg_iter_times) for _t in avg_iter_times]

    plt_idxes = [
        [idx for (idx, config) in enumerate(configs) if config[0] == 1.0],
        [idx for (idx, config) in enumerate(configs) if config[1] == 8],
        [len(configs) - 1]
    ]

    for ax_idx, (plt_idx, ax) in enumerate(zip(plt_idxes, axes)):
        width = widths[ax_idx]
        iter_times = [normalized_iter_times[i] if i < len(normalized_iter_times) else np.nan for i in plt_idx]
        ax.grid(True, axis='y', alpha=0.5)
        bars = ax.bar(
            np.arange(len(iter_times)) - width + global_idx * width, 
            iter_times, 
            width=width, 
            color=sns.color_palette(colors[global_idx])[2], 
            hatch=hatches[global_idx],
            edgecolor='black',
            linewidth=edge_width,
            label=f'Length={length}K'
        )

        actual_times = [int(avg_iter_times[i]) if i < len(avg_iter_times) else 0 for i in plt_idx]
        bar_label_texts = ax.bar_label(bars, labels=[f'{t}' for t in actual_times], padding=1.5, size=20)

        for text in bar_label_texts:
            text.set_rotation(90)
        
        for bar in bars:
            bar.set_zorder(2)      


       
        if global_idx == len(context_lengths) - 1:
            ax.tick_params(axis='both', which='major', labelsize=23)
            if ax_idx == 0:
                ax.set_ylabel("Norm. Time", size=23, weight='bold')
            else:
                for tick in ax.get_yticklabels():
                    tick.set_visible(False)
            
            ax.text(
                0.5, -0.18, config_names[ax_idx], 
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=23,
                weight='bold'
            )

            ax.set_ylim(0, 130)
            if ax_idx == len(widths) - 1:
                ax.set_xlim(-0.52, 0.52)
            ax.set_xticks(range(len(iter_times)))
            if configs[i][1] == base_n and float(configs[i][1]) > 1:
                ax.set_xticklabels([float(configs[i][0]) for i in plt_idx])
            else:
                ax.set_xticklabels([float(configs[i][1]/base_n) for i in plt_idx])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, 
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.06), 
    frameon=False, ncol=3, 
    prop={
        'weight': 'bold',
        'size': 22
    },
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"./img/prompt_suqeezer_diff_config.png")