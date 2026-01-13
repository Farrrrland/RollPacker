import os
import glob
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

edge_width = 0.5
plt.rc('font', family=plt.rcParams["font.family"][0], weight='medium')
plt.rcParams['hatch.linewidth'] = edge_width

colors = ['Blues', 'Reds']
line_styles = ['-', '--']

# adaptive timeout, 7B, 8K, batch_size=44, change dataset to code_KodCode_data_with_time_e2e_filtered.jsonl
adaptive_log_dir = os.path.expanduser("~/logs/code_time_test/20250911-211641-0-0.75-44-1/iter_time/")
# no adaptive timeout, 7B, 8K, batch_size=44, change dataset to code_KodCode_data_with_time_e2e_filtered.jsonl
no_adaptive_log_dir = os.path.expanduser("~/logs/code_time_test/20250911-213741-0-0.75-44-0/iter_time/")

def extract_iters_and_durations(log_dir):
    jsonl_files = glob.glob(os.path.join(log_dir, "time_iter*_sum.jsonl"))
    iters = []
    durations = []
    for jsonl_file in jsonl_files:
        basename = os.path.basename(jsonl_file)
        m = re.search(r'time_iter(\d+)_sum\.jsonl', basename)
        if m:
            iter_num = int(m.group(1))
            with open(jsonl_file, "r") as f:
                line = f.readline()
                if line:
                    data = json.loads(line)
                    duration = data.get("duration")
                    if duration is not None:
                        iters.append(iter_num)
                        durations.append(duration)
    if iters:
        sorted_pairs = sorted(zip(iters, durations))
        iters_sorted, durations_sorted = zip(*sorted_pairs)
        iters_sorted = list(iters_sorted)
        durations_sorted = list(durations_sorted)
        if 0 in iters_sorted:
            idx0 = iters_sorted.index(0)
            del iters_sorted[idx0]
            del durations_sorted[idx0]
        if iters_sorted:
            iters_sorted = [0] + iters_sorted
            durations_sorted = [durations_sorted[-1]] + durations_sorted
        if iters_sorted:
            iters_sorted = iters_sorted[:-1]
            durations_sorted = durations_sorted[:-1]
        return list(iters_sorted), list(durations_sorted)
    else:
        return [], []

adaptive_iters, adaptive_durations = extract_iters_and_durations(adaptive_log_dir)
no_adaptive_iters, no_adaptive_durations = extract_iters_and_durations(no_adaptive_log_dir)

common_iters = sorted(set(adaptive_iters) & set(no_adaptive_iters))
if not common_iters:
    raise ValueError("No common iters found in both logs. adaptive_iters: {}, no_adaptive_iters: {}".format(adaptive_iters, no_adaptive_iters))

common_iters = common_iters[:20]

adaptive_iter2dur = dict(zip(adaptive_iters, adaptive_durations))
no_adaptive_iter2dur = dict(zip(no_adaptive_iters, no_adaptive_durations))
adaptive_common_durations = [adaptive_iter2dur[i] for i in common_iters]
no_adaptive_common_durations = [no_adaptive_iter2dur[i] for i in common_iters]

x = np.arange(1, len(common_iters) + 1)

fig, ax = plt.subplots(figsize=(6.8, 3))

# adaptive
adaptive_line, = ax.plot(
    x, adaptive_common_durations,
    color=sns.color_palette(colors[0])[3],
    linewidth=4,
    marker='o',
    markersize=7,
    markeredgewidth=1.5,
    markeredgecolor=sns.color_palette(colors[0])[3],
    markerfacecolor='white',
    linestyle=line_styles[0],
    label='w/   Adpt. Time Limit'
)
# no adaptive
no_adaptive_line, = ax.plot(
    x, no_adaptive_common_durations,
    color=sns.color_palette(colors[1])[3],
    linewidth=4,
    marker='s',
    markersize=7,
    markeredgewidth=1.5,
    markeredgecolor=sns.color_palette(colors[1])[3],
    markerfacecolor='white',
    linestyle=line_styles[1],
    label='w/o Adpt. Time Limit'
)

font_size = 20
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xticklabels(['1', '5', '10', '15', '20'])


ax.set_xlabel("Steps", size=font_size, weight='bold')
ax.set_ylabel("Time (s)", size=font_size, weight='bold')

ax.set_ylim(bottom=0)
ax.set_ylim(top=90)

ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.grid(True, alpha=0.3)

short_indices = []
long_indices = []
short_ratios = []
long_ratios = []
for idx, iter_num in enumerate(common_iters):
    if (iter_num + 1) % 5 == 0:
        short_indices.append(idx)
        if no_adaptive_common_durations[idx] > 0:
            ratio = no_adaptive_common_durations[idx] / adaptive_common_durations[idx]
            short_ratios.append(ratio)
    else:
        long_indices.append(idx)
        if no_adaptive_common_durations[idx] > 0:
            ratio = no_adaptive_common_durations[idx] / adaptive_common_durations[idx]
            long_ratios.append(ratio)

from matplotlib.legend_handler import HandlerLine2D

ax.legend(
    [adaptive_line, no_adaptive_line],
    ['w/   Adpt. Timeout', 'w/o Adpt. Timeout'],
    loc='lower right',
    bbox_to_anchor=(1, -0.1),
    prop={
        'weight': 'bold',
        'size': font_size
    },
    frameon=False,
    handletextpad=0.2,
    columnspacing=0.8,
    labelspacing=0.0,
    handler_map={adaptive_line: HandlerLine2D(numpoints=2), no_adaptive_line: HandlerLine2D(numpoints=2)}
)

plt.tight_layout()
plt.savefig(f"./2_code_timeout/code_adaptive_time_limit.pdf", bbox_inches='tight')
print("Saved figure to ./2_code_timeout/code_adaptive_time_limit.pdf")
plt.close()