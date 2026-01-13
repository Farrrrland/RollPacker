import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
edge_width = 0.1
plt.rc('font', family=plt.rcParams["font.family"][0], weight='medium')
plt.rcParams['hatch.linewidth'] = edge_width

def collect_compute_reward_mean(log_dir):
    jsonl_files = glob.glob(os.path.join(log_dir, "*.jsonl"))
    compute_rewards = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r") as f:
            for line in f:
                data = json.loads(line)
                timings = data.get("timings", {})
                start_compute_reward = timings.get("start_compute_reward")
                end_compute_reward = timings.get("end_compute_reward")
                if None not in (start_compute_reward, end_compute_reward):
                    compute_reward = end_compute_reward - start_compute_reward
                    compute_rewards.append(compute_reward)
    if len(compute_rewards) > 0:
        return np.mean(compute_rewards)
    else:
        return np.nan


def get_latest_request_time_dir(base_dir):
    if not os.path.isdir(base_dir):
        raise ValueError(f"{base_dir} is not a valid directory")
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs = [d for d in subdirs if d[0].isdigit()]
    if not subdirs:
        raise ValueError(f"No timestamp directories found in {base_dir}")
    latest_subdir = max(subdirs)
    return os.path.join(base_dir, latest_subdir, "request_time")

pipeline_log_dir_7B = get_latest_request_time_dir("output/profiler/llmjudge_qwen2-7B_layerbylayer")
no_pipeline_log_dir_7B = get_latest_request_time_dir("output/profiler/llmjudge_qwen2-7B_layerbylayer_nopipeline")
print(f"pipeline_log_dir_7B: {pipeline_log_dir_7B}")
print(f"no_pipeline_log_dir_7B: {no_pipeline_log_dir_7B}")


mean_no_pipeline_7B = collect_compute_reward_mean(no_pipeline_log_dir_7B)
mean_pipeline_7B = collect_compute_reward_mean(pipeline_log_dir_7B)

labels = ["7B/8k"]
bar_width = 0.6
x = np.array([0])

means_no_pipeline = [mean_no_pipeline_7B]
means_pipeline = [mean_pipeline_7B]

norm_no_pipeline = []
norm_pipeline = []
for no_pipe, pipe in zip(means_no_pipeline, means_pipeline):
    if not np.isnan(no_pipe) and no_pipe != 0:
        norm_no_pipeline.append(100.0)
        norm_pipeline.append(pipe / no_pipe * 100 if not np.isnan(pipe) else np.nan)
    else:
        norm_no_pipeline.append(np.nan)
        norm_pipeline.append(np.nan)

fig, ax = plt.subplots(figsize=(4, 3.5*1.1), dpi=200)

rects1 = ax.bar(
    x - bar_width/2, norm_no_pipeline, bar_width,
    label='w/o Pipe',
    color=(252/255, 131/255, 98/255),
    edgecolor='black',
    hatch='/',
    linewidth=edge_width,
    zorder=3
)
rects2 = ax.bar(
    x + bar_width/2, norm_pipeline, bar_width,
    label='w/ Pipe',
    color=(137/255, 190/255, 220/255),
    edgecolor='black',
    hatch='x',
    linewidth=edge_width,
    zorder=3
)

if not (np.isnan(mean_no_pipeline_7B) or np.isnan(mean_pipeline_7B) or mean_pipeline_7B == 0):
    speedup = mean_no_pipeline_7B / mean_pipeline_7B
    rect = rects2[0]
    ax.text(
        rect.get_x() + rect.get_width()/2,
        rect.get_height(),
        f"{speedup:.1f}x",
        ha='center',
        va='bottom',
        fontsize=24,
        fontweight='medium',
        zorder=3,
        clip_on=False,
        rotation=90
    )

ax.set_xlabel("Sequence Length", fontsize=24, fontweight='bold')
ax.set_ylabel("Norm. Time", fontsize=24, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=24)

ax.set_ylim(0, 150)
ax.set_xlim(-0.7, 0.7)

yticks = [0, 50, 100]
ax.set_yticks(yticks)
ax.set_yticklabels([str(int(y)) for y in yticks], fontsize=24)

ax.grid(axis='y', alpha=0.3)

ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 0.95),
    ncol=2,
    prop={
        'weight': 'bold',
        'size': 24
    },
    frameon=False,
    borderaxespad=0.,
    handletextpad=0.2,
    columnspacing=0.8,
    labelspacing=0.0,
)

plt.tight_layout()
plt.savefig(f"plot/img/layerwise_pipeline_7B.png", bbox_inches='tight')
print("Saved figure to plot/img/layerwise_pipeline_7B.png")
plt.close()
