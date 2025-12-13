set +x

# Run TP comparison experiments
export WANDB_MODE=online
export RAY_DEDUP_LOGS=0 

export TORCH_CUDA_ARCH_LIST="9.0+PTX"

# install yq
pip install yq

sudo yum install -y epel-release
sudo yum install -y jq

CONFIG_PATH=$(basename $(dirname $0))
CONFIG_FILE="./examples/$CONFIG_PATH/rlvr_config_qwen2_5_7B.yaml"

if ! command -v yq &> /dev/null; then
    echo "Error, yq unsinstalled, install yq first (https://github.com/mikefarah/yq)"
    exit 1
fi

EXPERIMENTS=(
    [0]="1.0 8"
    [1]="1.0 9"
    [2]="1.0 10"
    [3]="1.0 12"
    [5]="1.125 8"
    [6]="1.25 8"
    [7]="1.5 8"
    [8]="1.25 10"
)

for idx in "${!EXPERIMENTS[@]}"; do
    IFS=' ' read -r ratio resp <<< "${EXPERIMENTS[$idx]}"

    yq -y --in-place ".max_prompts_ratio = $ratio" "$CONFIG_FILE"
    yq -y --in-place ".max_num_return_sequences = $resp" "$CONFIG_FILE"

    python3 -u examples/start_rlvr_infer_only.py \
        --config_path "$CONFIG_PATH" \
        --total_iterations 10 \
        --config_name rlvr_config_qwen2_5_7B 
done

