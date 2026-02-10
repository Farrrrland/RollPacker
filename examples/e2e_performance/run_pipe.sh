set +x

export WANDB_MODE=online
export RAY_DEDUP_LOGS=0 
export SKIP_OLD_LOG_PROBS=True
export NVTE_TORCH_COMPILE=0

export TORCH_CUDA_ARCH_LIST="9.0+PTX"

CONFIG_PATH=$(basename $(dirname $0))

ray stop --force
export PYTHONPATH=$PYTHONPATH:$(pwd)

export PYTHONUNBUFFERED=1
stdbuf -oL -eL python3 -u examples/start_rlvr_pipeline_async.py \
    --config_path "$CONFIG_PATH" \
    --config_name rlvr_config_rollpacker_full_7B > e2e.log 2>&1 &
tail -f e2e.log
