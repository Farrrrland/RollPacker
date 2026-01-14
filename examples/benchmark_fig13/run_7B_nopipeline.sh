#!/bin/bash
set +x

export RAY_DEDUP_LOGS=0 

ray stop --force
CONFIG_PATH=$(basename $(dirname $0))

echo $CONFIG_PATH

python examples/start_rlvr_pipeline_async.py --config_path $CONFIG_PATH --config_name rlvr_config_qwen2-7B_layerbylayer_nopipeline