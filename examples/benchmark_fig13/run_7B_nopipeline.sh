#!/bin/bash
set +x

export RAY_DEDUP_LOGS=0 

ray stop --force
CONFIG_PATH=dakai/$(basename $(dirname $0))

echo $CONFIG_PATH

for ROLLOUT_BATCH_SIZE in 44
do
    if [ "$ROLLOUT_BATCH_SIZE" -eq 44 ]; then
        GRAD_ACC_STEPS=22
    elif [ "$ROLLOUT_BATCH_SIZE" -eq 128 ]; then
        GRAD_ACC_STEPS=64
    else
        echo "Unsupported ROLLOUT_BATCH_SIZE: $ROLLOUT_BATCH_SIZE"
        exit 1
    fi
    for MAX_NEW_TOKENS in 1
    do
        timestamp=$(date +%Y%m%d-%H%M%S)
        echo "Running with rollout_batch_size=${ROLLOUT_BATCH_SIZE}, actor_train.training_args.gradient_accumulation_steps=${GRAD_ACC_STEPS}, rewards.llm_judge.generating_args.max_new_tokens=${MAX_NEW_TOKENS}"
        python examples/start_rlvr_pipeline_async.py \
            --config_path $CONFIG_PATH \
            --config_name rlvr_config_qwen2-7B_layerbylayer_nopipeline \
            rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
            actor_train.training_args.gradient_accumulation_steps=${GRAD_ACC_STEPS} \
            rewards.llm_judge.generating_args.max_new_tokens=${MAX_NEW_TOKENS} \
            > ${timestamp}_rollout${ROLLOUT_BATCH_SIZE}_nopipeline_maxnew${MAX_NEW_TOKENS}.log 2>&1
        echo "save log to ${timestamp}_rollout${ROLLOUT_BATCH_SIZE}_nopipeline_maxnew${MAX_NEW_TOKENS}.log"
        ray stop --force
    done
done