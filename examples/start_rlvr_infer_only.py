import argparse
import os

from dacite import from_dict, Config
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig

from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline
from roll.pipeline.rlvr.rlvr_pipeline_async import RLVRPipelineAsync


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    parser.add_argument(
        "--total_iterations", type=int, default=20,
        help="Number of iterations to run for debug inference."
    )
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()
    ppo_config.generate_opt_level = 2
    pipeline_config = ppo_config
    number_of_requests = pipeline_config.rollout_batch_size * pipeline_config.num_return_sequences_in_group
    tp_size = pipeline_config.actor_train.strategy_args.strategy_config.get('tensor_model_parallel_size', 1)
    pp_size = pipeline_config.actor_train.strategy_args.strategy_config.get('pipeline_model_parallel_size', 1)
    cp_size = pipeline_config.actor_train.strategy_args.strategy_config.get('context_parallel_size', 1)
    if pp_size == 1:
        pipeline_config.actor_train.strategy_args.strategy_config.pop('virtual_pipeline_model_parallel_size', None)
    if cp_size == 1: 
        pipeline_config.actor_train.strategy_args.strategy_config.pop('sequence_parallel', None)


    if pipeline_config.actor_train.training_args.gradient_accumulation_steps < 0: 
        pipeline_config.actor_train.training_args.gradient_accumulation_steps = int(number_of_requests / (pipeline_config.actor_train.world_size // (tp_size * pp_size * cp_size) * \
                            pipeline_config.actor_train.training_args.per_device_train_batch_size ))
        
    training_batch_size = pipeline_config.actor_train.world_size // (tp_size * pp_size * cp_size) * \
                        pipeline_config.actor_train.training_args.per_device_train_batch_size * pipeline_config.actor_train.training_args.gradient_accumulation_steps

    assert training_batch_size == number_of_requests, ("the number of samples in gen stage is {} v.s. the number of samples in training stage is {}".format(number_of_requests, training_batch_size))

    pipeline = RLVRPipelineAsync(pipeline_config=ppo_config)
    pipeline.debug_infer(total_iterations=args.total_iterations)

if __name__ == "__main__":
    main()
