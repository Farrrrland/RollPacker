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
        "--total_iterations", type=int, default=21,
        help="Number of iterations to run for debug inference."
    )
    args = parser.parse_args()

    initialize(config_path=args.config_path, job_name="app")
    cfg = compose(config_name=args.config_name)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()
    ppo_config.generate_opt_level = 2

    pipeline = RLVRPipelineAsync(pipeline_config=ppo_config)
    pipeline.debug_infer(total_iterations=args.total_iterations)

if __name__ == "__main__":
    main()
