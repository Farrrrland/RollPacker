import argparse
import os

from dacite import from_dict, Config
from hydra.experimental import compose, initialize
import hydra
from omegaconf import OmegaConf
import roll 
import roll.distributed 
from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig

from roll.pipeline.rlvr.rlvr_pipeline_async import RLVRPipelineAsync


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default="config")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="sppo_config"
    )
    args, unknown = parser.parse_known_args()

    with initialize(config_path=args.config_path, job_name="app"):
        cfg = compose(config_name=args.config_name, overrides=unknown)

    ppo_config = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    init()
    ppo_config.generate_opt_level = 2
    ppo_config.fake_old_log_probs = True

    pipeline_config = ppo_config
    # check autoscaling 
    if pipeline_config.autoscaling == True: 
        assert eval(os.environ.get('SKIP_OLD_LOG_PROBS', 'False')), 'we should directly skip old log probs computation'
        assert pipeline_config.infer_scaling_down_progress_ratio > 0 and pipeline_config.scaling_down_train_batch_size > 0, 'shoud set a postive number.'

    elif pipeline_config.autoscaling == False:
        assert pipeline_config.infer_scaling_down_progress_ratio < 0 and pipeline_config.scaling_down_train_batch_size < 0, 'shoud set a negative number.'

    if pipeline_config.actor_train.use_remove_padding: 
        assert eval(os.environ.get('NVTE_TORCH_COMPILE', '1')) == 0, 'we should set it as zero'

    print(OmegaConf.to_yaml(cfg, resolve=True))
    pipeline = RLVRPipelineAsync(pipeline_config=ppo_config)

    pipeline.run()


if __name__ == "__main__":
    main()
