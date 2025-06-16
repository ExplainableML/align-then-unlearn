#!/usr/bin/env python

import faulthandler
import logging
import os
import socket
import warnings
import torch
import wandb
from hydra.utils import instantiate, get_class
from lightning.pytorch.callbacks import (
    EarlyStopping,
    TQDMProgressBar,
)
from omegaconf import DictConfig, OmegaConf, open_dict

from project.utils import (
    filter_device_available,
    get_logger,
    print_config,
    set_seed,
)

# Log to traceback to stderr on segfault
faulthandler.enable(all_threads=False)

# If data loading is really not a bottleneck for you, uncomment this to silence the
# warning about it
# warnings.filterwarnings(
#     "ignore",
#     "The '\w+_dataloader' does not have many workers",
#     module="lightning",
# )
warnings.filterwarnings(
    "ignore",
    "The `srun` command is available on your system but is not used",
    module="lightning",
)
logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(
    filter_device_available
)


log = get_logger()


def store_job_info(config: DictConfig):
    host = socket.gethostname()
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    process_id = os.getpid()

    with open_dict(config):
        config.host = host
        config.process_id = process_id
        if array_job_id is not None and array_task_id is not None:
            config.slurm_job_id = f"{array_job_id}_{array_task_id}"
        elif job_id is not None:
            config.slurm_job_id = job_id



# @hydra.main(config_path="config", config_name="train", version_base=None)
# @print_exceptions
def train(config: DictConfig):
    print(f"Running training on {socket.gethostname()}")
    rng = set_seed(config)

    # Log host and slurm job ID
    store_job_info(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    print_config(config)

    torch.set_float32_matmul_precision(config.matmul_precision)

    target_id: str = config.unlearning_target
    target_name = target_id.split("_", 1)[1].replace("_", " ")

    log.info("Instantiating pre-trained model")
    pre_trained_llm, pre_trained_llm_tokenizer = instantiate(config.pre_trained_llm)

    log.info("Instantiating logger")
    logger = instantiate(
        config.wandb,
        _target_="lightning.pytorch.loggers.WandbLogger",
        resume=(config.wandb.mode == "online") and "allow",
        log_model=True,
    )

    log.info("Running task")
    TaskClass = get_class(config.task._target_)
    task = TaskClass(
        config,
        target_id=target_id,
        target_name=target_name,
        pre_trained_llm=pre_trained_llm,
        pre_trained_llm_tokenizer=pre_trained_llm_tokenizer,
        logger=logger,
    )
    task.unlearn()

    wandb.finish()


if __name__ == "__main__":
    print("Use launch_training.py to run this script")
#     main()
