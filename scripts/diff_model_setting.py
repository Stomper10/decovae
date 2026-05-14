# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import argparse
import json
import logging

import torch
import torch.distributed as dist
from monai.utils import RankFilter

from scripts.config_utils import load_json


def setup_logging(logger_name: str = "") -> logging.Logger:
    """
    Setup the logging configuration.

    Args:
        logger_name (str): logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(logger_name)
    if dist.is_initialized():
        logger.addFilter(RankFilter())
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logger


def load_config(env_config_path: str, model_config_path: str, model_def_path: str) -> argparse.Namespace:
    """
    Load configuration from JSON files.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.

    Returns:
        argparse.Namespace: Loaded configuration.
    """
    args = argparse.Namespace()

    for k, v in load_json(env_config_path).items():
        setattr(args, k, v)
    for k, v in load_json(model_config_path).items():
        setattr(args, k, v)
    for k, v in load_json(model_def_path).items():
        setattr(args, k, v)

    return args

def initialize_distributed() -> tuple:
    """
    Initialize distributed training based on environment variables.
    """
    # torchrun sets LOCAL_RANK. If it's not set, we're not in a distributed run.
    if 'LOCAL_RANK' in os.environ and torch.cuda.is_available():
        # These are set by torchrun
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Initialize the process group
        dist.init_process_group(backend="nccl", init_method="env://")
        
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        print(f"Initialized process {local_rank}/{world_size} on device {device}.")
    else:
        # Single GPU or CPU run
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return local_rank, world_size, device