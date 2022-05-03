from datetime import datetime
from functools import partial

import math
import sys

import torch
import deepspeed
import numpy as np

from megatron.utils import (
    Timers,
    init_wandb,
    get_ltor_masks_and_position_ids,
    reduce_losses,
)
import json

from megatron import print_rank_0, mpu
from megatron.model import (
    GPT2ModelPipe,
    SoftEmbedding,
    get_params_for_weight_decay_optimization,
)
from megatron.neox_arguments.arguments import (
    NeoXArgsModel,
    NeoXArgsTokenizer,
    NeoXArgsTraining,
    NeoXArgsParallelism,
    NeoXArgsLogging,
    NeoXArgsOther,
    NeoXArgsTextgen,
    NeoXArgsOptimizer,
    NeoXArgsLRScheduler,
    ATTENTION_TYPE_CHOICES,
)


BASE_CLASSES = [
    NeoXArgsModel,
    NeoXArgsLRScheduler,
    NeoXArgsOptimizer,
    NeoXArgsTokenizer,
    NeoXArgsTraining,
    NeoXArgsParallelism,
    NeoXArgsLogging,
    NeoXArgsTextgen,
    NeoXArgsOther,
]

from megatron.neox_arguments import NeoXArgs
# make a mock neoxarg set

NeoXArgs(BASE_CLASSES)

