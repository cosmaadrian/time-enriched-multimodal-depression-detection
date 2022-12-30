import torch

torch.multiprocessing.set_sharing_strategy("file_system")

import argparse
import torch
import wandb
import yaml
import os

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pprint

import callbacks
from trainer import NotALightningTrainer
from loggers import WandbLogger

from utils import load_args, load_model

import nomenclature

parser = argparse.ArgumentParser(description="Do stuff.")
parser.add_argument("--config_file", type=str, required=True)

parser.add_argument("--name", type=str, default="test")
parser.add_argument("--group", type=str, default="default")

parser.add_argument("--output_dir", type=str, default="reddit")
parser.add_argument("--dataset", type=str, required=True)

parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--position_embeddings", type=str, default=None)
parser.add_argument("--image_embeddings_type", type=str, default=None)
parser.add_argument("--text_embeddings_type", type=str, default=None)
parser.add_argument("--fold", type=int, default=None)

parser.add_argument("--batch_size", type=int, default=256)

args = parser.parse_args()
args, cfg = load_args(args)

dataset = nomenclature.DATASETS[args.dataset]
model = nomenclature.MODELS[args.model](args)

state_dict = load_model(args)
state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()
model.train(False)
model.cuda()

evaluator = nomenclature.EVALUATORS["multimodal-evaluator"](args, model)

results = evaluator.evaluate(save=True)
print(evaluator.__class__.__name__)
pprint.pprint(results)
