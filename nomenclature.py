import yaml
from datasets import *
from models import *
from evaluators import MultimodalEvaluator

import torch

device = torch.device("cuda")

DATASETS = {
    "reddit": RedditDataset,
    "twitter": TwitterDataset,
}

EVALUATORS = {
    "multimodal-evaluator": MultimodalEvaluator,
}

MODELS = {
    "multimodal-transformer": MultiModalTransformer,
    "singlemodal-transformer": SingleModalTransformer,
    "tlstm": TimeLSTM,
}
