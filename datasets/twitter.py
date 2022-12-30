from torch.utils.data import Dataset
import json
import time
import os
import random
import pandas as pd
import numpy as np
import cv2
import pickle
import glob
import pprint
import random
import multiprocessing

from transformers import BertTokenizer, BertTokenizerFast
from datasets.time_dataset import TimeDataset

DATA_PATH = "../../MultiModalDataset"
EMBEDDINGS_PATH_TEXT = "embeddings/twitter"
EMBEDDINGS_PATH_IMAGES = "embeddings_correct/twitter"


class TwitterDataset(TimeDataset):
    def __init__(self, args, kind="train"):
        self.args = args
        self.kind = kind

        positive_users = sorted(glob.glob(f"{DATA_PATH}/positive/*"))
        negative_users = sorted(glob.glob(f"{DATA_PATH}/negative/*"))

        # No timeline
        if "../../MultiModalDataset/positive/IAMABBIE" in positive_users:
            positive_users.remove("../../MultiModalDataset/positive/IAMABBIE")

        users_per_fold = len(positive_users) // self.args.num_folds

        start_idx_fold = self.args.fold * users_per_fold
        end_idx_fold = (self.args.fold + 1) * users_per_fold

        if self.kind in ["valid", "test"]:
            positive_users_fold = positive_users[start_idx_fold:end_idx_fold]
            negative_users_fold = negative_users[start_idx_fold:end_idx_fold]
            self.users = positive_users_fold + negative_users_fold
            self.window_size = self.args.window_size

        if self.kind == "train":
            positive_users_fold = (
                positive_users[:start_idx_fold] + positive_users[end_idx_fold:]
            )
            negative_users_fold = (
                negative_users[:start_idx_fold] + negative_users[end_idx_fold:]
            )
            self.users = positive_users_fold + negative_users_fold

            self.window_size = self.args.window_size

        self.labels = [
            1 if user_path.split("/")[-2] == "positive" else 0
            for user_path in self.users
        ]

        self.positive_users = positive_users_fold
        self.negative_users = negative_users_fold

        self.users = list(map(lambda x: x.split("/")[-1], self.users))

        with open("datasets/twitter-dates.json", "rt") as f:
            self.user_dates = json.load(f)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        label = self.labels[idx]
        label_name = "positive" if label == 1 else "negative"

        dates = np.array(self.user_dates[user])

        image_embeddings = None
        text_embeddings = None

        ########################################
        if self.args.modality in ["image", "both"]:
            with open(
                f"{EMBEDDINGS_PATH_IMAGES}/{label_name}/{user}/{self.args.image_embeddings_type}.pkl",
                "rb",
            ) as f:
                image_embeddings = pickle.load(f)

        if self.args.modality in ["text", "both"]:
            with open(
                f"{EMBEDDINGS_PATH_TEXT}/{label_name}/{user}/{self.args.text_embeddings_type}.pkl",
                "rb",
            ) as f:
                text_embeddings = pickle.load(f)
        ########################################

        if self.args.modality == "both":
            sample = self.load_multimodal(
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                label=label,
                dates=dates,
                user_name=user,
            )
        else:
            modality = image_embeddings if text_embeddings is None else text_embeddings

            sample = self.load_singlemodal(
                modality=modality,
                label=label,
                dates=dates,
                user_name=user,
            )

        return sample
