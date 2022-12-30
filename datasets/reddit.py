from torch.utils.data import Dataset
import os
import random
import pandas as pd
import numpy as np
import cv2
import pickle
import glob
import pprint
import json
import time
import multiprocessing

from transformers import BertTokenizer, BertTokenizerFast
from datasets.time_dataset import TimeDataset

SPLITS_PATH = "../../data/splits/users_without_many_posts"
EMBEDDINGS_PATH_TEXT = "embeddings/for_experiments"
EMBEDDINGS_PATH_IMAGES = "embeddings_correct/for_experiments"
SHARDS_PATH = "embeddings/reddit-sharded-2/"


class RedditDataset(TimeDataset):
    def __init__(self, args, kind="train"):
        self.args = args
        self.kind = kind

        self.df = pd.read_csv(
            f"{SPLITS_PATH}/{kind}_users_multimodal.csv",
            lineterminator="\n",
            low_memory=False,
        )
        self.users = self.df["user"].tolist()

        self.sharded_users = list(
            set(
                [
                    "-".join(u.split("-")[:-1]).split("/")[-1]
                    for u in glob.glob(f"{SHARDS_PATH}/{kind}/*.pkl")
                ]
            )
        )

        self.window_size = self.args.window_size
        self.labels = self.df["label"].tolist()

        with open("datasets/reddit-dates.json", "rt") as f:
            self.user_dates = json.load(f)

    def _read_text_embeddings(self, user):
        user_key = f"{user}-{self.args.text_embeddings_type}"
        if user_key not in self.sharded_users:
            with open(
                f"{EMBEDDINGS_PATH_TEXT}/{self.kind}/{user}/{self.args.text_embeddings_type}.pkl",
                "rb",
            ) as f:
                text_embeddings = pickle.load(f)
            return text_embeddings, 0, text_embeddings.shape[0]

        shard_paths = sorted(
            glob.glob(
                f"{SHARDS_PATH}/{self.kind}/{user}-{self.args.text_embeddings_type}*.pkl"
            )
        )

        # while True:
        # 	text_embeddings_path = random.choice(shard_paths)
        # 	start_idx, end_idx = text_embeddings_path.split('-')[-1].split('.')[0].split(':')
        # 	with open(text_embeddings_path, 'rb') as f:
        # 		text_embeddings = pickle.load(f)

        # 	if text_embeddings.shape[0] >= self.args.window_size:
        # 		break
        # 	else:
        # 		print('trying again ... ')
        # 		print(text_embeddings.shape[0], int(end_idx) - int(start_idx), self.args.window_size)
        # 		continue

        text_embeddings_path = random.choice(shard_paths)
        start_idx, end_idx = (
            text_embeddings_path.split("-")[-1].split(".")[0].split(":")
        )
        with open(text_embeddings_path, "rb") as f:
            text_embeddings = pickle.load(f)

        if text_embeddings.shape[0] < self.args.window_size:
            print(text_embeddings.shape[0], self.args.window_size)
            print(text_embeddings_path)

        assert text_embeddings.shape[0] >= self.args.window_size

        return text_embeddings, int(start_idx), int(end_idx)

    def __getitem__(self, idx):
        user = self.users[idx]
        label = self.labels[idx]
        dates = np.array(self.user_dates[user])

        image_embeddings = None
        text_embeddings = None

        ########################################
        if self.args.modality in ["image", "both"]:
            with open(
                f"{EMBEDDINGS_PATH_IMAGES}/{self.kind}/{user}/{self.args.image_embeddings_type}.pkl",
                "rb",
            ) as f:
                image_embeddings = pickle.load(f)

        start_time = time.time()
        if self.args.modality in ["text", "both"]:
            text_embeddings, start_idx, end_idx = self._read_text_embeddings(user)
            dates = dates[start_idx:end_idx]

            if self.args.modality == "both":
                image_embeddings = image_embeddings[start_idx:end_idx]
        ########################################
        end_time = time.time()

        # print(round(end_time - start_time, 3), user, text_embeddings.shape)

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
