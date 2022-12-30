from torch.utils.data import Dataset
import os
import random
import pandas as pd
import numpy as np
import cv2
import pickle
from transformers import BertTokenizer, BertTokenizerFast

IMAGE_PATH = [
    "../../../data/notdepressed/notdepressed1-reddit_crawling_2021-06-01 11:49:29",
    "../../../data/depressed/depressed-reddit_crawling_2021-05-26 13:32:03",
]

SPLITS_PATH = "../../../data/splits"
# SPLITS_PATH = '../../../data/splits/users_without_many_posts'


class SubmissionsDataset(Dataset):
    def __init__(self, args=None, kind="train"):
        self.args = args
        self.kind = kind

        self.df = pd.read_csv(
            f"{SPLITS_PATH}/users_without_many_posts/{kind}_users_multimodal.csv",
            lineterminator="\n",
            low_memory=False,
        )

        self.users = self.df["user"].tolist()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]

        user_df = pd.read_csv(
            f"{SPLITS_PATH}/for_experiments/{self.kind}/{user}.csv", lineterminator="\n"
        )

        label = int(user_df["label"].unique()[0])

        user_df["title"] = user_df["title"].replace(np.nan, "")
        user_df["selftext"] = user_df["selftext"].replace(np.nan, "")
        user_df["body"] = user_df["body"].replace(np.nan, "")

        user_df["fulltext"] = (
            user_df["title"] + " " + user_df["selftext"] + " " + user_df["body"]
        )

        images = [
            cv2.cvtColor(
                cv2.imread(f"{IMAGE_PATH[label]}/images/{path}"), cv2.COLOR_RGBA2RGB
            )
            if not pd.isna(path)
            else np.nan
            for path in user_df["image_path"]
        ]

        sample = {
            "author": user,
            "ids": user_df["id"].tolist(),
            "date": user_df["created_utc"].tolist(),
            "texts": user_df["fulltext"].tolist(),
            "images_paths": user_df["image_path"].tolist(),
            "images": images,
        }

        return sample
