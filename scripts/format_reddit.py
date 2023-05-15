import pandas as pd
import os
import numpy as np
import tqdm
import pickle
from collections import defaultdict

SPLITS_PATH = "../../../data/splits"
EMBEDDINGS_PATH = "../embeddings/for_experiments"
KIND = "train"

df = pd.read_csv(
    f"{SPLITS_PATH}/{KIND}_users_multimodal.csv", lineterminator="\n", low_memory=False
)
users = df["user"].tolist()

user_dates = {}
for i, user in enumerate(sorted(users)):

    user_df = pd.read_csv(
        f"{SPLITS_PATH}/for_experiments/{KIND}/{user}.csv", lineterminator="\n"
    )
    label = int(user_df["label"].unique()[0])
    user_df = user_df.drop(
        [
            "author",
            "subreddit",
            "id",
            "url",
            "body",
            "title",
            "selftext",
            "image_path",
            "label",
        ],
        axis=1,
    )

    user_dates[user] = user_df["created_utc"]

    print(user_df)
