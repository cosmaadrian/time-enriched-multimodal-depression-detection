import pandas as pd
import os
import numpy as np
import tqdm
import pickle
import json
from collections import defaultdict

SPLITS_PATH = "../../../data/splits"
EMBEDDINGS_PATH = "../embeddings/for_experiments"

user_dates = {}

for kind in ["train", "valid", "test"]:

    df = pd.read_csv(
        f"{SPLITS_PATH}/{kind}_users_multimodal.csv",
        lineterminator="\n",
        low_memory=False,
    )
    users = df["user"].tolist()

    users_with_large_files = []
    for i, user in tqdm.tqdm(enumerate(sorted(users)), total=len(users)):
        user_df = pd.read_csv(
            f"{SPLITS_PATH}/for_experiments/{kind}/{user}.csv", lineterminator="\n"
        )
        if len(user_df) > 50_000:
            users_with_large_files.append(user)
            continue

        # user_df = user_df.drop(['author', 'subreddit', 'id', 'url', 'body', 'title', 'selftext', 'image_path', 'label'], axis = 1)
        user_dates[user] = user_df["created_utc"].values.tolist()

    # print(len(df))
    # df = df[~df['user'].isin(users_with_large_files)]
    # print(len(df))
    # df.to_csv(f'{SPLITS_PATH}/users_without_many_posts/{kind}_users_multimodal.csv', index = False)

with open("../datasets/reddit-dates.json", "wt") as f:
    json.dump(user_dates, f)
