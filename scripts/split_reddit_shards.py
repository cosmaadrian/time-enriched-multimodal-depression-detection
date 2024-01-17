import pandas as pd
import json
import os
import numpy as np
import time
import tqdm
import pickle
from collections import defaultdict

KIND = "test"

SPLITS_PATH = "../../../data/splits"
EMBEDDINGS_PATH = "../embeddings/for_experiments"
SHARDS_PATH = f"../embeddings/reddit-sharded-2/{KIND}"
os.makedirs(SHARDS_PATH, exist_ok=True)

SHARD_SIZE = 4096
OVERLAP = 512
TIME_THRESHOLD = 0.05

IMAGE_EMBEDDINGS = ["clip", "dino"]
TEXT_EMBEDDINGS = ["roberta", "emoberta", "minilm"]


def split_into_shards(user, embedding_types, size):

    paths = defaultdict(list)
    for _type in embedding_types:

        start_time = time.time()
        with open(f"{EMBEDDINGS_PATH}/{KIND}/{user}/{_type}.pkl", "rb") as f:
            embeddings = pickle.load(f)
        end_time = time.time()

        if embeddings.shape[0] < 10_000:
            continue

        print(f"{user} ::: {round(end_time - start_time, 3)} ::: {embeddings.shape[0]}")

        total_size = int(np.ceil(size / SHARD_SIZE))
        for k in range(total_size):
            start_idx, end_idx = k * SHARD_SIZE, (k + 1) * SHARD_SIZE + OVERLAP
            num_posts_contained = min(len(embeddings), end_idx) - start_idx

            if num_posts_contained < 512:
                end_idx = start_idx + num_posts_contained

            if k == total_size - 1 and num_posts_contained < 512:
                start_idx = start_idx - (OVERLAP - num_posts_contained)

            sharded_embeddings = embeddings[start_idx:end_idx]
            path = f"{user}-{_type}-{start_idx}:{end_idx}.pkl"

            with open(f"{SHARDS_PATH}/{path}", "wb") as f:
                pickle.dump(sharded_embeddings, f)

            paths[_type].extend([_type])

    return paths


df = pd.read_csv(
    f"{SPLITS_PATH}/{KIND}_users_multimodal.csv", lineterminator="\n", low_memory=False
)
users = df["user"].tolist()

for i, user in enumerate(sorted(users)):
    paths_df = {}

    user_df = pd.read_csv(
        f"{SPLITS_PATH}/for_experiments/{KIND}/{user}.csv", lineterminator="\n"
    )
    label = int(user_df["label"].unique()[0])
    user_df = user_df.drop(
        ["subreddit", "id", "url", "body", "title", "selftext", "image_path", "label"],
        axis=1,
    )

    paths = split_into_shards(
        user=user, embedding_types=TEXT_EMBEDDINGS, size=len(user_df.index)
    )

    if not paths:
        continue

    for key, values in paths.items():
    paths_df[key] = values

    save_path = f'{SPLITS_PATH}/for_experiments-sharded/{KIND}'
    os.makedirs(save_path, exist_ok = True)

    with open(save_path + f'/{user}-shards.json', 'wt') as f:
        json.dump(paths_df, f, indent = 4)
    user_df = user_df.to_csv(save_path + f'/{user}-shards.csv', index = False)
