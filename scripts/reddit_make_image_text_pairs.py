import pandas as pd
import pprint
import pickle
import json
import tqdm
import glob
import os
import numpy as np

print = pprint.pprint

OUTPUT_PATH = "../../../image-text-pairs-reddit"

SPLITS_PATH = "../../../data/splits/users_without_many_posts/"
EMBEDDINGS_PATH = "../embeddings/for_experiments/"

IMAGE_EMBEDDING_TYPE = ["clip", "dino"]
TEXT_EMBEDDING_TYPE = ["emoberta", "roberta", "minilm"]

KIND = "valid"

df = pd.read_csv(
    f"{SPLITS_PATH}/{KIND}_users_multimodal.csv", lineterminator="\n", low_memory=False
)
users = df["user"].tolist()
labels = df["label"].tolist()

with open("../datasets/reddit-dates.json", "rt") as f:
    user_dates = json.load(f)

users_no_images = []
for user, label_name in tqdm.tqdm(zip(users, labels), total=len(users)):
    dates = user_dates[user]

    image_embeddings = dict()
    for image_type in IMAGE_EMBEDDING_TYPE:
        with open(f"{EMBEDDINGS_PATH}/{KIND}/{user}/{image_type}.pkl", "rb") as f:
            image_embeddings[image_type] = pickle.load(f)

    text_embeddings = dict()
    for text_type in TEXT_EMBEDDING_TYPE:
        with open(f"{EMBEDDINGS_PATH}/{KIND}/{user}/{text_type}.pkl", "rb") as f:
            text_embeddings[text_type] = pickle.load(f)

    indices = []
    for i, img in enumerate(image_embeddings[IMAGE_EMBEDDING_TYPE[0]]):
        if isinstance(img, float):
            continue
        indices.append(i)

    if not len(indices):
        print(f"::: User {user} has no images.")
        users_no_images.append(user)
        continue

    new_data = {
        "image_embeddings": {
            kind: np.array([image_embeddings[kind][idx] for idx in indices]).reshape((len(indices), -1))
            for kind in IMAGE_EMBEDDING_TYPE
        },
        "text_embeddings": {
            kind: np.array([text_embeddings[kind][idx] for idx in indices]).reshape((len(indices), -1))
            for kind in TEXT_EMBEDDING_TYPE
        },
        "dates": np.array([dates[idx] for idx in indices]),
        "label": label_name,
    }

    os.makedirs(f"{OUTPUT_PATH}/{KIND}", exist_ok=True)
    filename = f"{OUTPUT_PATH}/{KIND}/{user}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(new_data, f)

print(f"{len(users_no_images)} / {len(users)} users with no images")
