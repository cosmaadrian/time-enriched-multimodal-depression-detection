import pandas as pd
import pprint
import pickle
import json
import tqdm
import glob
import numpy as np

print = pprint.pprint

OUTPUT_PATH = "./image-text-pairs/"

DATA_PATH = "../../../MultiModalDataset"
EMBEDDINGS_PATH = "../embeddings/twitter"

IMAGE_EMBEDDING_TYPE = ["clip", "dino"]
TEXT_EMBEDDING_TYPE = ["emoberta", "roberta", "minilm"]

positive_users = sorted(glob.glob(f"{DATA_PATH}/positive/*"))
negative_users = sorted(glob.glob(f"{DATA_PATH}/negative/*"))
users = positive_users + negative_users

labels = {user_path.split("/")[-1]: user_path.split("/")[-2] for user_path in users}

with open("../datasets/twitter-dates.json", "rt") as f:
    data = json.load(f)

users_no_images = []
for user in sorted(tqdm.tqdm(list(data.keys()))):
    dates = data[user]
    label_name = labels[user]

    image_embeddings = dict()
    for image_type in IMAGE_EMBEDDING_TYPE:
        with open(f"{EMBEDDINGS_PATH}/{label_name}/{user}/{image_type}.pkl", "rb") as f:
            image_embeddings[image_type] = pickle.load(f)

    text_embeddings = dict()
    for text_type in TEXT_EMBEDDING_TYPE:
        with open(f"{EMBEDDINGS_PATH}/{label_name}/{user}/{text_type}.pkl", "rb") as f:
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
            kind: np.array([image_embeddings[kind][idx] for idx in indices]).reshape(
                (len(indices), -1)
            )
            for kind in IMAGE_EMBEDDING_TYPE
        },
        "text_embeddings": {
            kind: np.array([text_embeddings[kind][idx] for idx in indices]).reshape(
                (len(indices), -1)
            )
            for kind in TEXT_EMBEDDING_TYPE
        },
        "dates": np.array([dates[idx] for idx in indices]),
        "label": label_name,
    }

    filename = f"{OUTPUT_PATH}/{user}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(new_data, f)

print(f"{len(users_no_images)} / {len(list(data.keys()))} users with no images")
