from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
import glob
import json
import tqdm
import os

DATA_PATH = [
    "../data/notdepressed/notdepressed1-reddit_crawling_2021-06-01 11:49:29",
    "../data/depressed/depressed-reddit_crawling_2021-05-26 13:32:03",
]

EXPERIMENTS_DATA_PATH = "../data/splits/for_experiments"

files = sorted(glob.glob(DATA_PATH[0]) + glob.glob(DATA_PATH[1]))

# too many posts
users_with_large_files = ["reddit_-peterboykin", "reddit_-itchyyyyscrotum"]


def get_users_labels(path):
    data_df = pd.read_csv(path)
    users = data_df["user"]
    labels = data_df["label"]
    return list(users), list(labels)


def make_dataset(users, labels, kind="train"):

    for user, label in tqdm.tqdm(zip(users, labels), total=len(users)):
        meta_list = []
        print(user)
        if user in users_with_large_files:
            print("Reading a very big file for user")
            print(user)

        data_path = DATA_PATH[int(label)]

        user_file = f"{data_path}/texts/{user}.jsonl"

        with pd.read_json(user_file, lines=True, chunksize=100000) as reader:

            for chunk in reader:

                if "body" not in chunk.columns:
                    chunk["body"] = np.nan

                if "selftext" not in chunk.columns:
                    chunk["selftext"] = np.nan

                if "title" not in chunk.columns:
                    chunk["title"] = np.nan

                if "url" not in chunk.columns:
                    chunk["url"] = np.nan

                chunk = chunk[
                    [
                        "author",
                        "id",
                        "subreddit",
                        "created_utc",
                        "title",
                        "selftext",
                        "body",
                        "url",
                    ]
                ]

                chunk["image_path"] = chunk["url"].apply(
                    lambda x: str(x).split("/")[-1]
                    if str(x).split(".")[-1] == "jpg" or str(x).split(".")[-1] == "png"
                    else np.nan
                )

                # print(':::: Initial Images: ', len(chunk[chunk['image_path'].notna() == True]))

                chunk["image_path"] = chunk["image_path"].apply(
                    lambda x: x
                    if cv2.imread(f"{data_path}/images/{str(x)}") is not None
                    else np.nan
                )

                # print(':::: Images after checking if they exist: ', len(chunk[chunk['image_path'].notna() == True]))

                chunk["selftext"] = chunk["selftext"].apply(
                    lambda x: x if str(x) != "[removed]" else np.nan
                )

                chunk["label"] = label

                # print(':::: User: ', chunk['author'].unique()[0])
                # print(':::: Images: ', len(chunk[chunk['image_path'].notna() == True]))
                # print(':::: Posts: ', len(chunk))
                # print(':::: Label: ', chunk['label'].unique()[0])

                if len(chunk) != 0:
                    meta_list.append(chunk)

            df = pd.concat(meta_list)

            if not os.path.exists(f"{EXPERIMENTS_DATA_PATH}/{kind}"):
                os.makedirs(f"{EXPERIMENTS_DATA_PATH}/{kind}")
            df.to_csv(f"{EXPERIMENTS_DATA_PATH}/{kind}/{user}.csv", index=False)


train_users, train_labels = get_users_labels(
    "../data/splits/train_users_multimodal.csv"
)
val_users, val_labels = get_users_labels("../data/splits/valid_users_multimodal.csv")
test_users, test_labels = get_users_labels("../data/splits/test_users_multimodal.csv")

make_dataset(train_users, train_labels, kind="train")
make_dataset(val_users, val_labels, kind="val")
make_dataset(test_users, test_labels, kind="test")
