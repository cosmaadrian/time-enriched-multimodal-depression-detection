from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from PIL import Image
import glob
import json
import tqdm

DATA_PATH = [
    "../data/notdepressed/notdepressed1-reddit_crawling_2021-06-01 11:49:29/texts",
    "../data/depressed/depressed-reddit_crawling_2021-05-26 13:32:03/texts",
]

files = sorted(glob.glob(DATA_PATH[0]) + glob.glob(DATA_PATH[1]))


def get_users_labels(path):
    df = pd.read_csv(path)
    users = df["user"]
    labels = df["label"]
    return list(users), list(labels)


def make_official_dataset(users, labels, kind="train"):

    meta_list = []
    num_users = 0

    for user, label in tqdm.tqdm(zip(users, labels), total=len(users)):
        data_path = DATA_PATH[int(label)]

        user_file = f"{data_path}/{user}.jsonl"

        user_posts = pd.read_json(user_file, lines=True)

        # keep only users that have images
        if "post_hint" not in user_posts.columns:
            continue

        if "image" not in user_posts["post_hint"].unique():
            continue

        user_posts = user_posts[
            (user_posts["post_hint"] == "image") & (user_posts["title"] != np.nan)
        ]

        user_posts = user_posts[
            [
                "author",
                "id",
                "subreddit",
                "created_utc",
                "title",
                "post_hint",
                "url",
                "over_18",
            ]
        ]

        user_posts["image_path"] = user_posts["url"].apply(
            lambda x: x.split("/")[-1] if type(x) == str else np.nan
        )
        user_posts["label"] = label

        # print(':::: Images: ', len(user_posts))

        if len(user_posts) != 0:
            meta_list.append(user_posts)
            num_users += 1

    df = pd.concat(meta_list)
    print(df)
    print(
        "::: Users: ",
        len(
            df.groupby("author")
            .apply(lambda x: x.notnull().sum())["image_path"]
            .values.tolist()
        ),
    )
    df.to_csv(f"../data/splits/image_titles/{kind}.csv", index=False)


def make_dataset_with_posts_and_images(users, labels, kind="train"):
    meta_list = []
    num_users = 0

    for user, label in tqdm.tqdm(zip(users, labels), total=len(users)):

        data_path = DATA_PATH[int(label)]

        user_file = f"{data_path}/{user}.jsonl"

        user_posts = pd.read_json(user_file, lines=True)

        # keep only users that also have images, besides texts
        if "post_hint" not in user_posts.columns:
            continue

        if "image" not in user_posts["post_hint"].unique():
            continue

        if "selftext" in user_posts.columns:
            user_posts = user_posts[user_posts["selftext"] != "[removed]"]

        if "body" not in user_posts.columns:
            user_posts["body"] = np.nan

        user_posts = user_posts[
            [
                "author",
                "id",
                "subreddit",
                "created_utc",
                "title",
                "selftext",
                "body",
                "post_hint",
                "url",
                "over_18",
            ]
        ]
        user_posts.loc[user_posts["post_hint"] != "image", "url"] = np.nan
        user_posts["image_path"] = user_posts["url"].apply(
            lambda x: x.split("/")[-1] if type(x) == str else np.nan
        )
        user_posts["label"] = label

        # print(':::: Images: ', len(user_posts))
        # print(user_posts)

        if len(user_posts) != 0:
            meta_list.append(user_posts)
            num_users += 1

    df = pd.concat(meta_list)
    print(df)
    print(
        "::: Users: ",
        len(
            df.groupby("author")
            .apply(lambda x: x.notnull().sum())["image_path"]
            .values.tolist()
        ),
    )
    df.to_csv(f"../data/splits/images_and_posts/{kind}.csv", index=False)


def make_dataset_with_only_texts(users, labels, kind="train"):
    meta_list = []
    num_users = 0

    for user, label in tqdm.tqdm(zip(users, labels), total=len(users)):

        data_path = DATA_PATH[int(label)]

        user_file = f"{data_path}/{user}.jsonl"

        user_posts = pd.read_json(user_file, lines=True)

        # keep only users that also have images, besides texts
        if "post_hint" not in user_posts.columns:
            continue

        if "image" not in user_posts["post_hint"].unique():
            continue

        if "selftext" in user_posts.columns:
            user_posts = user_posts[user_posts["selftext"] != "[removed]"]

        if "body" not in user_posts.columns:
            user_posts["body"] = np.nan

        user_posts = user_posts[
            [
                "author",
                "id",
                "subreddit",
                "created_utc",
                "title",
                "selftext",
                "body",
                "post_hint",
                "url",
                "over_18",
            ]
        ]
        user_posts.loc[user_posts["post_hint"] != "image", "url"] = np.nan
        user_posts["image_path"] = user_posts["url"].apply(
            lambda x: x.split("/")[-1] if type(x) == str else np.nan
        )
        user_posts["label"] = label
        user_posts = user_posts[user_posts["post_hint"] != "image"]

        # print(':::: Images: ', len(user_posts))
        # print(user_posts)

        if len(user_posts) != 0:
            meta_list.append(user_posts)
            num_users += 1

    df = pd.concat(meta_list)
    print(df)
    print(df["post_hint"].unique())
    print(
        "::: Users: ",
        len(
            df.groupby("author")
            .apply(lambda x: x.notnull().sum())["image_path"]
            .values.tolist()
        ),
    )
    df.to_csv(f"../data/splits/only_text/{kind}.csv", index=False)


train_users, train_labels = get_users_labels(
    "../data/splits/train_users_multimodal.csv"
)
val_users, val_labels = get_users_labels("../data/splits/valid_users_multimodal.csv")
test_users, test_labels = get_users_labels("../data/splits/test_users_multimodal.csv")

make_dataset_with_only_texts(train_users, train_labels, kind="train")
make_dataset_with_only_texts(val_users, val_labels, kind="val")
make_dataset_with_only_texts(test_users, test_labels, kind="test")
