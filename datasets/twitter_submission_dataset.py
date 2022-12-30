from torch.utils.data import Dataset
import os
import random
import pandas as pd
import numpy as np
import cv2
import pickle
import glob
import json
from transformers import BertTokenizer, BertTokenizerFast

DATASET_PATH = '../../../MultiModalDataset'

class TwitterSubmissionDataset(Dataset):
    def __init__(self, args=None, label="positive", users=None):
        self.args = args
        self.label = label

        self.users_paths = sorted(glob.glob(f"{DATASET_PATH}/{self.label}/*"))

        if users is not None:
            only_users = list(map(lambda x: x.split("/")[-1], users))
            self.users_paths = [
                p for p in self.users_paths if p.split("/")[-1] in only_users
            ]

    def __len__(self):
        return len(self.users_paths)

    def __getitem__(self, idx):
        user_path = self.users_paths[idx]
        user = user_path.split("/")[-1]
        if not os.path.isfile(f"{user_path}/timeline.txt"):
            print(f"user {user} does not have timeline")
            image_paths = sorted(glob.glob(f"{DATASET_PATH}/{self.label}/{user}/*.jpg"))
            images = [cv2.imread(img_path) for img_path in image_paths]
            sample = {
                "author": user,
                "ids": [],
                "date": [],
                "texts": [],
                "images_paths": image_paths,
                "images": images,
            }
            return sample

        user_timeline = pd.read_json(f"{user_path}/timeline.txt", lines=True)
        images = []
        image_paths = []
        for id, row in user_timeline.iterrows():
            entities = pd.json_normalize(row.entities)
            if "media" in entities.columns:
                img = cv2.imread(f"{user_path}/{row['id']}.jpg")
                image_path = f"{user_path}/{row['id']}.jpg"
                if img is None:
                    img = np.nan
            else:
                img = np.nan
                image_path = np.nan
            images.append(img)
            image_paths.append(image_path)

        dates = [
            int(round(date.timestamp()))
            for date in user_timeline["created_at"].tolist()
        ]

        sample = {
            "author": user,
            "ids": user_timeline["id"].tolist(),
            "date": dates,
            "texts": user_timeline["text"].tolist(),
            "images_paths": image_paths,
            "images": images,
        }

        return sample
