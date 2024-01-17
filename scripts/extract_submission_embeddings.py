from sentence_transformers import SentenceTransformer
from transformers import (
    ViTFeatureExtractor,
    ViTModel,
    CLIPProcessor,
    CLIPModel,
    CLIPVisionModel,
    CLIPFeatureExtractor,
    CLIPTextModel,
    CLIPTokenizer,
    RobertaModel,
    RobertaTokenizer,
    XLMRobertaTokenizer,
    BertModel,
    BertTokenizer,
    AutoModel,
    AutoTokenizer,
)
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from datasets import SubmissionsDataset
from vit_pytorch import ViT
import torch
import pickle
import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Do stuff.")
parser.add_argument("--modality", type=str, required=True)
parser.add_argument("--embs", type=str, required=True)

args = parser.parse_args()
modality = args.modality
embs = args.embs

device = "cuda"

DATA_PATH = [
    "../../data/notdepressed/notdepressed1-reddit_crawling_2021-06-01 11:49:29",
    "../../data/depressed/depressed-reddit_crawling_2021-05-26 13:32:03",
]

EXPERIMENTS_DATA_PATH = "../../data/splits/for_experiments"
# EMBEDDINGS_PATH = '../embeddings/for_experiments'
EMBEDDINGS_PATH = "../embeddings_correct/for_experiments"

embs_type = {
    "image": {
        "clip": {
            "model_name": "openai/clip-vit-base-patch16",
            "model_class": CLIPVisionModel,
            "input_representation": CLIPProcessor,
        },
        "dino": {
            "model_name": "facebook/dino-vitb16",
            "model_class": ViTModel,
            "input_representation": ViTFeatureExtractor,
        },
    },
    "text": {
        "bert": {
            "model_name": "bert-base-uncased",
            "model_class": BertModel,
            "input_representation": BertTokenizer,
        },
        "roberta": {
            "model_name": "sentence-transformers/stsb-roberta-base",
            # 'model_name': 'roberta-base',
            "model_class": RobertaModel,
            "input_representation": RobertaTokenizer,
        },
        "minilm": {
            "model_name": "microsoft/Multilingual-MiniLM-L12-H384",
            "model_class": AutoModel,
            "input_representation": AutoTokenizer,
        },
        "emoberta": {
            "model_name": "tae898/emoberta-base",
            "model_class": AutoModel,
            "input_representation": AutoTokenizer,
        },
        "clip": {
            "model_name": "openai/clip-vit-base-patch32",
            "model_class": CLIPModel,
            "input_representation": CLIPProcessor,
        },
    },
}

def extract_text_embedding_sent(dataset, model, embs, kind):

    BATCH_SIZE = 128
    text_encoder = SentenceTransformer(model).cuda()

    for i in tqdm.tqdm(range(len(dataset))):

        sample = dataset[i]
        user = sample["author"]

        path = f"{EMBEDDINGS_PATH}/{kind}/{user}"
        os.makedirs(path, exist_ok=True)

        if (os.path.exists(f"{path}/{embs}.pkl")) and not os.stat(
            f"{path}/{embs}.pkl"
        ).st_size == 0:
            continue

        encoded_texts = text_encoder.encode(
            sample["texts"], batch_size=BATCH_SIZE, convert_to_numpy=True
        )

        with open(f"{path}/{embs}.pkl", "wb") as f:
            pickle.dump(encoded_texts, f)


def extract_image_embedding(dataset, input_representation, model, embs, kind):

    for i in tqdm.tqdm(range(len(dataset))):

        sample = dataset[i]
        user = sample["author"]
        path = f"{EMBEDDINGS_PATH}/{kind}/{user}"
        os.makedirs(path, exist_ok=True)

        embeddings_list = []

        if (os.path.exists(f"{path}/{embs}.pkl")) and not os.stat(
            f"{path}/{embs}.pkl"
        ).st_size == 0:
            continue

        for id, image in zip(sample["ids"], sample["images"]):
            submission_dict = {"image": {}, "text": {}}

            if image is np.nan or image.shape == (1, 1, 3):
                embedding = np.nan

            elif image is not np.nan:

                inputs = input_representation(
                    images=image.transpose(2, 0, 1), return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    out = model(**inputs)
                    embedding = out.last_hidden_state.mean(dim=1).detach().cpu().numpy()

            embeddings_list.append(embedding)

        with open(f"{path}/{embs}.pkl", "wb") as f:
            pickle.dump(embeddings_list, f)


def get_embeddings_patches():

    print("MODALITY ", modality, "EMBS_TYPE ", embs)
    if modality == "image":
        input_representation = embs_type[modality][embs][
            "input_representation"
        ].from_pretrained(embs_type[modality][embs]["model_name"])
        model = embs_type[modality][embs]["model_class"].from_pretrained(
            embs_type[modality][embs]["model_name"]
        )
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        model = model.to(device)

    for kind in ["train", "valid", "test"]:
        print(kind)
        dataset = SubmissionsDataset(kind=kind)

        if modality == "image":
            extract_image_embedding(dataset, input_representation, model, embs, kind)

        elif modality == "text":
            extract_text_embedding_sent(
                dataset, embs_type[modality][embs]["model_name"], embs, kind
            )

get_embeddings_patches()
