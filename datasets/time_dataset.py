from torch.utils.data import Dataset
import random
import pandas as pd
import time
import operator
import numpy as np
import pprint


class TimeDataset(Dataset):
    def __len__(self):
        return len(self.users)

    def _emb_size(self, kind):
        if kind == "image":
            return self.args.IMAGE_EMBEDDING_SIZES[self.args.image_embeddings_type]

        if kind == "text":
            return self.args.IMAGE_EMBEDDING_SIZES[self.args.image_embeddings_type]

    def load_multimodal(
        self, image_embeddings, text_embeddings, dates, label, user_name=None
    ):
        image_size = self._emb_size(kind="image")
        if self.args.position_embeddings != "zero":
            order_idx = np.argsort(dates).ravel()
        else:
            order_idx = np.random.permutation(len(image_embeddings))

        high_ = len(dates) - self.window_size

        if high_ <= 0:
            start_idx = 0
            end_idx = len(dates)
            padding_amount = abs(high_)
        else:
            start_idx = np.random.randint(low=0, high=high_)
            end_idx = start_idx + self.window_size
            padding_amount = 0

        idxs = order_idx[start_idx:end_idx]
        text_embeddings = text_embeddings[idxs]
        dates = dates[idxs]

        image_embeddings_ = np.zeros((len(idxs), image_size))
        image_mask = np.zeros(len(idxs))

        zeros = np.zeros((1, image_size), dtype=np.int32)

        for i, idx in enumerate(idxs):
            if type(image_embeddings[idx]) == float:
                image_embeddings_[i, :] = zeros
                image_mask[i] = 0
            else:
                image_embeddings_[i, :] = image_embeddings[idx]
                image_mask[i] = 1

        image_embeddings = image_embeddings_.reshape((len(image_embeddings_), -1))

        # TODO maybe apply log
        if self.args.timestamp_kind == "delta":
            dates = np.hstack(([0], np.diff(dates))) / 60 / 60
        elif self.args.timestamp_kind == "relative":
            dates = (dates - np.min(dates)) / 60 / 60  # hour difference

        # apply padding
        image_mask = np.pad(
            image_mask, (0, padding_amount), "constant", constant_values=0
        )
        text_mask = np.ones(len(dates))
        text_mask = np.pad(
            text_mask, (0, padding_amount), "constant", constant_values=0
        )
        image_embeddings = np.pad(
            image_embeddings,
            ((0, padding_amount), (0, 0)),
            "constant",
            constant_values=0.0,
        )
        text_embeddings = np.pad(
            text_embeddings,
            ((0, padding_amount), (0, 0)),
            "constant",
            constant_values=0.0,
        )
        dates = np.pad(dates, (0, padding_amount), "constant", constant_values=0.0)

        sample = {
            # 'user': user,
            "image_embeddings": image_embeddings.astype(np.float32),
            "text_embeddings": text_embeddings.astype(np.float32),
            "image_mask": image_mask.astype(np.float32).reshape(
                (-1, 1, self.window_size)
            ),
            "text_mask": text_mask.astype(np.float32).reshape(
                (-1, 1, self.window_size)
            ),
            "time": dates.astype(np.float32),
            "label": np.array([label]).astype(np.float32),
        }

        return sample

    def load_singlemodal(self, modality, dates, label, user_name=None):
        modality_size = self._emb_size(kind=self.args.modality)

        modality_ = []
        mask = []
        for i, emb in enumerate(modality):
            if type(emb) == float:
                modality_.append(np.zeros((1, modality_size), dtype=np.int32))
                mask.append(0)
            else:
                modality_.append(emb)
                mask.append(1)

        modality = np.array(modality_).reshape((len(modality_), -1))
        mask = np.array(mask)

        order_idx = np.argsort(dates).ravel()

        # remove nans
        bool_mask = mask.astype(bool)
        order_idx = order_idx[bool_mask]
        modality = modality[bool_mask]
        dates = dates[bool_mask]

        if self.args.position_embeddings != "zero":
            order_idx = np.argsort(dates).ravel()
            modality = modality[order_idx]
            mask = mask[order_idx]
            dates = dates[order_idx]
        else:
            p = np.random.permutation(len(modality))
            modality = modality[p]
            mask = mask[p]
            dates = dates[p]

        high_ = len(dates) - self.window_size

        if high_ <= 0:
            start_idx = 0
            end_idx = len(dates)
            padding_amount = abs(high_)
        else:
            start_idx = np.random.randint(low=0, high=high_)
            end_idx = start_idx + self.window_size
            padding_amount = 0

        modality = modality[start_idx:end_idx]
        mask = mask[start_idx:end_idx]

        dates = dates[start_idx:end_idx]

        # TODO maybe apply log
        if self.args.timestamp_kind == "delta":
            dates = np.hstack(([0], np.diff(dates))) / 60 / 60
        elif self.args.timestamp_kind == "relative":
            dates = (dates - np.min(dates)) / 60 / 60  # hour difference

        # apply padding
        mask = np.pad(mask, (0, padding_amount), "constant", constant_values=0)
        modality = np.pad(
            modality, ((0, padding_amount), (0, 0)), "constant", constant_values=0.0
        )
        dates = np.pad(dates, (0, padding_amount), "constant", constant_values=0.0)

        sample = {
            "modality": modality.astype(np.float32),
            "time": dates.astype(np.float32),
            "label": np.array([label]).astype(np.float32),
        }

        return sample
