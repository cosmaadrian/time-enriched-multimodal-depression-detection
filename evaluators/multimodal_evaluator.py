import torch
from scipy import stats
import json
import numpy as np
import os

import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn import metrics
from evaluators import BaseEvaluator

import nomenclature


class MultimodalEvaluator(BaseEvaluator):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.num_runs = 10
        self.dataset = nomenclature.DATASETS[self.args.dataset]
        self.test_dataset = self.dataset(args=args, kind="test")
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            num_workers=15,
            pin_memory=False,
            shuffle=False,
        )

    def trainer_evaluate(self, step):
        print("Running Evaluation.")
        results = self.evaluate(save=False)
        return results[-1]["f1"]

    def evaluate(self, save=True):
        y_preds = []
        y_preds_proba = []
        true_labels = []

        for _ in range(self.num_runs):
            y_pred = []
            y_pred_proba = []
            true_label = []

            with torch.no_grad():
                for i, batch in enumerate(
                    tqdm(self.test_dataloader, total=len(self.test_dataloader))
                ):
                    for key, value in batch.items():
                        batch[key] = value.to(nomenclature.device)

                    output = self.model(batch)["probas"]

                    preds = np.vstack(output.detach().cpu().numpy()).ravel()
                    labels = np.vstack(batch["label"].detach().cpu().numpy()).ravel()

                    y_pred.extend(np.round(preds))
                    y_pred_proba.extend(preds)
                    true_label.extend(labels)

            y_preds.append(y_pred)
            y_preds_proba.append(y_pred_proba)
            true_labels.append(true_label)

        y_preds = np.array(y_preds)
        y_preds_proba = np.array(y_preds_proba)
        true_labels = np.array(true_labels)

        y_preds_voted = stats.mode(y_preds).mode[0]
        true_labels = stats.mode(true_labels).mode[0]
        y_preds_proba = y_preds_proba.mean(axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(
            true_labels, y_preds_proba, pos_label=1
        )
        acc = metrics.accuracy_score(true_labels, y_preds_voted)
        auc = metrics.auc(fpr, tpr)
        precision = metrics.precision_score(true_labels, y_preds_voted)
        recall = metrics.recall_score(true_labels, y_preds_voted)
        f1 = metrics.f1_score(true_labels, y_preds_voted)

        results = pd.DataFrame.from_dict(
            {
                "f1": [f1],
                "recall": [recall],
                "precision": [precision],
                "auc": [auc],
                "accuracy": [acc],
                "name": [f"{self.args.group}:{self.args.name}"],
                "dataset": [self.args.dataset],
                "text_embedding": [self.args.text_embeddings_type],
                "image_embedding": [self.args.image_embeddings_type],
                "window_size": [self.args.window_size],
                "position_embedding": [self.args.position_embeddings],
                "fold": [self.args.fold],
                "modality": [self.args.modality],
            }
        )

        if save:
            results.to_csv(
                f"results/{self.args.output_dir}/{self.args.group}:{self.args.name}.csv",
                index=False,
            )

        return results
