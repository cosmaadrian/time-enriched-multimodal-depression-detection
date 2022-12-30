import pickle
import torch.nn as nn
import torch

import torch.nn.functional as F
import numpy as np
from sklearn import metrics

from .acumen_trainer import AcumenTrainer


class Trainer(AcumenTrainer):
    def __init__(self, args, model, class_weights=None):
        super().__init__()
        self.args = args

        self.weights = torch.Tensor(np.array([1.0, 1.0]).astype(np.float32)).cuda()
        if class_weights is not None:
            self.weights = torch.from_numpy(class_weights.astype(np.float32)).cuda()

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.model = model

    def configure_optimizers(self, lr=0.00001):
        self._optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )

        return self._optimizer

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss_non_reduced = self.criterion(output["logits"], batch["label"])

        if self.weights is None:
            loss = torch.mean(loss_non_reduced)
        else:
            batch_weights = torch.cat([1 - batch["label"], batch["label"]], dim=-1)
            batch_weights = batch_weights * self.weights
            batch_weights = batch_weights.sum(dim=-1)
            loss = batch_weights.view(-1) * loss_non_reduced.view(-1)
            loss = torch.mean(loss)

        self.log("train/loss", loss.item(), on_step=True)

        return loss

    def validation_step(self, batch, i):
        output = self.model(batch)
        loss = self.criterion(output["logits"], batch["label"])
        loss = torch.mean(loss)

        self.log("val_loss", loss.item(), on_step=True, force_log=True)

        return (
            output["probas"].detach().cpu().numpy(),
            batch["label"].detach().cpu().numpy(),
        )

    def validation_epoch_end(self, outputs):
        out, labels = zip(*outputs)
        out = np.vstack(out)
        labels = np.vstack(labels)

        acc = metrics.accuracy_score(labels, np.round(out))
        print("::: Accuracy", acc)
        self.log("val/accuracy", acc, on_step=False, force_log=True)

        fpr, tpr, thresholds = metrics.roc_curve(labels, out, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("::: AUC", auc)
        self.log("val/auc", auc, on_step=False, force_log=True)

        precision = metrics.precision_score(labels, np.round(out))
        self.log("val/precision", precision, on_step=False, force_log=True)
        print("::: Precision", precision)

        recall = metrics.recall_score(labels, np.round(out))
        self.log("val/recall", recall, on_step=False, force_log=True)
        print("::: Recall", recall)

        f1 = metrics.f1_score(labels, np.round(out))
        self.log("val_f1", f1, on_step=False, force_log=True)
        print("::: F1", f1)
