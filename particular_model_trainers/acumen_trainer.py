import torch
import torch.nn as nn


class AcumenTrainer(object):
    def configure_optimizers(self, lr=0.1):
        if self._optimizer is not None:
            return self._optimizer

        self._optimizer = torch.optim.AdamW(
            filter(lambda x: x.requires_grad, self.model.parameters()), lr
        )

        return self._optimizer

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, epoch=None):
        pass

    def training_epoch_start(self, epoch=None):
        pass
