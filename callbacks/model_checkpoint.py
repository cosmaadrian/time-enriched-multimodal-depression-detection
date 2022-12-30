from .callback import Callback
import os
import torch


class ModelCheckpoint(Callback):
    def __init__(
        self,
        monitor="val_loss",
        direction="down",
        dirpath="checkpoints/",
        save_weights_only=False,
        filename="checkpoint",
        save_best_only=True,
    ):
        self.trainer = None
        self.monitor = monitor
        self.direction = direction
        self.dirpath = dirpath
        self.save_weights_only = save_weights_only
        self.filename = filename
        self.save_best_only = save_best_only

        self.previous_best = None
        self.previous_best_path = None

    def on_epoch_end(self):

        trainer_quantity = self.trainer.logger.metrics[self.monitor]

        if self.previous_best is not None:
            if self.direction == "down":
                if self.previous_best <= trainer_quantity:
                    print(
                        f"No improvement. Current: {trainer_quantity} - Previous {self.previous_best}"
                    )
                    return
            else:
                if self.previous_best >= trainer_quantity:
                    print(
                        f"No improvement. Current: {trainer_quantity} - Previous {self.previous_best}"
                    )
                    return

        if self.previous_best_path is not None:
            os.unlink(self.previous_best_path)

        path = os.path.join(
            self.dirpath,
            self.filename.format(
                **{"epoch": self.trainer.epoch, self.monitor: trainer_quantity}
            ),
        )

        print(f"🔥 Saving model to: {path}")

        os.makedirs(self.dirpath, exist_ok=True)

        self.previous_best = trainer_quantity
        self.previous_best_path = path

        # TODO SAVE SCHEDULER / GRADSCALER / OPTIMIZER

        if self.save_weights_only:
            torch.save(self.trainer.model_hook.state_dict(), path)
        else:
            torch.save(self.trainer.model_hook, path)
