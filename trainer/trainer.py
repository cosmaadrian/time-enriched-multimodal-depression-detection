import torch
import tqdm
import torch.nn as nn
import numpy as np


class NotALightningTrainer(object):
    def __init__(self, args, callbacks, logger):
        self.args = args

        self.epoch = 0
        self.global_step = 0

        self.logger = logger
        self.logger.trainer = self

        self.callbacks = callbacks
        for callback in callbacks:
            callback.trainer = self

        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def fit(self, model, train_dataloader, val_dataloader):
        model.log = self.logger.log

        optimizer = model.configure_optimizers()

        if not hasattr(model.model, "module"):
            # distributed data parallel??
            model.model = nn.DataParallel(model.model, device_ids=[0, 1])
            model.model = model.model.cuda()

        self.logger.watch(model.model)
        self.model_hook = model.model

        for epoch in tqdm.tqdm(range(self.args.epochs)):

            if self.should_stop:
                break

            for callback in self.callbacks:
                callback.on_epoch_start()

            pbar = tqdm.tqdm(train_dataloader, total=len(train_dataloader))

            model.training_epoch_start(epoch)
            for i, data in enumerate(pbar):
                self.global_step += 1
                optimizer.zero_grad()

                for callback in self.callbacks:
                    callback.on_batch_start()

                for key in data.keys():
                    data[key] = data[key].cuda()

                loss = model.training_step(data, i)
                loss = loss / self.args.accumulation_steps

                loss.backward()

                if (i + 1) % self.args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                    optimizer.step()

                    for callback in self.callbacks:
                        callback.on_batch_end()

                pbar.set_description(
                    f"Epoch {self.epoch} / {self.args.epochs} | "
                    + " | ".join(
                        [
                            f"{k}={np.round(v, 4)}"
                            for k, v in self.logger.on_step_metrics.items()
                        ]
                    )
                )

            model.training_epoch_end()
            self.epoch += 1
            model.model.train(False)
            with torch.no_grad():
                outputs = []
                pbar_eval = tqdm.tqdm(val_dataloader, total=len(val_dataloader))
                for i, data in enumerate(pbar_eval):
                    for key in data.keys():
                        data[key] = data[key].cuda()
                    out = model.validation_step(data, i)
                    outputs.append(out)
                    pbar_eval.set_description(f"Validating")

                model.validation_epoch_end(outputs)

            model.model.train(True)

            for callback in self.callbacks:
                callback.on_epoch_end()
