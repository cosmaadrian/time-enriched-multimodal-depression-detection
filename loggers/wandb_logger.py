import wandb


class WandbLogger(object):
    def __init__(self):
        self.on_step_metrics = dict()
        self.trainer = None

        self.metrics = dict()

    def watch(self, model):
        wandb.watch(model)

    def log(self, key, value, on_step=True, force_log=False):
        self.metrics[key] = value

        if on_step:
            self.on_step_metrics[key] = value

        if (self.trainer.global_step % self.trainer.args.log_every == 0) or force_log:
            wandb.log({key: value}, step=self.trainer.global_step)
            return
