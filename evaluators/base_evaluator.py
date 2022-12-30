import os


class BaseEvaluator(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model

        if "output_dir" in self.args and self.args.output_dir is not None:
            os.makedirs(f"results/{self.args.output_dir}", exist_ok=True)

    def evaluate(self):
        raise NotImplementedError

    def trainer_evaluate(self):
        raise NotImplementedError
