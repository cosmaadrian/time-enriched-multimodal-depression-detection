import yaml
import os
import glob
import torch


def extend_config(cfg_path, child=None):
    with open(cfg_path, "rt") as f:
        parent_cfg = yaml.load(f, Loader=yaml.FullLoader)

    if child is not None:
        parent_cfg.update(child)

    if "$extends$" in parent_cfg:
        path = parent_cfg["$extends$"]
        del parent_cfg["$extends$"]
        parent_cfg = extend_config(child=parent_cfg, cfg_path=path)

    return parent_cfg


def load_args(args):
    cfg = extend_config(cfg_path=f"{args.config_file}", child=None)

    for key, value in cfg.items():
        if key in args and args.__dict__[key] is not None:
            continue
        args.__dict__[key] = value

    return args, cfg


def load_model(args):
    checkpoint_path = f"{os.path.abspath(os.path.dirname(__file__))}/checkpoints/{args.group}:{args.name}/*.ckpt"
    print("::: Searching for model at", checkpoint_path)
    checkpoints = glob.glob(checkpoint_path)

    if len(checkpoints) > 1:
        print("::: Multiple checkpoints found !!")
        exit(-1)

    try:
        state_dict = torch.load(checkpoints[-1])
        print("::: Found model at:", checkpoints[-1])
    except Exception as e:
        print("No checkpoints found: ", checkpoint_path)
        raise e

    return state_dict
