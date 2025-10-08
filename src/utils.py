import os
import ast
import torch
import random
import argparse
import numpy as np

def set_seed(seed: int = 42):
    """
    Fix random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed value. Default is 42.
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For some environments (like DataLoader workers)
    os.environ["PYTHONHASHSEED"] = str(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Arguemtn \"%s\" is not a list" % (s))
    return v