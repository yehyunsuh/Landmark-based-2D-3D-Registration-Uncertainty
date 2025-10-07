import os
import torch
import random
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