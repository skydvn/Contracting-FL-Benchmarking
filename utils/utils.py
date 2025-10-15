# utils.py
import os
import random
import numpy as np
import torch

def mkdirs_if_needed(base_path, subdirs=None):
    if subdirs is None:
        subdirs = []
    for s in subdirs:
        path = os.path.join(base_path, s)
        os.makedirs(path, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
