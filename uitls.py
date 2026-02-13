import os
import torch
import random
import numpy as np
import subprocess

from tqdm import tqdm
from collections import defaultdict


def fix_random_seed(seed):
    '''  Fix all the random seed for reproducibility  '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True





