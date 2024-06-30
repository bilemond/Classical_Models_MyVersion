import os
import random
import numpy as np
import torch
import time
import copy
import torch.nn as nn

def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, 'log.txt')
        self.log = open(self.log_file, 'w')

    def write(self, message):
        print(message)
        self.log.write(f'{message}\n')

    def close(self):
        self.log.close()


class Timer:
    def __init__(self):
        self.tik = None

    def start(self):
        self.tik = time.time()

    def stop(self):
        return time.time() - self.tik

def clones(module, N):
    """
    module克隆函数
    :param module: 被克隆的module
    :param N: 克隆的次数
    :return: ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def pair(t):
    """
    :return: t if t is a tuple, else (t, t)
    """
    return t if isinstance(t, tuple) else (t, t)