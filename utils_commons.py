
import os
import sys
import torch
import random
import traceback
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime

def initialize_logger(exp_name, stdout='DEBUG'):
    start_time = datetime.now()
    logger.remove()
    log_dir = Path("logs") / exp_name / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level=stdout)
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    sys.excepthook = lambda _, value, tb: logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
    logger.info(" ".join(sys.argv))
    logger.info(f"The outputs are being saved in {log_dir}")
    return log_dir

def make_deterministic(seed: int = 0):
    seed = int(seed)
    if seed == -1:
        return
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from time import time
class Timer:
    def __init__(self):
        self.start_time = time()
    def __call__(self, s):
        print(f"{s}: {time() - self.start_time:.4f} seconds")
        self.start_time = time()