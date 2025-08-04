import random
import numpy as np
import torch
import logging

def setup_logger():
    """Sets up a basic logger for consistent output format."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """
    Sets the random seed for PyTorch, NumPy, and Python's random module
    to ensure reproducibility of experiments.
    
    Args:
        seed (int): The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # The following two lines are often used for reproducibility,
        # but can impact performance. Enable them if exact reproducibility
        # is more important than speed during training.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")
