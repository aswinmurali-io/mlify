"""Create a Machine Learning environment."""


import os
import random
import numpy as np
import tensorflow as tf

from typing import List, Tuple


# One class that freezes your machine learning environment.
class ModelEnvironment(object):
    def __init__(
        self,
        seed: int = 0,
        cuda: bool = True,
        shuffle: bool = True,
        name: str = "untitled",
        modules: List[str] = [],
        validation_split: float = 0.2,
        batch_size: Tuple[int, int] = (32, 32),
    ) -> None:
        super().__init__()

        self.name: str = name
        self.seed: int = seed
        self.cuda: bool = cuda

        self.shuffle: bool = shuffle
        self.modules: List[str] = modules
        self.batch_size: Tuple[int, int] = batch_size
        self.validation_split: float = validation_split

        self.check_cuda()
        self.seed_everything()
        self.install_modules()

    def seed_everything(self) -> None:
        """Freeze all the random functions.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        tf.random.set_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

    def check_cuda(self) -> None:
        if not tf.test.is_built_with_cuda() and self.cuda:
            print("No Cuda!")

    def install_modules(self) -> None:
        # pip install module -q
        for module in self.modules:
            print(module)
