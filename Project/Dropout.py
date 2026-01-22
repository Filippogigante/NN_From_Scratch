import numpy as np
from layer import *

class Dropout():
    def __init__(self, rate: float):
        
        """
        :param rate: probability of dropping a neuron (0 < rate < 1)
        """
        if rate <= 0 or rate >= 1:
            raise ValueError("Dropout rate must be in (0, 1)")
        
        self.type = "dropout"
        self.rate = rate
        self.mask = None
        self.training = True

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        if not self.training:
            return x

        # Inverted dropout
        self.mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
        return x * self.mask

    def backward_pass(self, delta: np.ndarray, input, m) -> np.ndarray:
        return delta * self.mask

    def set_training(self):
        self.training = True

    def set_validation(self):
        self.training = False

    def get_type(self):
        return self.type