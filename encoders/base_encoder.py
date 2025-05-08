import torch.nn as nn
from abc import ABC, abstractmethod

class BaseEncoder(nn.Module, ABC):
    def __init__(self):
        super(BaseEncoder, self).__init__()

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x: Input tensor
        Returns:
            torch.Tensor: Encoded output tensor (embeddings)
        """
        pass