from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class SignatureVerifier(ABC):

    @abstractmethod
    def preprocess_signature(self, signature: pd.DataFrame, n_points=100) -> np.ndarray:
        pass

    @abstractmethod
    def verify(self, test_signature: pd.DataFrame):
        pass

    @abstractmethod
    def print_result(self, result, is_genuine):
        pass