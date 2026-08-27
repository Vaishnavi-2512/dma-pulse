import numpy as np
from sklearn.preprocessing import MinMaxScaler


class PulsePreprocessor:
    """Preprocess tabular activity data into fixed-length sequences."""

    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()

    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        shape = X.shape
        flat = X.reshape(-1, shape[-1]) if X.ndim == 3 else X
        flat = self.scaler.fit_transform(flat)
        if X.ndim == 3:
            return flat.reshape(shape).astype(np.float32)
        return flat.astype(np.float32)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        shape = X.shape
        flat = X.reshape(-1, shape[-1]) if X.ndim == 3 else X
        flat = self.scaler.transform(flat)
        if X.ndim == 3:
            return flat.reshape(shape).astype(np.float32)
        return flat.astype(np.float32)
