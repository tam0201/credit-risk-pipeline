import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd 

class SparseTensorDataset(Dataset):
    def __init__(self, x_sparse, y_dense):
        self.x_sparse = x_sparse
        self.y_dense = y_dense
        if isinstance(y_dense, pd.Series) or isinstance(y_dense, pd.DataFrame):
            self.y_dense = y_dense.reset_index(drop=True).values

    def __len__(self):
        return self.x_sparse.shape[0]

    def __getitem__(self, idx):
        if isinstance(self.x_sparse, pd.DataFrame):
            # If it's a pandas DataFrame, use iloc to get the row
            x_tensor = torch.from_numpy(self.x_sparse.iloc[idx].values).float()
        else:
            # If it's a scipy sparse matrix, use the toarray method after indexing
            x_tensor = torch.from_numpy(self.x_sparse[idx].toarray().reshape(-1)).float()

        # Handle y_dense based on whether it's an ndarray or a DataFrame/Series
        if isinstance(self.y_dense, (np.ndarray, torch.Tensor)):
            y_tensor = torch.tensor(self.y_dense[idx], dtype=torch.float32)
        else:
            # If y_dense is a DataFrame or Series, use iloc for proper row selection
            y_tensor = torch.tensor(self.y_dense.iloc[idx], dtype=torch.float32)

        return x_tensor, y_tensor

class DataLoaderFactory:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def create_dataloader(self, X, y, model_type):
        """Creates DataLoader based on model type."""
        if model_type == 'nn':
            return self._create_nn_dataloader(X, y)
        elif model_type == 'gbm':
            return None
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _create_nn_dataloader(self, X, y):
        """Creates DataLoader for neural network models."""
        # Check if X is a sparse matrix
        if isinstance(X, np.ndarray):
            tensor_x = torch.tensor(X.astype(np.float32)) if not isinstance(X, torch.Tensor) else X
            tensor_y = torch.tensor(y.astype(np.float32)) if not isinstance(y, torch.Tensor) else y
            dataset = TensorDataset(tensor_x, tensor_y)
        else:  # Handle sparse matrix for X
            dataset = SparseTensorDataset(X, y)
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
