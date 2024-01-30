from abc import ABCMeta, abstractmethod
from pydantic import BaseModel as PydanticBaseModel
from typing import Any, Callable, Dict
import numpy as np
from omegaconf import DictConfig

class Result:
    preds: np.ndarray
    models: Dict[str, Any]
    scores: Dict[str, Dict[str, float]]


class AbstractModel(metaclass=ABCMeta):
    """
    An abstract base class that defines a standard interface for machine learning models.
    """
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def _train(self, X_train, y_train):
        """
        Trains the model on the provided training data.

        Parameters:
            X_train: Training feature data.
            y_train: Training target data.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Makes predictions using the trained model.

        Parameters:
            X: Data to make predictions on.

        Returns:
            Predictions for the input data.
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Saves the model to the specified path.

        Parameters:
            path: File path to save the model to.
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Loads the model from the specified path.

        Parameters:
            path: File path to load the model from.
        """
        pass

    def infer(self, result: Result, test_x: pd.Dataframe) -> np.ndarray:
        folds = len(result.models)
        preds_proba = np.zeros((test_x.shape[0],))

        for model in tqdm(result.models.values(), total=folds):
            preds_proba += (
                model.predict(test_x) / folds
                if isinstance(model, lgb.Booster)
                else model.predict_proba(test_x)[:, 1] / folds
            )

        assert len(preds_proba) == len(test_x)

        return preds_proba