import lightgbm as lgb
import numpy as np
import os
from pathlib import Path
from typing import NoReturn, Callable
from models.base import AbstractModel
from omegaconf import DictConfig
from v2.models.callbacks import CallbackEnv

class LightGBM(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_score = float('-inf')

    def _save_checkpoint(self) -> Callable[[CallbackEnv], NoReturn]:
        """Creates a callback to save the model checkpoint based on improved score."""
        def callback(env: CallbackEnv) -> None:
            """The callback function for saving checkpoints."""
            score_index = 1 if self.config.model.loss.is_customized else 3
            score = env.evaluation_result_list[score_index][2]
            if self._max_score < score:
                self._max_score = score
                model_name = f"{self.config.model.name}_fold{self._num_fold_iter}.lgb"
                model_path = Path(self.config.model.path) / model_name

                # Remove the existing file if it already exists
                if model_path.is_file():
                    os.remove(model_path)

                # Save the model
                env.model.save_model(model_path)

            callback.order = 0
            return callback

        return callback

    def _train(self, X_train, y_train, X_valid, y_valid) -> lgb.Booster:
        """Trains the LightGBM model."""
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=[*self.config.features.cat_features]
        )
        valid_data = lgb.Dataset(
            X_valid, label=y_valid,
            categorical_feature=[*self.config.features.cat_features]
        )

        # Train the model with callbacks for checkpointing and early stopping
        self.model = lgb.train(
            params=dict(self.config.model.params),
            train_set=train_data,
            valid_sets=[train_data, valid_data],
            verbose_eval=self.config.model.verbose_eval,
            callbacks=[
                self._save_checkpoint(),
                lgb.early_stopping(self.config.model.early_stopping_rounds, first_metric_only=True),
                lgb.log_evaluation(self.config.model.verbose)
            ]
        )
        return self.model

    def predict(self, results, test_x) -> np.ndarray:
        """Predicts using the trained LightGBM models."""
        folds = len(results.models)
        preds_proba = np.zeros((test_x.shape[0], folds))

        for i, model in enumerate(results.models):
            preds_proba[:, i] = model.predict(test_x)

        # Average over the folds
        return preds_proba.mean(axis=1)

    def save(self, path):
        """Saves the LightGBM model."""
        if not self.model:
            raise ValueError("No model is trained to be saved.")
        self.model.save_model(path)

    def load(self, path):
        """Loads a LightGBM model."""
        self.model = lgb.Booster(model_file=path)
