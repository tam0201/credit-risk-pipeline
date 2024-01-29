import gc
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from models.base import AbstractModel


class Tabnet(AbstractModel):
    def __init__(self, num_features, num_classes, num_layers=1, num_decision_steps=5, relaxation_factor=1.5, batch_momentum=0.7, virtual_batch_size=128, num_groups=1, epsilon=1e-15, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.build()

    def build(self):
        self.shared_block = SharedBlock(self.num_features, self.num_groups, self.epsilon)
        self.initial_step = InitialStep(self.num_features, self.num_classes, self.num_groups, self.epsilon)
        self.decision_steps = nn.ModuleList([DecisionStep(self.num_features, self.num_classes, self.num_groups, self.epsilon) for _ in range(self.num_decision_steps)])
        self.final_step = FinalStep(self.num_features, self.num_classes, self.num_groups, self.epsilon)

    def _train(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
    ) -> TabNetClassifier:
        self.model = TabNetClassifier(
            **self._get_default_params()
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_train.to_numpy(), y_train.to_numpy()),
                (X_valid.to_numpy(), y_valid.to_numpy()),
            ]
            max_epochs=self.config.model.max_epochs,
            patience=self.config.model.patience,
            batch_size=self.config.model.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            num_workers=self.config.model.num_workers,
            drop_last=self.config.model.drop_last,
            pin_memory=self.config.model.pin_memory,
            eval_name=["train", "val"]
            eval_metric=["auc"],
            callbacks=[
                EarlyStoppingCallback(
                    patience=self.config.model.patience,
                    metric=self.config.model.metric,
                ),
            ],
        )
        return self.model
