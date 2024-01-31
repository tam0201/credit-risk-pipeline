import os
import random
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
from loguru import logger as logging
from pathlib import Path 

from data.metrics import amex_metric
from models.gbm import LightGBMTrainer
from data.preprocess import Preprocessor

cfg = OmegaConf.load("config/features/data_features.yaml")

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONASHSEED"] = str(seed)
    np.random.seed(seed)


def load_train_dataset(config: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load train dataset
    Args:
        config: config
    Returns:
        train_x: train dataset
        train_y: train target
    """
    path = Path(os.getcwd()) / config.dataset.path
    logging.info("Loading train dataset...")

    train = pd.read_csv(path / config.dataset.train)
    all_cols = [c for c in list(train.columns) if c not in ['customer_ID','S_2']]
    num_features = [col for col in all_cols if col not in cfg.cat_features]
    # initialize preprocessor
    preprocessor = Preprocessor(
        cat_features=cfg.cat_features,
        num_features=num_features,
        label_name=config.target,
    )
    df = preprocessor.fit_transform(train)
    train_y = df[config.target]
    train_x = df.drop(columns=[*config.drop_features, config.target])
    logging.info(f"train: {train_x.shape}, target: {train_y.shape}")
    
    return train_x, train_y


def load_test_dataset(config: DictConfig) -> pd.DataFrame:
    """
    Load train dataset
    Args:
        config: config
    Returns:
        test_x: test dataset
    """
    path = Path(os.getcwd() / config.dataset.path)
    logging.info("Loading test dataset...")
    test = pd.read_csv(path / config.dataset.test)
    all_cols = [c for c in list(test.columns) if c not in ['customer_ID','S_2']]
    num_features = [col for col in all_cols if col not in cfg.cat_features]
    # initialize preprocessor
    preprocessor = Preprocessor(
        cat_features=cfg.cat_features,
        num_features=num_features,
        labels=config.target,
    )
    test_x = test.drop(columns=[*config.drop_features])
    test_x = preprocessor.transform(test_x)
    logging.info(f"test: {test_x.shape}")

    return test_x

def _main(train_cfg: DictConfig, feature_cfg: DictConfig) -> None:
    seed_everything(train_cfg.params.seed)
    # create dataset
    train_x, train_y = load_train_dataset(feature_cfg)
    logging.debug(train_x)
    # train model
    lgb_trainer = LightGBMTrainer(model_config=train_cfg, data_config=feature_cfg, metric=amex_metric)
    lgb_trainer.train(train_x, train_y)

    # save model
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main(train_cfg=OmegaConf.load("config/model/lgbm.yaml"),
          feature_cfg=OmegaConf.load("config/features/data_features.yaml"))