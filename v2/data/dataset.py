from __future__ import annotations

import logging
from pathlib import Path
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import Sampler

class BaseDataset(metaclass=ABCMeta): 
    def __init__(
            self, 
            df_series,
            df_feature,
            df_y=None,
            label_name='LABEL',
            ):
        self.df_series = df_series
        self.df_feature = df_feature
        self.df_y = df_y
        self.label_name = label_name

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass     


class NNDataset(BaseDataset):
    def __init__(self,df_series,df_feature,uidxs,df_y=None):
        self.df_series = df_series
        self.df_feature = df_feature
        self.df_y = df_y
        self.uidxs = uidxs

    def __len__(self):
        return (len(self.uidxs))

    def __getitem__(self, index):
        i1,i2,idx = self.uidxs[index]
        series = self.df_series.iloc[i1:i2+1,1:].values

        if len(series.shape) == 1:
            series = series.reshape((-1,)+series.shape[-1:])
        series_ = series.copy()
        series_[series_!=0] = 1.0 - series_[series_!=0] + 0.001
        feature = self.df_feature.loc[idx].values[1:]
        feature_ = feature.copy()
        feature_[feature_!=0] = 1.0 - feature_[feature_!=0] + 0.001
        if self.df_y is not None:
            label = self.df_y.loc[idx,[self.label_name]].values
            return {
                    'SERIES': series,#np.concatenate([series,series_],axis=1),
                    'FEATURE': np.concatenate([feature,feature_]),
                    'LABEL': label,
                    }
        else:
            return {
                    'SERIES': series,#np.concatenate([series,series_],axis=1),
                    'FEATURE': np.concatenate([feature,feature_]),
                    }

    def collate_fn(self, batch):
        """
        Padding to same size.
        """

        batch_size = len(batch)
        batch_series = torch.zeros((batch_size, 13, batch[0]['SERIES'].shape[1]))
        batch_mask = torch.zeros((batch_size, 13))
        batch_feature = torch.zeros((batch_size, batch[0]['FEATURE'].shape[0]))
        batch_y = torch.zeros((batch_size, 1))

        for i, item in enumerate(batch):
            v = item['SERIES']
            batch_series[i, :v.shape[0], :] = torch.tensor(v).float()
            batch_mask[i,:v.shape[0]] = 1.0
            v = item['FEATURE'].astype(np.float32)
            batch_feature[i] = torch.tensor(v).float()
            if self.df_y is not None:
                v = item['LABEL'].astype(np.float32)
                batch_y[i] = torch.tensor(v).float()

        return {'batch_series':batch_series,'batch_mask':batch_mask,'batch_feature':batch_feature,'batch_y':batch_y}
