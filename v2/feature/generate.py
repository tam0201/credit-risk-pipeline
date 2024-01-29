from re import L
from tqdm import tqdm
import pandas as pd
from omegaconf import DictConfig

tqdm.pandas()

class InferencerMeta(ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.preprocess_kwargs, set)
        assert isinstance(self.forward_kwargs, set)

        all_kwargs = (
            self.process_kwargs | self.forward_kwargs
        )

        assert len(all_kwargs) == (len(self.preprocess_kwargs) + len(self.forward_kwargs))

class BaseInference(metaclass=InferencerMeta):
    preprocess_kwargs: set = set()
    forward_kwargs: set = set()

    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def preprocess(self, df: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def forward(self, df: pd.DataFrame, **kwargs):
        pass

    def __call__(self, df: pd.DataFrame, **kwargs):
        return self.forward(self.preprocess(df, **kwargs), **kwargs)
