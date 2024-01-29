class DataLoader:
    def __len__(self) -> int:
        raise TypeError

    def __iter__(self) -> Iterator[TensorDict]:
            raise NotImplementedError

    def iter_instances(self) -> Iterator[Instance]:
        raise NotImplementedError

    def index_with(self, vocab: Vocabulary) -> None:
        raise NotImplementedError

    def set_target_device(self, device: torch.device) -> None:
        raise NotImplementedError

class AmexDataLoader(DataLoader):
    def __init__(
        self,
        data_series: pd.Series,
        data_feature: pd.DataFrame,
        data_label: pd.DataFrame,
        uidxs: List[str],
        config: DictConfig,
    ):
        self.data_series = data_series
        self.data_feature = data_feature
        self.data_label = data_label
        self.uidxs = uidxs
        self.config = config

    def __len__(self):
        return (len(self.uidxs))

    def __getitem__(self):
