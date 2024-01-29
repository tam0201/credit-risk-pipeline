import collections

CallbackEnv = collections.namedtuple(
    "CallbackEnv",
    [
        "model",
        "params",
        "iteration",
        "begin_iteration",
        "end_iteration",
        "eval_results",
    ],
)

class DartEarlyStopException(Exception):
    """Exception of early stopping in DART."""

    def __init__(self,):
        """Create instance of DartEarlyStopException."""
        self.best_score = None
        self.best_model = None
        self.best_iteration = None

    def __call__(self, env: CallbackEnv):
        pass
