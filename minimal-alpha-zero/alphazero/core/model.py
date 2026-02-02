from abc import ABC, abstractmethod
from typing import Optional

from .game import Action, InputData


class ModelConfig:
    """ """

    mcts_simulations_num: Optional[int]

    def __init__(
        self,
        *,
        mcts_simulations_num: Optional[int] = 1,
    ):
        self.mcts_simulations_num = mcts_simulations_num


class Model(ABC):
    """
    Inner model of the network.
    NOTE:
    A model must implement a method, which runs inference on input data, then returns the predicted prior probabilities and value,
    and which will be called inside the `predict_single` method.
    Example signature (where `Array` represents the batch data):
    ```
    def __call__(self, x: Array) -> tuple[Array, Array]:
    ```
    The order of prior probabilities must match the order of actions returned by `Game.list_all_actions` method.
    This is because `Game.list_all_actions` controls the order of improved probabilities which used as target data,
    and consistent ordering is crucial for calculating the loss gradient during training.
    Ref: `ReplayBuffer` class.
    """

    name: str

    def set_name(self, name: str):
        self.name = name

    @abstractmethod
    def predict_single(self, input_data: InputData) -> tuple[dict[Action, float], float]:
        """
        Predict prior probabilities (policy over all actions, including illegal ones) and value of a given state.
        The value will then be trained to be positive if the current player is predicted to win,
        in contrast, it should tend toward negative if the rival is predicted to win.
        NOTE:
        In the paper "Mastering the game of Go without human knowledge", authors use "evaluate",
        but I prefer "predict" to distinguish it from "evaluator", which is used for choosing the best model.
        """
