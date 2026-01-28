import os
from abc import ABC, abstractmethod

from .game import Action, InputData, Game, ReplayBuffer


class ModelConfig:
    """ """

    should_execute_mcts: bool

    def __init__(
        self,
        *,
        should_execute_mcts: bool = True,
    ):
        self.should_execute_mcts = should_execute_mcts


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

    @abstractmethod
    def predict_single(self, input_data: InputData) -> tuple[dict[Action, float], float]:
        """
        Predict prior probabilities (policy over all actions, including illegal ones) and value of a given state.
        The value will then be trained to be close to 1 if the current player is predicted to win,
        in contrast, it should tend toward -1 if the rival is predicted to win.
        NOTE:
        In the paper "Mastering the game of Go without human knowledge", authors use "evaluate",
        but I prefer "predict" to distinguish it from "evaluator", which is used for choosing the best model.
        """


class Network(ABC):
    """ """

    @abstractmethod
    def train_and_evaluate(self, replay_buffer: ReplayBuffer, game: Game, output_dir: os.PathLike) -> bool:
        """
        Train and evaluate models to choose the best for generating new data.
        Return `True` if the previous best model has been updated with new weights.
        """

    @abstractmethod
    def get_best_model(self) -> Model:
        """ """
