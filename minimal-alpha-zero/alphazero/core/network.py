from abc import ABC, abstractmethod

from .game import Action, InputData, Game, ReplayBuffer


class Model(ABC):
    """
    Inner model of the network.
    """

    @abstractmethod
    def __call__(self, x) -> tuple:
        """
        Run inference on input data, then return the predicted prior probabilities and value.
        NOTE:
        The order of prior probabilities must match the order of actions returned by `.game.Game.list_all_actions` method.
        This is because `.game.Game.list_all_actions` controls the order of improved probabilities which used as target data,
        and consistent ordering is crucial for calculating the loss gradient during training.
        Ref: `.game.ReplayBuffer` class.
        """


class Network(ABC):
    """ """

    @abstractmethod
    def best_model_predict_single(self, input_data: InputData) -> tuple[dict[Action, float], float]:
        """
        Predict prior probabilities (policy over all actions, including illegal ones) and value of a given state.
        The prior probabilities are logits; therefore, a softmax must be applied during inference.
        The value will then be trained to be close to 1 if the current player is predicted to win,
        in contrast, it should tend toward -1 if the opponent is predicted to win.
        NOTE:
        In the paper "Mastering the game of Go without human knowledge", authors use "evaluate",
        but I prefer "predict" to distinguish it from "evaluator", which is used for choosing the best networks.
        """

    @abstractmethod
    def train_and_evaluate(self, replay_buffer: ReplayBuffer, game: Game):
        """
        Train and evaluate networks to choose the best for generating new data.
        """
