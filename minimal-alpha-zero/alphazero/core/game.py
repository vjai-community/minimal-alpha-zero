import math
from abc import ABC, abstractmethod
from typing import Optional


class Action(ABC):
    """ """

    @abstractmethod
    def __eq__(self, value):
        """
        Used to distinguish it from other actions.
        For example, when checking its presence in a state's legal actions list (ref: `Game.get_legal_actions` method).
        """

    @abstractmethod
    def __hash__(self):
        """
        Used as a dictionary key (ref: `Node` class).
        """


class InputData(ABC):
    """ """


class State(ABC):
    """
    NOTE: A state should include information indicating which player is in turn.
    """

    @abstractmethod
    def make_input_data(self) -> InputData:
        """ """


class Game(ABC):
    """ """

    @abstractmethod
    def begin(self) -> State:
        """
        Begin a new game, reset all state, and return a new initial state.
        """

    @abstractmethod
    def list_all_actions(self) -> list[Action]:
        """
        The order of actions is important. Ref: `Model.__call__` method.
        """

    @abstractmethod
    def list_legal_actions(self, state: State) -> list[Action]:
        """ """

    @abstractmethod
    def simulate(self, state: State, action: Action) -> State:
        """
        Simulate the move of an action from a given state to a new state without actually playing the move.
        """

    @abstractmethod
    def receive_reward_if_terminal(self, state: State) -> Optional[float]:
        """
        If the state is terminal, return a final reward of 1 if current player wins, -1 if they lose, and 0 in the case of a draw.
        If the state is not terminal, return `None`.
        """


class ReplayBuffer:
    """ """

    # `list[float]` is a list of probabilities, ordered to match the actions returned by `Game.list_all_actions` method.
    # Ref: `generator.play` function.
    buffer: list[tuple[State, list[float], float]]
    buffer_size: int

    def __init__(self, buffer_size: int):
        self.buffer = []
        self.buffer_size = buffer_size

    def append(self, data: tuple[State, list[float], float]):
        """ """
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(data)

    def reset(self):
        """ """
        self.buffer = []


def calculate_legal_actions(
    prior_probabilities: dict[Action, float],
    game: Game,
    state: State,
    temperature: float = 1.0,
) -> tuple[list[Action], list[float]]:
    legal_actions = game.list_legal_actions(state)
    legal_action_prior_probabilities = {a: p for a, p in prior_probabilities.items() if a in legal_actions}
    legal_prior_probabilities = _softmax(list(legal_action_prior_probabilities.values()), temperature)
    return legal_actions, legal_prior_probabilities


def _softmax(logits: list[float], temperature: float) -> list[float]:
    exp_values = [math.exp(l / temperature) for l in logits]  # noqa: E741 (https://docs.astral.sh/ruff/rules/ambiguous-variable-name/)
    exp_values_sum = sum(exp_values)
    return [v / exp_values_sum for v in exp_values]
