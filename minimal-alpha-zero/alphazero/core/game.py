from abc import ABC, abstractmethod
from typing import Optional


class Action(ABC):
    """ """

    @abstractmethod
    def __eq__(self, value):
        """
        Used to distinguish it from other actions.
        For example, when checking its presence in a state's legal actions list (ref: `Game.list_legal_actions` method).
        """

    @abstractmethod
    def __hash__(self):
        """
        Used as a dictionary key (ref: `.generator.Node` class).
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
        The order of actions is important. Ref: `.model.Model` class.
        """

    @abstractmethod
    def list_legal_actions(self, state: State) -> list[Action]:
        """
        TODO:
        Consider returning an empty list of legal actions for a terminal state.
        Currently, this is handled externally in `.generator._expand` function.
        """

    @abstractmethod
    def simulate(self, state: State, action: Action) -> State:
        """
        Simulate the move of an action from a given state to a new state without actually playing the move.
        """

    @abstractmethod
    def receive_reward_if_terminal(self, state: State) -> Optional[float]:
        """
        If the state is terminal, return a final reward with a positive value if the last player wins,
        a negative value if they lose, and 0 in the case of a draw.
        If the state is not terminal, return `None`.
        NOTE:
        Remember, the reward is not assigned to this state, but to the move made by the last player,
        i.e., the action from the previous state that results in this state.
        This is crucial for generating data and for running backups to update action values during MCTS simulations.
        Ref: `.generator.generate_data` and `.generator._backup` functions.
        """


class ReplayBuffer:
    """ """

    # `list[float]` is a list of probabilities, ordered to match the actions returned by `Game.list_all_actions` method.
    # Ref: `.generator.generate_data` function.
    # NOTE: The "probabilities" stored in the replay buffer are referred to as "search probabilities" when generating data,
    # and as "improved probabilities" during training; they are interchangeable.
    buffer_queue: list[list[tuple[State, list[float], float]]]
    buffer_queue_size: int

    def __init__(self, buffer_queue_size: int = 1):
        self.buffer_queue = []
        self.buffer_queue_size = buffer_queue_size

    def enqueue_new_buffer(self):
        """ """
        self.buffer_queue.append([])

    def append_to_newest_buffer(self, data: tuple[State, list[float], float]):
        """ """
        if len(self.buffer_queue) == 0:
            self.enqueue_new_buffer()
        self.buffer_queue[-1].append(data)

    def discard_oldest_buffers(self):
        """ """
        while len(self.buffer_queue) >= self.buffer_queue_size:
            self.buffer_queue.pop(0)
