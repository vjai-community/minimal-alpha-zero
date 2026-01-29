"""
Please read https://en.wikipedia.org/wiki/M,n,k-game, I'm too lazy to write the description.
"""

import csv
import logging
import os
import random
from enum import Enum
from typing import Optional

import optax
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from jax import Array, nn, numpy as jnp
from jax.scipy.special import entr

from ..core.game import Action, InputData, State, Game, ReplayBuffer
from ..core.model import ModelConfig, Model
from ..core.generator import EvaluationConfig, PlayRecord, evaluate


EVALUATION_DUMMY_BEST_DIR_NAME = "evaluation-dummy-best"
EVALUATION_BEST_CANDIDATE_DIR_NAME = "evaluation-best-candidate"
logger = logging.getLogger(__name__)


class StoneColor(Enum):
    """ """

    RED = ("x", "X")
    GREEN = ("o", "O")

    _mark: str
    _big_mark: str

    def __init__(self, mark: str, big_mark: str):
        self._mark = mark
        self._big_mark = big_mark

    def get_mark(self, is_last_stone: bool = False) -> str:
        return self._mark if not is_last_stone else self._big_mark


class Stone:
    """ """

    color: StoneColor

    def __init__(self, color: StoneColor):
        self.color = color


class MnkAction(Action):
    """ """

    x: int
    y: int

    def __init__(self, x: int, y: int):
        super().__init__()
        self.x = x
        self.y = y

    def __eq__(self, value: "MnkAction"):
        return self.x == value.x and self.y == value.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"{self.x}-{self.y}"


class MnkInputData(InputData):
    """ """

    board_data: list[list[float]]

    def __init__(self, board_data: list[list[float]]):
        self.board_data = board_data


class MnkState(State):
    """ """

    SHOULD_USE_FIXED_COLOR_REPRESENTATION = False

    board: list[list[Optional[Stone]]]

    # Used to simplify checking whether the state is terminal.
    stone_count: int
    last_action: Optional[MnkAction]

    def __init__(
        self,
        board: list[list[Optional[Stone]]],
        stone_count: int,
        last_action: Optional[MnkAction],
    ):
        if last_action is not None and board[last_action.y][last_action.x] is None:
            raise ValueError("Invalid last action: its position on the board is empty.")
        self.board = board
        self.stone_count = stone_count
        self.last_action = last_action

    def make_input_data(self) -> MnkInputData:
        """ """
        board_data: list[list[float]]
        if self.SHOULD_USE_FIXED_COLOR_REPRESENTATION:
            # Representing input data from the current player's perspective has some benefits,
            # but in our case we can simply use a fixed value for each color.
            # The reason is that Red always moves first, which means the current player's color for a given state is fixed,
            # therefore, we will theoretically never encounter the opposite perspective (although I might be wrong).
            board_data = [
                [(-1.0 if s.color == StoneColor.RED else 1.0) if s is not None else 0.0 for s in r]  # Fixed stone color
                for r in self.board
            ]
        else:
            # Represent input data from the current player's perspective.
            # TODO:
            # This approach appears to converge too quickly and performs worse on simple games with short training times.
            # We need to identify the cause.
            color = self.get_current_player_color()
            board_data = [
                # The current player's stones are represented as `1`, and the opponent's as `-1`.
                [(1.0 if s.color == color else -1.0) if s is not None else 0.0 for s in r]
                for r in self.board
            ]
        return MnkInputData(board_data)

    def get_current_player_color(self) -> StoneColor:
        """ """
        # Determine whether it is Red's turn.
        is_in_red_turn = (
            self.last_action is None  # Red always moves first
            or self.board[self.last_action.y][self.last_action.x].color == StoneColor.GREEN
        )
        return StoneColor.RED if is_in_red_turn else StoneColor.GREEN

    def __str__(self):
        board_str = ""
        for y, row in enumerate(reversed(self.board)):  # Print rows from top to bottom
            for x, stone in enumerate(row):
                is_last_stone = False
                if self.last_action is not None:
                    is_last_stone = x == self.last_action.x and y == len(self.board) - 1 - self.last_action.y
                board_str += "." if stone is None else stone.color.get_mark(is_last_stone=is_last_stone)
                board_str += " "
            if y < len(self.board) - 1:
                board_str += "\n"
        return board_str


class MnkGame(Game):
    """ """

    #   y
    #   | . . . . . . .
    #   | . . . . . . .
    # n | . . . . . . .
    #   | . . . . . . .
    #   | . . . . . . .
    #     - - - - - - - x
    #           m
    m: int  # Number of columns
    n: int  # Number of rows
    k: int  # Number of stones to connect
    initial_stones_nums: Optional[list[int]]

    # With some choices of m, n, k, it is probably too easy for the first player to win.
    # Hence, we give the winner a larger reward if the game ends quickly and a smaller reward if it lasts longer,
    # hopefully encouraging the second player to prolong the game as much as possible to reduce the reward their opponent receives.
    should_prefer_fast_win: bool

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        initial_stones_nums: Optional[list[int]] = None,
        should_prefer_fast_win: bool = False,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.initial_stones_nums = (
            [n for n in initial_stones_nums if n >= 0 and n < self.m * self.n and n % 2 == 0]  # Keep only even numbers
            if initial_stones_nums is not None
            else None
        )
        self.should_prefer_fast_win = should_prefer_fast_win

    def begin(self) -> MnkState:
        """ """
        state = MnkState([[None for _ in range(self.m)] for _ in range(self.n)], 0, None)
        if self.initial_stones_nums is None or len(self.initial_stones_nums) == 0:
            return state
        positions = [(x, y) for y in range(self.n) for x in range(self.m)]
        initial_stones_num = random.choice(self.initial_stones_nums)
        for x, y in random.sample(positions, initial_stones_num):
            state = self.simulate(state, MnkAction(x, y))
        return state

    def list_all_actions(self) -> list[MnkAction]:
        """ """
        return [MnkAction(x, y) for y in range(self.n) for x in range(self.m)]

    def list_legal_actions(self, state: MnkState) -> list[MnkAction]:
        """ """
        return [MnkAction(x, y) for y, r in enumerate(state.board) for x, s in enumerate(r) if s is None]

    def simulate(self, state: MnkState, action: MnkAction) -> MnkState:
        """
        Only return a new state without storing it in history.
        """
        new_board: list[list[Optional[Stone]]] = []
        for cur_row in state.board:
            new_row: list[Optional[Stone]] = []
            for stone in cur_row:
                new_row.append(Stone(stone.color) if stone is not None else None)
            new_board.append(new_row)
        stone = Stone(state.get_current_player_color())  # Next stone
        stone_count = state.stone_count
        if new_board[action.y][action.x] is None:
            new_board[action.y][action.x] = stone
            stone_count += 1
        else:
            # An illegal action. This should never happen.
            pass
        new_state = MnkState(new_board, stone_count, action)
        return new_state

    def receive_reward_if_terminal(self, state: MnkState) -> Optional[float]:
        """ """
        reward = 1.0  # Reward when there is a winner
        if self.should_prefer_fast_win:
            # Adjust the reward based on the number of stones.
            # max reward (1) = `k * 2` stones (fastest win), min reward (0) = `m * n` stones (board full).
            reward = min((state.stone_count - self.m * self.n) / (self.k * 2 - self.m * self.n), 1.0)
        # The board is empty.
        if state.last_action is None:
            return None  # Not terminal
        last_x, last_y = state.last_action.x, state.last_action.y
        last_color = state.board[last_y][last_x].color
        # Check rows containing the last action. Only consider stones of the same color as the last move.
        m, n, k = self.m, self.n, self.k
        ## The horizontal row.
        row = [state.board[last_y][x] for x in range(m)]
        if self._check_consecutive_stones(row, last_color, k):
            return reward
        ## The vertical row.
        row = [state.board[y][last_x] for y in range(n)]
        if self._check_consecutive_stones(row, last_color, k):
            return reward
        ## The upward diagonal row.
        left_x, left_y = (last_x - last_y, 0) if last_x >= last_y else (0, last_y - last_x)
        row = [state.board[left_y + i][left_x + i] for i in range(min(n - left_y, m - left_x))]
        if self._check_consecutive_stones(row, last_color, k):
            return reward
        ## The downward diagonal row.
        left_x, left_y = (last_x - (n - 1 - last_y), n - 1) if last_x >= n - 1 - last_y else (0, last_y + last_x)
        row = [state.board[left_y - i][left_x + i] for i in range(min(left_y + 1, m - left_x))]
        if self._check_consecutive_stones(row, last_color, k):
            return reward
        # Declare a draw if the board is full. Otherwise, it is not in a terminal state.
        is_full = state.stone_count == m * n
        return 0.0 if is_full else None

    @staticmethod
    def _check_consecutive_stones(row: list[Optional[Stone]], color: StoneColor, k: int) -> bool:
        """ """
        count = 0
        for stone in row:
            if stone is not None and stone.color == color:
                count += 1
                if count == k:
                    return True
            else:
                count = 0
        return False


class MnkModel(nnx.Module, Model):
    """ """

    INPUT_CHANNEL = 1

    def __init__(self, m: int, n: int, rngs: nnx.Rngs):
        # Convolutional
        self.stem_conv = nnx.Sequential(
            nnx.Conv(self.INPUT_CHANNEL, 64, kernel_size=3, rngs=rngs),
            nnx.BatchNorm(64, rngs=rngs),
            nnx.relu,
        )
        self.body_convs = nnx.List(
            [
                nnx.Sequential(
                    nnx.Conv(64, 64, kernel_size=3, rngs=rngs),
                    nnx.BatchNorm(64, rngs=rngs),
                    nnx.relu,
                ),
            ]
        )
        # Fully connected
        self.fc = nnx.Sequential(
            nnx.Linear(64 * m * n, 128, rngs=rngs),
            nnx.relu,
            nnx.Linear(128, 128, rngs=rngs),
            nnx.relu,
        )
        # Heads
        self.prior_probs_head = nnx.Linear(128, m * n, rngs=rngs)
        self.value_head = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, x: Array) -> tuple[Array, Array]:
        x = self.stem_conv(x)
        for body_conv in self.body_convs:
            # TODO: Consider adding a residual connection.
            x = body_conv(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fc(x)
        prior_probs_output = self.prior_probs_head(x)  # Raw logits
        value_output = nnx.tanh(self.value_head(x)).squeeze(-1)
        return prior_probs_output, value_output

    def predict_single(self, input_data: MnkInputData) -> tuple[dict[MnkAction, float], float]:
        """ """
        self.eval()  # Switch to eval mode
        x = jnp.array(input_data.board_data)
        n, m = x.shape  # Fortunately, the action space has the same total size as the input data shape
        x = x.reshape((1, *x.shape, MnkModel.INPUT_CHANNEL))
        prior_probs_output, value_output = self(x)
        prior_probs: dict[MnkAction, float] = {}
        prior_probs_output = nn.softmax(prior_probs_output[0])
        for y in range(n):
            for x in range(m):
                # Use the same layout as `MnkGame.list_all_actions` method to ensure the same action order.
                # Please refer to `..core.model.Model` class for details.
                prior_probs[MnkAction(x, y)] = prior_probs_output[y * m + x].item()
        value: float = value_output.item()
        return prior_probs, value


class DummyModel(Model):
    """ """

    m: int  # Number of columns
    n: int  # Number of rows

    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.name = "dummy"

    def predict_single(self, input_data: MnkInputData) -> tuple[dict[MnkAction, float], float]:
        """ """
        m, n = self.m, self.n
        prior_probs = {MnkAction(x, y): 1.0 / (m * n) for y in range(n) for x in range(m)}
        value = 0.0
        return prior_probs, value


class MnkTrainingConfig:
    """ """

    data_split_ratio: float
    learning_rate: float
    epochs_num: int
    batch_size: int
    stopping_patience: int

    def __init__(
        self,
        *,
        data_split_ratio: float,
        learning_rate: float,
        epochs_num: int,
        batch_size: int,
        stopping_patience: int,
    ):
        self.data_split_ratio = data_split_ratio
        self.learning_rate = learning_rate
        self.epochs_num = epochs_num
        self.batch_size = batch_size
        self.stopping_patience = stopping_patience


class MnkNetwork:
    """ """

    m: int  # Number of columns
    n: int  # Number of rows
    best_model: MnkModel
    rngs: nnx.Rngs

    def __init__(self, m: int, n: int, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.m = m
        self.n = n
        self.best_model = MnkModel(m, n, rngs)
        self.best_model.set_name("best")

    def train_and_evaluate(
        self,
        replay_buffer: ReplayBuffer,
        training_config: MnkTrainingConfig,
        evaluation_config: EvaluationConfig,
        model_config: ModelConfig,
        output_dir: os.PathLike,
        workers_num: Optional[int] = None,
    ) -> bool:
        """ """
        # Prepare the data.
        data_list = [(s.make_input_data(), p, w) for b in replay_buffer.buffer_queue for (s, p, w) in b]
        data_list = random.sample(data_list, len(data_list))
        train_len = int(len(data_list) * training_config.data_split_ratio)
        train_data_list = data_list[:train_len]
        val_data_list = data_list[train_len:]
        # Train a new candidate model.
        candidate_model = nnx.clone(self.best_model)
        candidate_model.set_name("candidate")
        candidate_model.train()  # Switch to train mode
        optimizer = nnx.Optimizer(candidate_model, optax.adamw(training_config.learning_rate), wrt=nnx.Param)
        early_stopping = EarlyStopping(patience=training_config.stopping_patience)
        with open(output_dir / "metric.csv", "a") as metric_file:
            metric_writer = csv.DictWriter(
                metric_file,
                # Ref: `self._train_one_epoch` method.
                ["epoch", "train_loss", "train_truth_loss", "val_loss", "val_truth_loss"],
            )
            metric_writer.writeheader()
            i = 0
            while True:
                metric_record = self._train_one_epoch(
                    candidate_model, optimizer, train_data_list, val_data_list, training_config.batch_size
                )
                # Log the metrics.
                metric_row = {"epoch": i}
                for name, value in metric_record.items():
                    metric_row[name] = value
                metric_writer.writerow(metric_row)
                metric_file.flush()
                # Check stopping condition.
                early_stopping = early_stopping.update(metric_record["val_loss"])
                if early_stopping.should_stop or i >= training_config.epochs_num - 1:
                    break
                # Next epoch.
                i += 1
        # Evaluate the candidate model against the current best model.
        self.best_model.eval()  # Switch to eval mode
        candidate_model.eval()  # Switch to eval mode
        is_best_model_updated = evaluate(
            (self.best_model, model_config),
            (candidate_model, model_config),
            evaluation_config,
            output_dir=output_dir / EVALUATION_BEST_CANDIDATE_DIR_NAME,
            workers_num=workers_num,
        )
        if is_best_model_updated:
            # The candidate model becomes the new best model.
            self.best_model = candidate_model
            self.best_model.set_name("best")
        return is_best_model_updated

    def get_best_model(self) -> MnkModel:
        """ """
        self.best_model.eval()  # Switch to eval mode
        return self.best_model

    @staticmethod
    def _train_one_epoch(
        model: MnkModel,
        optimizer: nnx.Optimizer,
        train_data_list: list[tuple[MnkInputData, list[float], float]],
        val_data_list: list[tuple[MnkInputData, list[float], float]],
        batch_size: int,
    ) -> dict[str, float]:
        """
        Train for one epoch.
        """

        def _loss_fn(
            model: MnkModel, data_batch: tuple[Array, Array, Array]
        ) -> tuple[float, tuple[float, Array, Array]]:
            # l = (z-v)^2 - π*log(p)
            boards, improved_probs, winners = data_batch
            prior_prob_logits, values = model(boards)
            value_loss = optax.l2_loss(values, winners).mean()  # MSE
            probs_loss = optax.softmax_cross_entropy(prior_prob_logits, improved_probs).mean()
            loss = value_loss + probs_loss
            # The ideal "ground truth" loss is the sum of the "floor" (minimum possible) MSE loss, which is zero,
            # and the "floor" cross-entropy loss, which is equal to the entropy of the target distribution itself.
            # NOTE:
            # In practice, this "ground truth" loss is rarely reachable because identical states
            # often have different prior probabilities and values due to noise and varying final results.
            truth_loss = jnp.sum(entr(improved_probs), axis=-1).mean()
            return loss, (truth_loss, prior_prob_logits, winners)

        metric = nnx.MultiMetric(loss=nnx.metrics.Average("loss"), truth_loss=nnx.metrics.Average("truth_loss"))
        metric_record: dict[str, float] = {}
        grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
        # Training
        train_data_list = random.sample(train_data_list, len(train_data_list))
        for data_batch in [train_data_list[i : i + batch_size] for i in range(0, len(train_data_list), batch_size)]:
            boards = jnp.array([i.board_data for (i, _, _) in data_batch])
            boards = boards.reshape((*boards.shape, MnkModel.INPUT_CHANNEL))
            improved_probs = jnp.array([p for (_, p, _) in data_batch])
            winners = jnp.array([w for (_, _, w) in data_batch])
            (loss, (truth_loss, *_)), grads = grad_fn(model, (boards, improved_probs, winners))
            metric.update(loss=loss, truth_loss=truth_loss)
            optimizer.update(model, grads)
        for name, value in metric.compute().items():
            metric_record[f"train_{name}"] = value
        metric.reset()
        # Evaluating
        for data_batch in [val_data_list[i : i + batch_size] for i in range(0, len(val_data_list), batch_size)]:
            boards = jnp.array([i.board_data for (i, _, _) in data_batch])
            boards = boards.reshape((*boards.shape, MnkModel.INPUT_CHANNEL))
            improved_probs = jnp.array([p for (_, p, _) in data_batch])
            winners = jnp.array([w for (_, _, w) in data_batch])
            loss, (truth_loss, *_) = _loss_fn(model, (boards, improved_probs, winners))
            metric.update(loss=loss, truth_loss=truth_loss)
        for name, value in metric.compute().items():
            metric_record[f"val_{name}"] = value
        metric.reset()
        return metric_record


def log_competition(competition_file_path: os.PathLike, play_record: PlayRecord):
    """ """
    last_state, moves, result = play_record
    model1_name, *_ = moves[0]
    model2_name, *_ = moves[1]
    with open(competition_file_path, "w") as competition_file:
        for j, (model_name, state, prior_probs, search_probs, value) in enumerate(moves):
            is_in_red_turn = j % 2 == 0  # Red always moves first. Ref: `MnkState.get_current_player_color` method.
            competition_file.write(f"{state}\n")
            competition_file.write(
                f"Model: {model_name} " + f"({(StoneColor.RED if is_in_red_turn else StoneColor.GREEN).get_mark()})\n"
            )
            competition_file.write(f"Prior probabilities (%):\n{format_board(state, prior_probs)}\n")
            competition_file.write(f"Search probabilities (%):\n{format_board(state, search_probs)}\n")
            competition_file.write(f"Value: {value:.2f}\n")
            competition_file.write("\n")
        competition_file.write(f"{last_state}\n")
        competition_file.write("\n")
        competition_file.write("Draw" if result == 0.0 else f"Winner: {model1_name if result < 0 else model2_name}")
        competition_file.write("\n")


def format_board(state: MnkState, flattened_probs: list[float]) -> str:
    """ """
    n = len(state.board)
    m = len(state.board[0])
    board_str = ""
    probs = [flattened_probs[i : i + m] for i in range(0, len(flattened_probs), m)]
    for y, (board_row, prob_row) in enumerate(
        zip(reversed(state.board), reversed(probs))
    ):  # Print rows from top to bottom
        for x, (stone, prob) in enumerate(zip(board_row, prob_row)):
            is_last_stone = False
            if state.last_action is not None:
                is_last_stone = x == state.last_action.x and y == len(state.board) - 1 - state.last_action.y
            board_str += (
                f"{int(prob * 100):3}"  # Percentage
                if stone is None
                else f"  {stone.color.get_mark(is_last_stone=is_last_stone)}"
            )
            if x < m - 1:
                board_str += " "
        if y < n - 1:
            board_str += "\n"
    return board_str


def augment_data(data: tuple[MnkState, list[float], float]) -> dict[str, tuple[MnkState, list[float], float]]:
    """
    Leverage the symmetry of the board to augment data by applying dihedral reflections or rotations.
    """
    state, probs, reward = data
    n = len(state.board)
    m = len(state.board[0])

    def _flip_state_vertically(state: MnkState) -> MnkState:
        """ """
        new_board: list[list[Optional[Stone]]] = []
        for row in reversed(state.board):
            new_row = [Stone(s.color) if s is not None else None for s in row]
            new_board.append(new_row)
        new_last_action = None
        if state.last_action is not None:
            new_last_action = MnkAction(state.last_action.x, n - 1 - state.last_action.y)
        new_state = MnkState(new_board, state.stone_count, new_last_action)
        return new_state

    def _flip_state_horizontally(state: MnkState) -> list[list[float]]:
        """ """
        new_board: list[list[Optional[Stone]]] = []
        for row in state.board:
            new_row = [Stone(s.color) if s is not None else None for s in reversed(row)]
            new_board.append(new_row)
        new_last_action = None
        if state.last_action is not None:
            new_last_action = MnkAction(m - 1 - state.last_action.x, state.last_action.y)
        new_state = MnkState(new_board, state.stone_count, new_last_action)
        return new_state

    def _flip_state_centrally(state: MnkState) -> list[list[float]]:
        """ """
        new_board: list[list[Optional[Stone]]] = []
        for row in reversed(state.board):
            new_row = [Stone(s.color) if s is not None else None for s in reversed(row)]
            new_board.append(new_row)
        new_last_action = None
        if state.last_action is not None:
            new_last_action = MnkAction(m - 1 - state.last_action.x, n - 1 - state.last_action.y)
        new_state = MnkState(new_board, state.stone_count, new_last_action)
        return new_state

    # `_flip_probs_*` functions use the same layout as `MnkGame.list_all_actions` method to ensure the same action order.
    # Please refer to `..core.model.Model` class for details.

    def _flip_probs_vertically(probs: list[float]) -> list[float]:
        """ """
        new_probs: list[float] = []
        for y in reversed(range(n)):
            new_row = [probs[y * m + x] for x in range(m)]
            new_probs += new_row
        return new_probs

    def _flip_probs_horizontally(probs: list[float]) -> list[float]:
        """ """
        new_probs: list[float] = []
        for y in range(n):
            new_row = [probs[y * m + x] for x in reversed(range(m))]
            new_probs += new_row
        return new_probs

    def _flip_probs_centrally(probs: list[float]) -> list[float]:
        """ """
        new_probs: list[float] = []
        for y in reversed(range(n)):
            new_row = [probs[y * m + x] for x in reversed(range(m))]
            new_probs += new_row
        return new_probs

    augmented_data_list: dict[str, tuple[MnkState, list[float], float]] = {
        "vertical": (_flip_state_vertically(state), _flip_probs_vertically(probs), reward),
        "horizontal": (_flip_state_horizontally(state), _flip_probs_horizontally(probs), reward),
        "central": (_flip_state_centrally(state), _flip_probs_centrally(probs), reward),
        # TODO: Add rotations.
    }
    return augmented_data_list
