"""
Please read https://en.wikipedia.org/wiki/M,n,k-game, I'm too lazy to write the description.
"""

import csv
import logging
import os
import random
from abc import ABC
from enum import Enum
from typing import Optional

import optax
from flax import nnx
from flax.training.early_stopping import EarlyStopping
from jax import Array, nn, numpy as jnp
from joblib import Parallel, delayed

from ..core.game import Action, InputData, State, Game, ReplayBuffer
from ..core.model import ModelConfig, Model
from ..core.generator import PlayConfig, play


EVALUATION_DUMMY_BEST_DIR_NAME = "evaluation-dummy-best"
EVALUATION_BEST_CANDIDATE_DIR_NAME = "evaluation-best-candidate"
logger = logging.getLogger(__name__)


class StoneColor(Enum):
    """ """

    RED = (-1, "x")
    GREEN = (1, "o")

    _mark: str

    def __new__(cls, value: int, mark: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._mark = mark
        return obj

    def get_mark(self, is_last_stone: bool = False) -> str:
        return self._mark if not is_last_stone else self._mark.upper()


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
        self.board = board
        self.stone_count = stone_count
        self.last_action = last_action

    def make_input_data(self) -> MnkInputData:
        """ """
        board_data = [[float(s.color.value) if s is not None else 0.0 for s in r] for r in self.board]
        return MnkInputData(board_data)

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
    initial_stones_num: int

    # With some choices of m, n, k, it is probably too easy for the first player to win.
    # Hence, we give the winner a larger reward if the game ends quickly and a smaller reward if it lasts longer,
    # hopefully encouraging the second player to prolong the game as much as possible to reduce the reward their opponent receives.
    should_prefer_fast_win: bool

    def __init__(self, m: int, n: int, k: int, initial_stones_num: int = 0, should_prefer_fast_win: bool = False):
        self.m = m
        self.n = n
        self.k = k
        self.initial_stones_num = initial_stones_num
        self.should_prefer_fast_win = should_prefer_fast_win

    def begin(self) -> MnkState:
        """ """
        state = MnkState([[None for _ in range(self.m)] for _ in range(self.n)], 0, None)
        if self.initial_stones_num == 0:
            return state
        positions = [(x, y) for y in range(self.n) for x in range(self.m)]
        initial_stones_num = self.initial_stones_num
        if initial_stones_num % 2 != 0:
            initial_stones_num += 1  # Should be an even number
        initial_stones_num = min(self.m * self.n, initial_stones_num)
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
        red_count = 0
        green_count = 0
        for cur_row in state.board:
            new_row: list[Optional[Stone]] = []
            for stone in cur_row:
                if stone is not None:
                    # Count the number of stones for each color.
                    if stone.color == StoneColor.RED:
                        red_count += 1
                    else:
                        green_count += 1
                    new_row.append(Stone(stone.color))
                else:
                    new_row.append(None)
            new_board.append(new_row)
        # Determine whether it is Red's turn.
        is_in_red_turn = red_count <= green_count
        color = Stone(StoneColor.RED if is_in_red_turn else StoneColor.GREEN)
        stone_count = state.stone_count
        if new_board[action.y][action.x] is None:
            new_board[action.y][action.x] = color
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


class NamedModel(Model, ABC):
    """ """

    name: str  # Mainly used for debugging

    def set_name(self, name: str):
        self.name = name


class MnkModel(nnx.Module, NamedModel):
    """ """

    INPUT_CHANNEL = 1

    def __init__(self, m: int, n: int, rngs: nnx.Rngs):
        # Convolutional
        self.conv = nnx.Conv(self.INPUT_CHANNEL, 16, kernel_size=3, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(16, rngs=rngs)
        self.dropout0 = nnx.Dropout(rate=0.025, rngs=rngs)
        # Fully connected
        self.linear = nnx.Linear(16 * m * n, 128, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.025, rngs=rngs)
        # Heads
        self.prior_probs_head = nnx.Linear(128, m * n, rngs=rngs)
        self.value_head = nnx.Linear(128, 1, rngs=rngs)

    def __call__(self, x: Array) -> tuple[Array, Array]:
        x = self.dropout0(nnx.relu(self.batch_norm(self.conv(x))))
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.dropout1(nnx.relu(self.linear(x)))
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


class DummyModel(NamedModel):
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

    learning_rate: float
    epochs_num: int
    batch_size: int
    stopping_patience: int

    def __init__(
        self,
        *,
        learning_rate: float,
        epochs_num: int,
        batch_size: int,
        stopping_patience: int,
    ):
        self.learning_rate = learning_rate
        self.epochs_num = epochs_num
        self.batch_size = batch_size
        self.stopping_patience = stopping_patience


class MnkEvaluationConfig:
    """ """

    competitions_num: int
    competition_margin: float  # Should be positive and less than 1
    game: MnkGame
    play_config: PlayConfig

    def __init__(
        self,
        *,
        competitions_num: int,
        competition_margin: float,
        game: MnkGame,
        play_config: PlayConfig,
    ):
        self.competitions_num = competitions_num
        self.competition_margin = competition_margin
        self.game = game
        self.play_config = play_config


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
        evaluation_config: MnkEvaluationConfig,
        model_config: ModelConfig,
        output_dir: os.PathLike,
        workers_num: Optional[int] = None,
    ) -> bool:
        """ """
        # Train a new candidate model.
        candidate_model = nnx.clone(self.best_model)
        candidate_model.set_name("candidate")
        candidate_model.train()  # Switch to train mode
        optimizer = nnx.Optimizer(candidate_model, optax.adamw(training_config.learning_rate), wrt=nnx.Param)
        early_stopping = EarlyStopping(patience=training_config.stopping_patience)
        with open(output_dir / "metric.csv", "a") as metric_file:
            metric_writer = csv.DictWriter(metric_file, ["epoch", "train_loss", "val_loss"])
            metric_writer.writeheader()
            i = 0
            while True:
                data_list = [(s.make_input_data(), p, w) for b in replay_buffer.buffer_queue for (s, p, w) in b]
                metric_record = self._train_one_epoch(candidate_model, optimizer, data_list, training_config.batch_size)
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
        data_list: list[tuple[MnkInputData, list[float], float]],
        batch_size: int,
        split_ratio: float = 0.9,
    ) -> dict[str, float]:
        """
        Train for one epoch.
        """

        def _loss_fn(model: MnkModel, data_batch: tuple[Array, Array, Array]) -> tuple[float, tuple[Array, Array]]:
            # l = (z-v)^2 - π*log(p)
            boards, improved_probs, winners = data_batch
            prior_prob_logits, values = model(boards)
            value_loss = optax.l2_loss(values, winners).mean()  # MSE
            probs_loss = optax.softmax_cross_entropy(prior_prob_logits, improved_probs).mean()
            loss = value_loss + probs_loss
            return loss, (prior_prob_logits, winners)

        metric = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
        metric_record: dict[str, float] = {}
        data_list = random.sample(data_list, len(data_list))  # TODO: Choose a deterministic approach
        train_len = int(len(data_list) * split_ratio)
        train_data_list = data_list[:train_len]
        val_data_list = data_list[train_len:]
        grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
        # Training
        for data_batch in [train_data_list[i : i + batch_size] for i in range(0, len(train_data_list), batch_size)]:
            boards = jnp.array([i.board_data for (i, _, _) in data_batch])
            boards = boards.reshape((*boards.shape, MnkModel.INPUT_CHANNEL))
            improved_probs = jnp.array([p for (_, p, _) in data_batch])
            winners = jnp.array([w for (_, _, w) in data_batch])
            (loss, _), grads = grad_fn(model, (boards, improved_probs, winners))
            metric.update(loss=loss)
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
            loss, _ = _loss_fn(model, (boards, improved_probs, winners))
            metric.update(loss=loss)
        for name, value in metric.compute().items():
            metric_record[f"val_{name}"] = value
        metric.reset()
        return metric_record


def evaluate(
    model_spec1: tuple[NamedModel, ModelConfig],
    model_spec2: tuple[NamedModel, ModelConfig],
    config: MnkEvaluationConfig,
    output_dir: Optional[os.PathLike] = None,
    workers_num: Optional[int] = None,
) -> Optional[bool]:
    """
    Return `False` if `model_spec1` is better by the given margin, `True` if `model_spec2` is better by the given margin,
    and `None` otherwise.
    """
    model1, _ = model_spec1
    model2, _ = model_spec2
    competitions_num, game, play_config = (config.competitions_num, config.game, config.play_config)

    def _compete_one_game(
        model_spec1: tuple[Model, ModelConfig], model_spec2: tuple[Model, ModelConfig]
    ) -> tuple[State, list[tuple[State, list[float], list[float], float]], float]:
        """
        Have two models play a game and choose the better one based on the reward.
        Return -1 if `model_spec1` wins, 1 if `model_spec2` wins, `0` for a draw.
        NOTE: `model_spec1` always moves first.
        """
        # Do not add noise during evaluation.
        last_state, moves, reward = play(game, [model_spec1, model_spec2], play_config)
        is_model1_last_mover = len(moves) % 2 == 1
        if reward == 0.0:
            return last_state, moves, 0.0
        result = -1.0 if is_model1_last_mover == (reward > 0) else 1.0
        return last_state, moves, result

    def _launch_one_competition(competition_index: int, seed: float) -> float:
        """ """
        random.seed(seed)
        is_model1_first_mover = competition_index % 2 == 0
        last_state: State
        moves: list[tuple[State, list[float], list[float], float]]
        result: float
        # Two models take turns having the first move in each competition.
        if is_model1_first_mover:
            last_state, moves, cur_result = _compete_one_game(model_spec1, model_spec2)
            result = cur_result
        else:
            last_state, moves, cur_result = _compete_one_game(model_spec2, model_spec1)
            result = -cur_result
        # Log the moves.
        if output_dir is not None:
            competition_file_path = output_dir / f"competition-{competition_index:0{len(str(competitions_num))}d}.txt"
            with open(competition_file_path, "w") as competition_file:
                for j, (state, prior_probs, search_probs, value) in enumerate(moves):
                    is_in_red_turn = j % 2 == 0  # Red always moves first. Ref: `MnkGame.simulate` method.
                    competition_file.write(f"{state}\n")
                    competition_file.write(
                        f"Model: {model1.name if is_model1_first_mover == is_in_red_turn else model2.name} "
                        + f"({(StoneColor.RED if is_in_red_turn else StoneColor.GREEN).get_mark()})\n"
                    )
                    competition_file.write(f"Prior probabilities:\n{format_board(prior_probs, game.m)}\n")
                    competition_file.write(f"Search probabilities:\n{format_board(search_probs, game.m)}\n")
                    competition_file.write(f"Value: {value:.2f}\n")
                    competition_file.write("\n")
                competition_file.write(f"{last_state}\n")
                competition_file.write("\n")
                competition_file.write(
                    "Draw"
                    if cur_result == 0.0
                    else f"Winner: {model1.name if is_model1_first_mover == (cur_result < 0) else model2.name}"
                )
                competition_file.write("\n")
        return result

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    workers_num = workers_num or os.cpu_count() or 1
    inputs = [(i, random.random()) for i in range(competitions_num)]  # Use a different random seed for each job
    results: list[float]
    results = Parallel(n_jobs=workers_num)(delayed(_launch_one_competition)(*i) for i in inputs)
    result = sum(results) / competitions_num
    # Log the result.
    if output_dir is not None:
        with open(output_dir / "@result.txt", "w") as result_file:
            result_file.write(f"model1={model1.name}, model2={model2.name}, result={result}\n")
    logger.info(f"model1={model1.name}, model2={model2.name}, result={result}")
    if abs(result) >= config.competition_margin:
        return result > 0
    else:
        return None


def format_board(flattened_board: list[float], m: int) -> str:
    """ """
    board_str = ""
    board = [flattened_board[i : i + m] for i in range(0, len(flattened_board), m)]
    for y, row in enumerate(reversed(board)):  # Print rows from top to bottom
        for value in row:
            board_str += f"{value:.2f} " if value != 0.0 else "____ "
        if y < len(board) - 1:
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
