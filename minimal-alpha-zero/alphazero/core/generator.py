"""
Use the best model to generate data for further training.
"""

import logging
import math
import os
import random
from typing import Callable, Iterable, Optional

import jax
from jax import numpy as jnp
from joblib import Parallel, delayed

from .game import Action, State, Game
from .model import ModelConfig, Model


logger = logging.getLogger(__name__)

PlayRecord = tuple[State, list[tuple[str, State, list[float], list[float], float]], float]


class Node:
    """ """

    state: State
    children: dict[Action, "Edge"]  # Store only legal actions
    value: Optional[float]

    def __init__(self, state: State):
        self.state = state
        self.children = {}
        self.value = None

    @property
    def is_leaf(self) -> bool:
        # We do not use `children` to determine whether a node is a leaf
        # because terminal states have empty children, similar to real leaf nodes.
        return self.value is None


class Edge:
    """
    Each edge connects a parent node to one of its child nodes via an action.
    """

    node: Node  # The child node, not the parent
    prior_prob: float  # P(s,a)
    action_value: float  # Q(s,a)
    visit_count: int  # N(s,a)

    def __init__(self, node: Node, prior_prob: float):
        self.prior_prob = prior_prob
        self.node = node
        self.action_value = 0.0
        self.visit_count = 0


class PlayConfig:
    """ """

    c_puct: float
    calc_temperature: Callable[[int], float]  # τ

    def __init__(
        self,
        *,
        c_puct: float = 1.0,
        calc_temperature: Callable[[int], float] = lambda _: 1.0,
    ):
        self.c_puct = c_puct
        self.calc_temperature = calc_temperature


class EvaluationConfig:
    """ """

    competitions_num: int
    competition_margin: float  # Should be positive and less than 1
    game: Game
    play_config: PlayConfig
    log_competition: Optional[Callable[[os.PathLike, PlayRecord], None]]

    def __init__(
        self,
        *,
        competitions_num: int,
        competition_margin: float,
        game: Game,
        play_config: PlayConfig,
        log_competition: Optional[Callable[[os.PathLike, PlayRecord], None]] = None,
    ):
        self.competitions_num = competitions_num
        self.competition_margin = competition_margin
        self.game = game
        self.play_config = play_config
        self.log_competition = log_competition


class NoiseSession:
    """ """

    key: jax.Array
    # There is a heuristic that uses α=10/n, where n is the maximum number of legal actions.
    # Doc: https://jonathan-laurent.github.io/AlphaZero.jl/stable/reference/params/#MCTS
    dirichlet_alpha: float  # α
    fraction: float  # ε

    def __init__(
        self,
        *,
        key: Optional[jax.Array] = None,
        dirichlet_alpha: float = 0.0,
        fraction: float = 0.0,
    ):
        self.key = key
        self.dirichlet_alpha = dirichlet_alpha
        self.fraction = fraction

    def split(self) -> "NoiseSession":
        key, subkey = jax.random.split(self.key)
        self.key = key
        return NoiseSession(key=subkey, dirichlet_alpha=self.dirichlet_alpha, fraction=self.fraction)


class GenerationConfig:
    """ """

    self_plays_num: int
    game: Game
    play_config: PlayConfig
    noise_session: Optional[NoiseSession] = None

    def __init__(
        self,
        *,
        self_plays_num: int,
        game: Game,
        play_config: PlayConfig,
        noise_session: Optional[NoiseSession] = None,
    ):
        self.self_plays_num = self_plays_num
        self.game = game
        self.play_config = play_config
        self.noise_session = noise_session


def generate_data(
    model_spec: tuple[Model, ModelConfig],
    config: GenerationConfig,
    partner_model_choices: Optional[list[tuple[Model, ModelConfig, float]]] = None,
    aggregation_excluded_model_names: Optional[set[str]] = None,
    workers_num: Optional[int] = None,
) -> Iterable[tuple[list[str], list[tuple[State, list[float], float]]]]:
    """
    Continuously play games over many self-plays to generate data.
    """

    def _self_play(
        noise_session: Optional[NoiseSession], seed: float
    ) -> tuple[list[str], list[tuple[State, list[float], float]]]:
        """ """
        random.seed(seed)
        model_specs = [model_spec]  # Use the same model for both players by default.
        if partner_model_choices is not None and len(partner_model_choices) > 0:
            partner_model_specs = [(m, c) for m, c, _ in partner_model_choices]
            partner_model_weights = [w for _, _, w in partner_model_choices]
            # Using not only the current model but also various partner models is not pure "self-play",
            # but this is necessary to generate more diverse data.
            partner_model_spec = random.choices(partner_model_specs, weights=partner_model_weights)[0]
            model, _ = model_spec
            partner_model, _ = partner_model_spec
            if model.name != partner_model.name:
                # NOTE: When playing a game with one player, a single MCTS tree is used for both sides,
                # but when playing with two or more players, separate trees are used for each player.
                model_specs = [model_spec, partner_model_spec]
                random.shuffle(model_specs)
        _, moves, reward = _play(config.game, model_specs, config.play_config, noise_session=noise_session)
        if len(moves) == 0 or reward is None:
            # Just in case.
            return []
        data_list: list[tuple[State, list[float], float]] = []
        for j, (model_name, state, _, search_probs, _) in enumerate(moves):
            if aggregation_excluded_model_names is not None and model_name in aggregation_excluded_model_names:
                continue
            # The sign of reward for each move (positive or negative) alternates based on current player's perspective,
            # since two players take turns: Player A moves first, Player B moves second, then Player A again, and so on.
            # It also depends on which player makes the last move and whether that player wins or loses.
            # Therefore, the final reward is assigned to the last move, with its sign flipping at each move as we traverse backward.
            # The same logic is applied to values in `_backup` function.
            # NOTE: In case of a draw, all rewards should be 0.
            is_current_player_last_mover = (len(moves) - 1 - j) % 2 == 0
            cur_reward = reward * (1 if is_current_player_last_mover else -1)
            data_list.append((state, search_probs, cur_reward))
        model_names = [m.name for m, _ in model_specs]
        return model_names, data_list

    # Divide into smaller segments to gradually handle the generated data.
    SEGMENT_SIZE = workers_num or os.cpu_count() or 1
    self_play_indices = list(range(config.self_plays_num))
    for i in range(0, config.self_plays_num, SEGMENT_SIZE):
        inputs: list[tuple[Optional[NoiseSession], float]] = []
        for _ in self_play_indices[i : i + SEGMENT_SIZE]:
            # Use a different session and random seed for each job.
            inputs.append((config.noise_session.split() if config.noise_session is not None else None, random.random()))
        aggregated_data_list: list[tuple[list[str], list[tuple[State, list[float], float]]]]
        aggregated_data_list = Parallel(n_jobs=SEGMENT_SIZE)(delayed(_self_play)(*i) for i in inputs)
        for model_names, data_list in aggregated_data_list:
            yield model_names, data_list


def evaluate(
    model_spec1: tuple[Model, ModelConfig],
    model_spec2: tuple[Model, ModelConfig],
    config: EvaluationConfig,
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

    def _compete_one_game(model_spec1: tuple[Model, ModelConfig], model_spec2: tuple[Model, ModelConfig]) -> PlayRecord:
        """
        Have two models play a game and choose the better one based on the reward.
        Return -1 if `model_spec1` wins, 1 if `model_spec2` wins, `0` for a draw.
        NOTE: `model_spec1` always moves first.
        """
        # Do not add noise during evaluation.
        last_state, moves, reward = _play(game, [model_spec1, model_spec2], play_config)
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
        moves: list[tuple[str, State, list[float], list[float], float]]
        result: float
        # Two models take turns having the first move in each competition.
        if is_model1_first_mover:
            last_state, moves, cur_result = _compete_one_game(model_spec1, model_spec2)
            result = cur_result
        else:
            last_state, moves, cur_result = _compete_one_game(model_spec2, model_spec1)
            result = -cur_result
        # Log the moves.
        if output_dir is not None and config.log_competition is not None:
            competition_file_path = output_dir / f"competition-{competition_index:0{len(str(competitions_num))}d}.log"
            config.log_competition(competition_file_path, (last_state, moves, cur_result))
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
        with open(output_dir / "@result.log", "w") as result_file:
            result_file.write(f"model1={model1.name}, model2={model2.name}, result={result}\n")
    logger.info(f"model1={model1.name}, model2={model2.name}, result={result}")
    if abs(result) >= config.competition_margin:
        return result > 0
    else:
        return None


def _play(
    game: Game,
    model_specs: list[tuple[Model, ModelConfig]],
    config: PlayConfig,
    noise_session: Optional[NoiseSession] = None,
) -> PlayRecord:
    """
    Play a game with multiple players.
    Each player (model) takes turns in order based on its index in the list.
    Each move is selected after executing a specific number of MCTS simulations.
    """
    all_actions = game.list_all_actions()
    # Since we need to switch to a different model and its corresponding tree when changing turns,
    # we must check whether the next state exists in the tree.
    # Theoretically, we could do this by storing only a root node for each model and performing a recursive search,
    # but since this approach is inefficient, we store the list of existing states in `state_caches` variable for faster checking.
    state_caches: list[dict[State, Node]] = [{} for _ in range(len(model_specs))]
    mcts_simulations_nums: list[Optional[int]] = [
        c.calc_mcts_simulations_num() if c.calc_mcts_simulations_num is not None else None for _, c in model_specs
    ]
    node = Node(game.begin())
    moves: list[tuple[str, State, list[float], list[float], float]] = []
    reward = 0.0
    i = 0
    while True:
        # The final reward is computed at this state, but it is assigned to the last move that produced this state.
        reward = game.receive_reward_if_terminal(node.state)
        is_terminated = reward is not None
        if is_terminated:
            break
        # Search for the probabilities of legal actions for the next move.
        search_probs: dict[Action, float]
        model, _ = model_specs[i % len(model_specs)]
        mcts_simulations_num = mcts_simulations_nums[i % len(model_specs)]
        if mcts_simulations_num is not None:
            state_cache = state_caches[i % len(model_specs)]
            _execute_mcts(state_cache, node, game, model, mcts_simulations_num, config.c_puct, noise_session)
            # Select a move according to the search probabilities π computed by MCTS.
            # π(a|s) = N(s,a)^(1/τ) / (∑b N(s,b)^(1/τ))
            scaled_visit_counts = {
                a: math.pow(e.visit_count, 1 / config.calc_temperature(i)) for a, e in node.children.items()
            }
            scaled_visit_counts_sum = sum(scaled_visit_counts.values())
            search_probs = {a: c / scaled_visit_counts_sum for a, c in scaled_visit_counts.items()}
        else:
            if node.is_leaf:
                # If MCTS is not executed, we directly use the prior probabilities instead of the visit counts.
                _expand(node, game, model)
            # Simply use the same formula as in the case where MCTS execution is enabled, although there may be a better approach.
            scaled_prior_probs = {
                a: math.pow(e.prior_prob, 1 / config.calc_temperature(i)) for a, e in node.children.items()
            }
            scaled_prior_probs_sum = sum(scaled_prior_probs.values())
            search_probs = {a: p / scaled_prior_probs_sum for a, p in scaled_prior_probs.items()}
        all_prior_probs = [node.children[a].prior_prob if a in node.children else 0.0 for a in all_actions]
        all_search_probs = [search_probs[a] if a in search_probs else 0.0 for a in all_actions]
        moves.append((model.name, node.state, all_prior_probs, all_search_probs, node.value))
        action = random.choices(all_actions, weights=all_search_probs)[0]  # Next action
        # Move to the next model and next node.
        i += 1
        state = node.children[action].node.state
        if state in state_caches[i % len(model_specs)]:
            node = state_caches[i % len(model_specs)][state]
        else:
            # Because trees are independent across different models, when we switch to the tree of the next model,
            # there is a chance that the new state has not been discovered by that model;
            # therefore, the corresponding node does not exist in the tree.
            # In that case, we create a new root node and reset the entire tree for the next model.
            node = Node(state)
            state_caches[i % len(model_specs)] = {}
    return node.state, moves, reward


def _execute_mcts(
    state_cache: dict[State, Node],
    node: Node,
    game: Game,
    model: Model,
    simulations_num: int,
    c_puct: float,
    noise_session: Optional[NoiseSession],
):
    """
    Execute multiple MCTS simulations.
    """

    def _execute_one_simulation(root: Node, game: Game, model: Model, c_puct: float) -> dict[State, Node]:
        """
        Return a dictionary mapping each expanded state to its corresponding node.
        NOTE: By "root", we mean the node where MCTS simulation begins, not an initial game state.
        """
        node = root
        edges: list[Edge] = []
        while True:
            # TODO: Store the reward to avoid calculating it multiple times.
            is_finished = node.is_leaf or game.receive_reward_if_terminal(node.state) is not None
            if is_finished:
                break
            # Select an action according to edge statistics.
            action = _select(node, c_puct)
            edge = node.children[action]
            edges.append(edge)
            # Move to the next node.
            node = edge.node
        expanded_states: dict[State, Node] = {}
        # We evaluate a leaf node only once during a game play.
        # If the node is not a leaf, it should be a terminal node, in that case, we do not expand it again.
        if node.is_leaf:
            # Expand and evaluate a leaf node.
            _expand(node, game, model)
            for edge in node.children.values():
                expanded_states[edge.node.state] = edge.node
        # Update edge statistics in a backward pass through each move.
        _backup(edges, node.value)
        return expanded_states

    i = 0
    # Execute MCTS simulations.
    if node.is_leaf:
        # Run an initial simulation to expand a leaf node.
        for expanded_state, expanded_node in _execute_one_simulation(node, game, model, c_puct).items():
            state_cache[expanded_state] = expanded_node
    # Add noise to the root node.
    if noise_session is not None:
        _add_noise(node, noise_session)
    while i < simulations_num:
        for expanded_state, expanded_node in _execute_one_simulation(node, game, model, c_puct).items():
            state_cache[expanded_state] = expanded_node
        # Next simulation.
        i += 1


def _select(node: Node, c_puct: float) -> Action:
    """
    Use a variant of PUCT algorithm to select an action during an MCTS simulation.
    """

    def _calc_puct_score(
        edge: Edge,
        visit_counts_sum: float,  # ∑b N(s,b)
    ) -> float:
        """ """
        # Q(s,a)
        exploitation_term = edge.action_value
        # U(s,a) = c_puct * P(s,a) * sqrt(∑b N(s,b)) / (1+N(s,a))
        exploration_term = c_puct * edge.prior_prob * math.sqrt(visit_counts_sum) / (1 + edge.visit_count)
        # Q(s,a) + U(s,a)
        return exploitation_term + exploration_term

    # NOTE: Assume the node is not a leaf.
    visit_counts_sum = sum([e.visit_count for e in node.children.values()])
    # Select action with the maximum PUCT score.
    action = max(node.children, key=lambda a: _calc_puct_score(node.children[a], visit_counts_sum))
    return action


def _expand(node: Node, game: Game, model: Model):
    """
    Evaluate a leaf node by assigning a value to its state and prior probabilities to each of its edges.
    """
    prior_probs: dict[Action, float]
    value = 0.0
    legal_actions: list[Action]
    reward = game.receive_reward_if_terminal(node.state)
    if reward is not None:
        # The minus sign in `-reward` indicates that the current state's value has the opposite sign of the reward,
        # since the current state belongs to the current player, but the reward comes from the opponent's last move.
        # For example, if a state is terminal and has a positive reward, it means the opponent won with the last move,
        # so the current player lost, and the current state's value is negative.
        # Ref: `.game.Game.receive_reward_if_terminal` method.
        prior_probs, value = {}, -reward
        # There are no legal actions available for a terminal state.
        legal_actions = []
    else:
        prior_probs, value = model.predict_single(node.state.make_input_data())
        # The prior may contain non-zero probabilities for illegal actions. We need to eliminate those and keep only the legal ones.
        legal_actions = game.list_legal_actions(node.state)
    legal_prior_probs = {a: p for a, p in prior_probs.items() if a in legal_actions}
    # Create new edges that include prior probabilities only for legal actions.
    for action in legal_actions:
        state = game.simulate(node.state, action)
        node.children[action] = Edge(Node(state), legal_prior_probs[action])
    node.value = value


def _backup(edges: list[Edge], value: float):
    """
    Update edge statistics.
    """
    for i, edge in enumerate(reversed(edges)):
        # The sign of value flips at each edge as we traverse backward.
        # See `generate_data` function for an explanation, since it applies the same logic.
        is_current_player_last_mover = i % 2 == 0
        # The last mover (i.e., the action of the last edge) receives `-value`
        # because `value` belongs to a state that comes from the opponent's perspective.
        cur_value = value * (-1 if is_current_player_last_mover else 1)
        edge.action_value = (edge.action_value * edge.visit_count + cur_value) / (edge.visit_count + 1)
        edge.visit_count += 1


def _add_noise(node: Node, session: NoiseSession):
    """
    Add Dirichlet noise to the prior probabilities.
    """
    key = session.split().key  # Create a new session with a new key
    noises = jax.random.dirichlet(key, jnp.full((len(node.children),), session.dirichlet_alpha))
    for edge, noise in zip(node.children.values(), noises):
        edge.prior_prob = edge.prior_prob * (1 - session.fraction) + session.fraction * noise
