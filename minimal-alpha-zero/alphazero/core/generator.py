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


def generate_data(
    self_plays_num: int,
    game: Game,
    model_spec: tuple[Model, ModelConfig],
    play_config: PlayConfig,
    noise_session: Optional[NoiseSession] = None,
    workers_num: Optional[int] = None,
) -> Iterable[list[tuple[State, list[float], float]]]:
    """
    Continuously play games over many self-plays to generate data.
    """

    def _self_play(noise_session: Optional[NoiseSession], seed: float) -> list[tuple[State, list[float], float]]:
        """ """
        random.seed(seed)
        # Use the same model for both players during self-play.
        _, moves, reward = play(game, [model_spec], play_config, noise_session=noise_session)
        if len(moves) == 0 or reward is None:
            # Just in case.
            return []
        data_list: list[tuple[State, list[float], float]] = []
        for j, (state, _, search_probs, _) in enumerate(moves):
            # The sign of reward for each move (positive or negative) alternates based on current player's perspective,
            # since two players take turns: Player A moves first, Player B moves second, then Player A again, and so on.
            # It also depends on which player makes the last move and whether that player wins or loses.
            # Therefore, the final reward is assigned to the last move, with its sign flipping at each move as we traverse backward.
            # The same logic is applied to values in `_backup` function.
            # NOTE: In case of a draw, all rewards should be 0.
            is_current_player_last_mover = (len(moves) - 1 - j) % 2 == 0
            cur_reward = reward * (1 if is_current_player_last_mover else -1)
            data_list.append((state, search_probs, cur_reward))
        return data_list

    # Divide into smaller segments to gradually handle the generated data.
    SEGMENT_SIZE = workers_num or os.cpu_count() or 1
    self_play_indices = list(range(self_plays_num))
    for i in range(0, self_plays_num, SEGMENT_SIZE):
        inputs: list[tuple[Optional[NoiseSession], float]] = []
        for _ in self_play_indices[i : i + SEGMENT_SIZE]:
            # Use a different session and random seed for each job.
            inputs.append((noise_session.split() if noise_session is not None else None, random.random()))
        aggregated_data_list: list[list[tuple[State, list[float], float]]]
        aggregated_data_list = Parallel(n_jobs=SEGMENT_SIZE)(delayed(_self_play)(*i) for i in inputs)
        for data_list in aggregated_data_list:
            yield data_list


def play(
    game: Game,
    model_specs: list[tuple[Model, ModelConfig]],
    config: PlayConfig,
    noise_session: Optional[NoiseSession] = None,
) -> tuple[State, list[tuple[State, list[float], list[float], float]], float]:
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
    node = Node(game.begin())
    moves: list[tuple[State, list[float], list[float], float]] = []
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
        model, model_config = model_specs[i % len(model_specs)]
        if model_config.mcts_simulations_num is not None:
            _execute_mcts(
                state_caches[i % len(model_specs)],
                node,
                game,
                model,
                model_config.mcts_simulations_num,
                config.c_puct,
                noise_session,
            )
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
        moves.append((node.state, all_prior_probs, all_search_probs, node.value))
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
