"""
Use the best model to generate data for further training.
"""

import logging
import math
import random
from typing import Iterable, Optional

from .game import Action, State, Game
from .network import Model


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
    prior_probability: float  # P(s,a)
    action_value: float  # Q(s,a)
    visit_count: int  # N(s,a)

    def __init__(self, node: Node, prior_probability: float):
        self.prior_probability = prior_probability
        self.node = node
        self.action_value = 0.0
        self.visit_count = 0


def generate_data(
    game: Game,
    model: Model,
    self_plays_num: Optional[int] = None,
    self_play_select_simulations_num: int = 1,
) -> Iterable[tuple[State, list[float], float]]:
    """
    Continuously play games over many self-plays to generate data.
    """
    i = 0
    while self_plays_num is None or i < self_plays_num:
        # Use the same model for both players during self-play.
        _, moves, reward = play(game, [model], self_play_select_simulations_num)
        if len(moves) == 0 or reward is None:
            # Just in case.
            continue
        for j, (state, _, search_probabilities, _) in enumerate(moves):
            # The sign of reward for each move (positive or negative) alternates based on current player's perspective,
            # since two players take turns: Player A moves first, Player B moves second, then Player A again, and so on.
            # It also depends on which player makes the last move and whether that player wins or loses.
            # Therefore, the final reward is assigned to the last move, with its sign flipping at each move as we traverse backward.
            # The same logic is applied to values in `_backup` function.
            # NOTE: In case of a draw, all rewards should be 0.
            is_current_player_last_mover = (len(moves) - 1 - j) % 2 == 0
            cur_reward = reward * (1 if is_current_player_last_mover else -1)
            yield state, search_probabilities, cur_reward
        # Next self-play.
        i += 1


def play(
    game: Game,
    models: list[Model],
    select_simulations_num: int,
    select_temperature: float = 1.0,  # TODO: Tune this hyperparameter
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
    state_caches: list[dict[State, Node]] = [{} for _ in range(len(models))]
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
        # Select the next action.
        # TODO: Allow models to apply their own custom selection strategies instead of executing MCTS for them,
        # as they may not be trained using AlphaZero.
        action, legal_searches = _play_select(
            state_caches[i % len(models)],
            node,
            game,
            models[i % len(models)],
            select_simulations_num,
            select_temperature,
        )
        prior_probabilities = [node.children[a].prior_probability if a in node.children else 0.0 for a in all_actions]
        search_probabilities = [legal_searches[a] if a in legal_searches else 0.0 for a in all_actions]
        moves.append((node.state, prior_probabilities, search_probabilities, node.value))
        # Move to the next model and next node.
        i += 1
        state = node.children[action].node.state
        if state in state_caches[i % len(models)]:
            node = state_caches[i % len(models)][state]
        else:
            # Because trees are independent across different models, when we switch to the tree of the next model,
            # there is a chance that the new state has not been discovered by that model;
            # therefore, the corresponding node does not exist in the tree.
            # In that case, we create a new root node and reset the entire tree for the next model.
            node = Node(state)
            state_caches[i % len(models)] = {}
    return node.state, moves, reward


def _play_select(
    state_cache: dict[State, Node],
    node: Node,
    game: Game,
    model: Model,
    simulations_num: int,
    temperature: float,  # τ
) -> tuple[Action, dict[Action, float]]:
    """
    Select a legal action to move to a new state.
    """
    i = 0
    # Execute MCTS simulations.
    if node.is_leaf:
        # Run an initial simulation to expand a leaf node.
        simulations_num += 1
    while i < simulations_num:
        for expanded_state, expanded_node in _execute_an_mcts_simulation(node, game, model).items():
            state_cache[expanded_state] = expanded_node
        # Next simulation.
        i += 1
    # Select a move according to the search probabilities π computed by MCTS.
    # π(a|s) = N(s,a)^(1/τ) / (∑b N(s,b)^(1/τ))
    scaled_visit_counts = {a: math.pow(e.visit_count, 1 / temperature) for a, e in node.children.items()}
    scaled_visit_counts_sum = sum(scaled_visit_counts.values())
    legal_searches = {a: c / scaled_visit_counts_sum for a, c in scaled_visit_counts.items()}
    action = random.choices(list(legal_searches.keys()), weights=list(legal_searches.values()))[0]
    return action, legal_searches


def _execute_an_mcts_simulation(root: Node, game: Game, model: Model) -> dict[State, Node]:
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
        action = _mcts_select(node)
        edge = node.children[action]
        edges.append(edge)
        # Move to the next node.
        node = edge.node
    expanded_states: dict[State, Node] = {}
    # We evaluate a leaf node only once during a single game play.
    # If the node is not a leaf, it should be a terminal node, in that case, we do not expand it again.
    if node.is_leaf:
        # Expand and evaluate a leaf node.
        _expand(node, game, model)
        for edge in node.children.values():
            expanded_states[edge.node.state] = edge.node
    # Update edge statistics in a backward pass through each move.
    _backup(edges, node.value)
    return expanded_states


def _mcts_select(node: Node) -> Action:
    """
    Use a variant of PUCT algorithm to select an action during an MCTS simulation.
    """

    def _calculate_puct_score(
        edge: Edge,
        visit_counts_sum: float,  # ∑b N(s,b)
        c_puct: float = 1.0,  # TODO: Tune this hyperparameter
    ) -> float:
        """ """
        # Q(s,a)
        exploitation_term = edge.action_value
        # U(s,a) = c_puct * P(s,a) * sqrt(∑b N(s,b)) / (1+N(s,a))
        exploration_term = c_puct * edge.prior_probability * math.sqrt(visit_counts_sum) / (1 + edge.visit_count)
        # Q(s,a) + U(s,a)
        return exploitation_term + exploration_term

    # NOTE: Assume the node is not a leaf.
    visit_counts_sum = sum([e.visit_count for e in node.children.values()])
    # Select action with the maximum PUCT score.
    action = max(node.children, key=lambda a: _calculate_puct_score(node.children[a], visit_counts_sum))
    return action


def _expand(node: Node, game: Game, model: Model):
    """
    Evaluate a leaf node by assigning a value to its state and prior probabilities to each of its edges.
    """
    # TODO: Confirm which value should be used for a terminal state: value predicted by the model or final reward from gameplay.
    prior_probabilities, value = model.predict_single(node.state.make_input_data())
    # The prior may contain non-zero probabilities for illegal actions. We need to eliminate those and keep only the legal ones.
    legal_actions = game.list_legal_actions(node.state)
    legal_prior_probabilities = {a: p for a, p in prior_probabilities.items() if a in legal_actions}
    # Create new edges that include prior probabilities only for legal actions.
    for action in legal_actions:
        state = game.simulate(node.state, action)
        node.children[action] = Edge(Node(state), legal_prior_probabilities[action])
    node.value = value


def _backup(edges: list[Edge], value: float):
    """
    Update edge statistics.
    """
    for i, edge in enumerate(reversed(edges)):
        # The sign of value flips at each edge as we traverse backward.
        # See `generate_data` function for an explanation, since it applies the same logic.
        is_current_player_last_mover = i % 2 == 0
        cur_value = value * (1 if is_current_player_last_mover else -1)
        edge.action_value = (edge.action_value * edge.visit_count + cur_value) / (edge.visit_count + 1)
        edge.visit_count += 1
