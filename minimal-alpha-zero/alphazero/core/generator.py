"""
Use the best network to generate data for further training.
"""

import logging
import math
import random
from typing import Iterable, Optional

from .game import Action, State, Game, calculate_legal_actions
from .network import Network


logger = logging.getLogger(__name__)


class Node:
    """ """

    state: State
    value: Optional[float]
    children: dict[Action, "Edge"]  # Store only legal actions

    def __init__(self, state: State):
        self.state = state
        self.value = None
        self.children = {}

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
    network: Network,
    self_play_games_num: int = 1,
    self_play_select_simulations_num: int = 1,
) -> Iterable[tuple[State, list[float], float]]:
    """
    Continuously retrieve the best network to play games over many self-plays and store the generated data in a buffer.
    """
    for moves, reward in play(game, network, self_play_games_num, self_play_select_simulations_num):
        if len(moves) == 0 or reward is None:
            # Just in case.
            continue
        for j, (state, search_probabilities) in enumerate(moves):
            # The sign of reward for each move (positive or negative) alternates based on current player's perspective,
            # since two players take turns: Player A moves first, Player B moves second, then Player A again, and so on.
            # It also depends on which player makes the last move and whether that player wins or loses.
            # Therefore, the final reward is assigned to the last move, with its sign flipping at each move as we traverse backward.
            # The same logic is applied to values in `_backup` function.
            # NOTE: In case of a draw, all rewards should be 0.
            is_current_player_last_mover = (len(moves) - 1 - j) % 2 == 0
            cur_reward = reward * (1 if is_current_player_last_mover else -1)
            yield state, search_probabilities, cur_reward


def play(
    game: Game,
    network: Network,
    games_num: int,
    select_simulations_num: int,
) -> list[tuple[list[tuple[State, list[float]]], float]]:
    """
    Play game to generate data for training a new network.
    Each move is selected after executing a specific number of MCTS simulations, guided by the current best network.
    """

    class Record:
        """ """

        node: Node
        moves: list[tuple[State, list[float]]]
        reward: Optional[float]

        def __init__(self, node: Node, moves: list[tuple[State, list[float]]]):
            self.node = node
            self.moves = moves
            self.reward = None

    all_actions = game.list_all_actions()
    records: list[Record] = [Record(Node(game.begin()), []) for _ in range(games_num)]
    while True:
        # The final reward is computed at this state, but it is attributed to the last move that produced this state.
        for record in records:
            if record.reward is None:
                record.reward = game.receive_reward_if_terminal(record.node.state)
        is_terminated = all(r.reward is not None for r in records)
        if is_terminated:
            break
        playing_records = [r for r in records if r.reward is None]
        # Select the next action.
        for (action, legal_searches), record in zip(
            _play_select([r.node for r in playing_records], game, network, select_simulations_num), playing_records
        ):
            search_probabilities = [legal_searches[a] if a in legal_searches else 0.0 for a in all_actions]
            record.moves.append((record.node.state, search_probabilities))
            # Move to the next node.
            record.node = record.node.children[action].node
    return [(r.moves, r.reward) for r in records]


def _play_select(
    nodes: list[Node],
    game: Game,
    network: Network,
    simulations_num: int,
    temperature: float = 1.0,  # TODO: Tune this hyperparameter
) -> list[tuple[Action, dict[Action, float]]]:
    """
    Select a legal action to move to a new state.
    """
    i = 0
    # Execute MCTS simulations.
    while i <= simulations_num:
        _execute_mcts_simulation(nodes, game, network)
        # Next simulation.
        i += 1
    selections: list[tuple[Action, dict[Action, float]]] = []
    for node in nodes:
        # Select a move according to the search probabilities π computed by MCTS.
        visit_counts_sum = sum([e.visit_count for e in node.children.values()])
        legal_searches = {
            a: math.pow(e.visit_count, 1 / temperature) / math.pow(visit_counts_sum, 1 / temperature)
            for a, e in node.children.items()
        }
        # TODO: Consider another approach (e.g., using softmax for the weights, or always selecting the node with the most visits).
        action = random.choices(list(legal_searches.keys()), weights=list(legal_searches.values()))[0]
        selections.append((action, legal_searches))
    return selections


def _execute_mcts_simulation(roots: list[Node], game: Game, network: Network):
    """
    NOTE: By "root", we mean the node where MCTS simulation begins, not an initial game state.
    """
    node_edges: list[tuple[Node, list[Edge]]] = []
    expanding_nodes: list[Node] = []
    for node in roots:
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
        # We evaluate a leaf node only once during a single game.
        # If the node is not a leaf, it should be a terminal node, in that case, we do not expand it again.
        if node.is_leaf:
            expanding_nodes.append(node)
        node_edges.append((node, edges))
    # Expand and evaluate a leaf node.
    if len(expanding_nodes) > 0:
        _expand(expanding_nodes, game, network)
    # Update edge statistics in a backward pass through each move.
    for node, edges in node_edges:
        _backup(edges, node.value)


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
        exploitation_term = edge.action_value
        exploration_term = c_puct * edge.prior_probability * math.sqrt(visit_counts_sum) / (1 + edge.visit_count)
        return exploitation_term + exploration_term

    # NOTE: Assume the node is not a leaf.
    visit_counts_sum = sum([e.visit_count for e in node.children.values()])
    # Select action with the maximum PUCT score.
    action = max(node.children, key=lambda a: _calculate_puct_score(node.children[a], visit_counts_sum))
    return action


def _expand(nodes: list[Node], game: Game, network: Network):
    """
    Evaluate a leaf node by assigning a value to its state and prior probabilities to each of its edges.
    """
    # TODO: Confirm which value should be used for a terminal state: value predicted by the network or final reward from gameplay.
    predictions = network.best_model_predict([n.state.make_input_data() for n in nodes])
    for i, (prior_probabilities, value) in enumerate(predictions):
        node = nodes[i]
        # The prior may contain non-zero probabilities for illegal actions. We need to eliminate those and keep only the legal ones.
        legal_actions, legal_prior_probabilities = calculate_legal_actions(prior_probabilities, game, node.state)
        # Update the prior probabilities to include only legal actions.
        node.children = {
            a: Edge(Node(game.simulate(node.state, a)), legal_prior_probabilities[legal_actions.index(a)])
            for a in legal_actions
        }
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
