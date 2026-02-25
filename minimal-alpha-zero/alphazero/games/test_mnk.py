import random
from typing import Optional

import pytest
from flax import nnx

from ..core.model import ModelConfig
from ..core.generator import PlayConfig, EvaluationConfig
from .mnk import (
    MnkPieceColor,
    MnkPiece,
    MnkAction,
    MnkState,
    MnkGame,
    MnkNetwork,
    evaluate,
)


class TestMnk:
    """ """

    @pytest.mark.parametrize("times", range(500))
    def test_game_transition(self, times):
        """ """
        # Generate a state.
        m = random.randint(8, 16)
        n = random.randint(8, 16)
        k = random.randint(4, 8)
        mnk_game = MnkGame(m, n, k)
        red_count = 0
        green_count = 0
        board: list[list[Optional[MnkPiece]]] = []
        actions: list[MnkAction] = []
        for y in range(n):
            row: list[Optional[MnkPiece]] = []
            for x in range(m):
                color = random.choice([MnkPieceColor.RED, MnkPieceColor.GREEN, None])
                row.append(MnkPiece(color) if color is not None else None)
                if color is not None:
                    actions.append(MnkAction(x, y))
                    if color == MnkPieceColor.RED:
                        red_count += 1
                    else:
                        green_count += 1
            board.append(row)
        state = MnkState(board, red_count + green_count, random.choice(actions))
        # Transition to a new state.
        new_action = MnkAction(random.randint(0, m - 1), random.randint(0, n - 1))
        new_state = mnk_game.transition(state, new_action)
        for y in range(n):
            for x in range(m):
                if x != new_action.x or y != new_action.y or state.board[y][x] is not None:
                    assert (
                        new_state.board[y][x] is None
                        and state.board[y][x] is None
                        or new_state.board[y][x].color == state.board[y][x].color
                    )
                else:
                    assert new_state.board[y][x].color == state.get_current_player_color()

    def test_network_evaluate(self):
        """ """
        m, n, k = 4, 4, 3
        mnk_game = MnkGame(m, n, k)
        mnk_model_config = ModelConfig(calc_mcts_simulations_num=lambda: m * n * 5)
        evaluation_config = EvaluationConfig(
            competitions_num=250,
            competition_margin=0.1,
            game=mnk_game,
            play_config=PlayConfig(
                c_puct=2.0,
                calc_temperature=lambda _: 0.1,  # Lower temperature for stronger play
            ),
        )
        mnk_network = MnkNetwork(m, n)
        candidate_model = nnx.clone(mnk_network.get_best_model())
        result = evaluate(
            (mnk_network.get_best_model(), mnk_model_config),
            (candidate_model, mnk_model_config),
            evaluation_config,
        )
        assert result is None
