import random
from typing import Optional

import pytest
from flax import nnx

from ..core.generator import PlayConfig
from .mnk import StoneColor, Stone, MnkAction, MnkState, MnkGame, MnkConfig, MnkNetwork, evaluate


class TestMnk:
    """ """

    @pytest.mark.parametrize("times", range(500))
    def test_game_simulate(self, times):
        """ """
        # Generate a state.
        m = random.randint(8, 16)
        n = random.randint(8, 16)
        k = random.randint(4, 8)
        mnk_game = MnkGame(m, n, k)
        red_count = 0
        green_count = 0
        board: list[list[Optional[Stone]]] = []
        actions: list[MnkAction] = []
        for y in range(m):
            row: list[Optional[Stone]] = []
            for x in range(n):
                color = random.choice([StoneColor.RED, StoneColor.GREEN, None])
                row.append(Stone(color) if color is not None else None)
                if color is not None:
                    actions.append(MnkAction(x, y))
                    if color == StoneColor.RED:
                        red_count += 1
                    else:
                        green_count += 1
            board.append(row)
        state = MnkState(board, red_count + green_count, random.choice(actions))
        # Simulate a new state.
        new_action = MnkAction(random.randint(0, n - 1), random.randint(0, m - 1))
        new_state = mnk_game.simulate(state, new_action)
        for y in range(m):
            for x in range(n):
                if x != new_action.x or y != new_action.y or state.board[y][x] is not None:
                    assert (
                        new_state.board[y][x] is None
                        and state.board[y][x] is None
                        or new_state.board[y][x].color == state.board[y][x].color
                    )
                else:
                    is_in_red_turn = red_count <= green_count
                    assert new_state.board[y][x].color == StoneColor.RED if is_in_red_turn else StoneColor.GREEN

    def test_network_evaluate(self):
        """ """
        m, n, k = 4, 4, 3
        mnk_game = MnkGame(m, n, k)
        play_config = PlayConfig(
            simulations_num=m * n * 5,
            c_puct=2.0,
            calculate_temperature=lambda _: 0.1,  # Lower temperature for stronger play
        )
        mnk_config = MnkConfig(
            learning_rate=0.0005,
            epochs_num=100,
            batch_size=128,
            stopping_patience=5,
            competitions_num=250,
            competition_margin=0.1,
            play_config=play_config,
        )
        mnk_network = MnkNetwork(m, n, mnk_config)
        candidate_model = nnx.clone(mnk_network.get_best_model())
        result = evaluate(
            mnk_network.get_best_model(),
            candidate_model,
            mnk_game,
            mnk_config.competitions_num,
            mnk_config.play_config,
        )
        assert abs(result) < mnk_config.competition_margin
