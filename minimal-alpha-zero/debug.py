import random
import time

from alphazero.core.model import ModelConfig
from alphazero.core.generator import PlayConfig, _play
from alphazero.games.mnk import MnkGame, DummyModel


m, n, k = 4, 4, 3
seed = int(time.time() * 1000)
print(f"seed={seed}")
random.seed(seed)

mnk_model_config = ModelConfig(calc_mcts_simulations_num=lambda: m * n * 5)
play_config = PlayConfig(
    calc_temperature=lambda _: 0.1,  # Lower temperature for stronger play
)
mnk_game = MnkGame(m, n, k)
dummy_model = DummyModel(m, n)
_play(mnk_game, [(dummy_model, mnk_model_config)], play_config)
