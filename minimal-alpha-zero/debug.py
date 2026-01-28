import random
import time

from alphazero.core.model import ModelConfig
from alphazero.core.generator import PlayConfig, play
from alphazero.games.mnk import MnkGame, DummyModel


m, n, k = 4, 4, 3
seed = int(time.time() * 1000)
print(f"seed={seed}")
random.seed(seed)

mnk_model_config = ModelConfig(mcts_simulations_num=m * n * 5)
play_config = PlayConfig(
    calc_temperature=lambda _: 0.1,  # Lower temperature for stronger play
)
mnk_game = MnkGame(m, n, k)
dummy_model = DummyModel(m, n)
play(mnk_game, [(dummy_model, mnk_model_config)], play_config)
