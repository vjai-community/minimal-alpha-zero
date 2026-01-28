import random
import time

from flax import nnx

from alphazero.core.network import ModelConfig
from alphazero.core.generator import PlayConfig, play
from alphazero.games.mnk import MnkGame, MnkModel


m, n, k = 4, 1, 2
seed = int(time.time() * 1000)
print(f"seed={seed}")
random.seed(seed)
rngs = nnx.Rngs(seed)

play_config = PlayConfig(
    simulations_num=m * n * 2,
    calc_temperature=lambda _: 0.1,  # Lower temperature for stronger play
)
mnk_game = MnkGame(m, n, k)
mnk_model = MnkModel(m, n, rngs)
play(mnk_game, [(mnk_model, ModelConfig())], play_config)
