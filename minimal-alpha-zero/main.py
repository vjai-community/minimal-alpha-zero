import logging
import pathlib
import random
import time

from flax import nnx

from alphazero.core.game import ReplayBuffer
from alphazero.core.generator import generate_data
from alphazero.games.mnk import MnkGame, MnkConfig, MnkNetwork


STORAGE_DIR = pathlib.Path(__file__).parent.parent / "storage"
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] (%(asctime)s) %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    seed = int(time.time() * 1000)
    logger.info(f"seed={seed}")
    # We want the first player to be able to force a win.
    # Doc: https://en.wikipedia.org/wiki/M,n,k-game#Specific_results.
    m, n, k = 4, 4, 3
    # If we want the first player to be unable to always force a win, but still have a reasonable chance of winning,
    # consider choosing: m, n, k = (5, 5, 4)
    random.seed(seed)
    mnk_game = MnkGame(m, n, k)
    mnk_config = MnkConfig(
        learning_rate=0.0005,
        epochs_num=100,
        batch_size=128,
        competitions_num=250,
        competition_margin=0.1,
        model_dir=STORAGE_DIR / "model",
        rngs=nnx.Rngs(seed),
    )
    mnk_network = MnkNetwork(m, n, mnk_config)
    replay_buffer = ReplayBuffer(1024)
    ITERATIONS_NUM = 100
    for i in range(ITERATIONS_NUM):
        logger.info(f"iteration_index={i}")
        for data in generate_data(mnk_game, mnk_network, self_plays_num=20, self_play_select_simulations_num=m * n * 2):
            replay_buffer.append(data)
        logger.info(f"replay_buffer_len={len(replay_buffer.buffer)}")
        mnk_network.train_and_evaluate(replay_buffer, mnk_game)
        # TODO: Consider clearing the replay buffer after updating the model weights.


if __name__ == "__main__":
    main()
