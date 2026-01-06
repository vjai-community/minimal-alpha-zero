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
    m, n, k = 8, 6, 4
    random.seed(seed)
    logger.info(f"seed={seed}")
    mnk_game = MnkGame(m, n, k, initial_stones_num=4)
    mnk_config = MnkConfig(
        learning_rate=0.005,
        epochs_num=1,
        batch_size=128,
        competitions_num=256,
        competition_margin=0.1,
        model_dir=STORAGE_DIR / "model",
        rngs=nnx.Rngs(seed),
    )
    mnk_network = MnkNetwork(m, n, mnk_config)
    replay_buffer = ReplayBuffer(1024)
    ITERATIONS_NUM = 100
    for i in range(ITERATIONS_NUM):
        logger.info(f"iteration_index={i}")
        for data in generate_data(mnk_game, mnk_network, self_plays_num=16, self_play_select_simulations_num=m * n * 2):
            replay_buffer.append(data)
        logger.info(f"replay_buffer_len={len(replay_buffer.buffer)}")
        mnk_network.train_and_evaluate(replay_buffer, mnk_game)
        # TODO: Consider clearing the replay buffer after updating the model weights.


if __name__ == "__main__":
    main()
