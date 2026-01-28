import logging
import pathlib
import random
import time

from flax import nnx

from alphazero.core.game import ReplayBuffer
from alphazero.core.generator import generate_data
from alphazero.games.mnk import MnkGame, MnkConfig, MnkNetwork, DummyModel, evaluate


STORAGE_DIR = pathlib.Path(__file__).parent.parent / "storage"
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] (%(asctime)s) %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Configs
    ITERATIONS_NUM = 100
    # We want the first player to be able to force a win.
    # Doc: https://en.wikipedia.org/wiki/M,n,k-game#Specific_results.
    m, n, k = 4, 4, 3
    # If we want the first player to be unable to always force a win, but still have a reasonable chance of winning,
    # consider choosing: m, n, k = (5, 5, 4)
    select_simulations_num = m * n * 2
    seed = int(time.time() * 1000)
    logger.info(f"seed={seed}")
    random.seed(seed)
    rngs = nnx.Rngs(seed)

    # Initialize the game.
    mnk_game = MnkGame(m, n, k)
    mnk_config = MnkConfig(
        learning_rate=0.0005,
        epochs_num=100,
        batch_size=128,
        competitions_num=250,
        competition_margin=0.1,
        select_simulations_num=select_simulations_num,
        select_temperature=0.1,
        model_dir=STORAGE_DIR / "model",
        rngs=rngs,
    )
    mnk_network = MnkNetwork(m, n, mnk_config)
    replay_buffer = ReplayBuffer(8192)
    dummy_model = DummyModel(m, n)

    # Training and evaluating.
    result = evaluate(
        dummy_model,
        mnk_network.best_model,
        mnk_game,
        mnk_config.competitions_num,
        mnk_config.select_simulations_num,
        mnk_config.select_temperature,
    )
    logger.info(f"model1=dummy_model, model2=best_model, result={result}")
    for i in range(ITERATIONS_NUM):
        logger.info(f"iteration_index={i}")
        for data in generate_data(
            mnk_game,
            mnk_network.best_model,
            self_plays_num=1000,
            self_play_select_simulations_num=select_simulations_num,
        ):
            replay_buffer.append(data)
        logger.info(f"replay_buffer_len={len(replay_buffer.buffer)}")
        mnk_network.train_and_evaluate(replay_buffer, mnk_game)
        # TODO: Consider clearing the replay buffer after updating the model weights.


if __name__ == "__main__":
    main()
