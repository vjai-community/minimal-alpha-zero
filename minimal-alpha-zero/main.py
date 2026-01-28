import datetime
import logging
import os
import pathlib
import random
import time

from flax import nnx

from alphazero.core.game import ReplayBuffer
from alphazero.core.generator import generate_data
from alphazero.games.mnk import MnkGame, MnkConfig, MnkNetwork, DummyModel, evaluate, EVALUATION_DUMMY_BEST_DIR_NAME


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
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"seed={seed}, run_time={run_time}")
    random.seed(seed)
    rngs = nnx.Rngs(seed)
    run_dir = STORAGE_DIR / run_time
    os.makedirs(run_dir, exist_ok=True)
    with open(run_dir / "config.log", "w") as config_file:
        config_file.write(f"seed={seed}\n")

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
        rngs=rngs,
    )
    mnk_network = MnkNetwork(m, n, mnk_config)
    replay_buffer = ReplayBuffer(8192)
    dummy_model = DummyModel(m, n)

    # Training and evaluating.
    evaluate(
        dummy_model,
        mnk_network.get_best_model(),
        mnk_game,
        mnk_config.competitions_num,
        mnk_config.select_simulations_num,
        mnk_config.select_temperature,
        output_dir=run_dir / "initial" / EVALUATION_DUMMY_BEST_DIR_NAME,
    )
    for i in range(ITERATIONS_NUM):
        logger.info(f"iteration_index={i:0{len(str(ITERATIONS_NUM))}d}")
        for data in generate_data(
            mnk_game,
            mnk_network.get_best_model(),
            self_plays_num=1000,
            self_play_select_simulations_num=select_simulations_num,
        ):
            replay_buffer.append(data)
        logger.info(f"replay_buffer_len={len(replay_buffer.buffer)}")
        is_best_model_updated = mnk_network.train_and_evaluate(
            replay_buffer,
            mnk_game,
            run_dir / f"iteration_{i:0{len(str(ITERATIONS_NUM))}d}",
        )
        # Clear the replay buffer after updating the best model.
        if is_best_model_updated:
            replay_buffer.reset()


if __name__ == "__main__":
    main()
