import datetime
import logging
import os
import pathlib
import random
import time

import jax
from flax import nnx

from alphazero.core.game import ReplayBuffer
from alphazero.core.generator import PlayConfig, NoiseSession, generate_data
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
    seed = int(time.time() * 1000)
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"seed={seed}, run_time={run_time}")
    random.seed(seed)
    noise_key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(seed)
    run_dir = STORAGE_DIR / run_time
    os.makedirs(run_dir, exist_ok=True)
    with open(run_dir / "config.log", "w") as config_file:
        config_file.write(f"seed={seed}\n")

    # Initialize the game.
    c_puct = 2.0  # TODO: Tune this hyperparameter
    mnk_game = MnkGame(m, n, k)
    training_play_config = PlayConfig(
        simulations_num=m * n * 5,
        c_puct=c_puct,
        temperature=1.0,  # TODO: Tune this hyperparameter
    )
    evaluation_play_config = PlayConfig(
        simulations_num=m * n * 5,
        c_puct=c_puct,
        temperature=0.1,  # Lower temperature for stronger play
    )
    noise_session = NoiseSession(
        key=noise_key,
        dirichlet_alpha=0.5,
        fraction=0.25,
    )
    mnk_config = MnkConfig(
        learning_rate=0.0005,
        epochs_num=100,
        batch_size=128,
        competitions_num=250,
        competition_margin=0.1,
        play_config=evaluation_play_config,
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
        mnk_config.play_config,
        output_dir=run_dir / "initial" / EVALUATION_DUMMY_BEST_DIR_NAME,
    )
    for i in range(ITERATIONS_NUM):
        logger.info(f"iteration_index={i:0{len(str(ITERATIONS_NUM))}d}")
        for data in generate_data(
            mnk_game,
            mnk_network.get_best_model(),
            1000,
            training_play_config,
            noise_session=noise_session,
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
