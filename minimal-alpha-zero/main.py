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
from alphazero.games.mnk import (
    MnkGame,
    MnkConfig,
    MnkNetwork,
    DummyModel,
    evaluate,
    augment_data,
    format_board,
    EVALUATION_DUMMY_BEST_DIR_NAME,
)


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
    self_plays_num = 250

    def _calculate_temperature(move_index: int) -> float:
        """ """
        MAX_TEMPERATURE = 1.0
        MIN_TEMPERATURE = 0.1
        COOLDOWN_START_INDEX = 0
        COOLDOWN_END_INDEX = (k - 1) * 2
        if move_index < COOLDOWN_START_INDEX:
            return MAX_TEMPERATURE
        scaler = (move_index - COOLDOWN_START_INDEX) / (COOLDOWN_END_INDEX - COOLDOWN_START_INDEX)
        temperature = scaler * (MIN_TEMPERATURE - MAX_TEMPERATURE) + MAX_TEMPERATURE
        return max(temperature, MIN_TEMPERATURE)

    mnk_game = MnkGame(m, n, k)
    training_play_config = PlayConfig(
        simulations_num=m * n * 5,
        c_puct=c_puct,
        calculate_temperature=_calculate_temperature,  # TODO: Tune this hyperparameter
    )
    evaluation_play_config = PlayConfig(
        simulations_num=m * n * 5,
        c_puct=c_puct,
        # Keep a high temperature for the first two moves (one move per turn for each player),
        # then lower the temperature for the rest to achieve stronger play.
        calculate_temperature=lambda i: 1.0 if i <= 1 else 0.1,
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
        stopping_patience=5,
        competitions_num=250,
        competition_margin=0.1,
        play_config=evaluation_play_config,
        rngs=rngs,
    )
    mnk_network = MnkNetwork(m, n, mnk_config)
    replay_buffer = ReplayBuffer(self_plays_num * 128)
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
        output_dir = run_dir / f"iteration_{i:0{len(str(ITERATIONS_NUM))}d}"
        data_output_dir = output_dir / "data"
        os.makedirs(data_output_dir, exist_ok=True)
        j = 0
        for data_list in generate_data(
            mnk_game,
            mnk_network.get_best_model(),
            self_plays_num,
            training_play_config,
            noise_session=noise_session,
        ):
            data_file_name = f"self_play-{j:0{len(str(self_plays_num))}d}"
            # Store and log the training data.
            with (
                open(data_output_dir / f"{data_file_name}-00_original.txt", "w") as original_data_file,
                open(data_output_dir / f"{data_file_name}-01_vertical.txt", "w") as vertical_data_file,
                open(data_output_dir / f"{data_file_name}-02_horizontal.txt", "w") as horizontal_data_file,
                open(data_output_dir / f"{data_file_name}-03_central.txt", "w") as central_data_file,
            ):
                for original_data in data_list:
                    augmented_data_list = augment_data(original_data)
                    for data, data_file in zip(
                        [
                            original_data,
                            augmented_data_list["vertical"],
                            augmented_data_list["horizontal"],
                            augmented_data_list["central"],
                        ],
                        [original_data_file, vertical_data_file, horizontal_data_file, central_data_file],
                    ):
                        replay_buffer.append(data)
                        state, search_probabilities, reward = data
                        data_file.write(f"{state}\n")
                        data_file.write(f"Search probabilities:\n{format_board(search_probabilities, m)}\n")
                        data_file.write(f"Reward: {reward}\n")
                        data_file.write("\n")
                        data_file.flush()
            j += 1
        logger.info(f"replay_buffer_len={len(replay_buffer.buffer)}")
        is_best_model_updated = mnk_network.train_and_evaluate(replay_buffer, mnk_game, output_dir)
        # Clear the replay buffer after updating the best model.
        if is_best_model_updated:
            replay_buffer.reset()


if __name__ == "__main__":
    main()
