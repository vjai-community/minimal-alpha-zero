import copy
import datetime
import logging
import os
import pathlib
import random
import re
import time

from flax import nnx

from alphazero.core.game import ReplayBuffer
from alphazero.core.model import ModelConfig
from alphazero.core.generator import PlayConfig, EvaluationConfig, GenerationConfig, evaluate, generate_data
from alphazero.games.mnk import (
    MnkGame,
    MnkTrainingConfig,
    MnkNetwork,
    DummyModel,
    augment_data,
    format_board,
    log_competition,
    EVALUATION_DUMMY_BEST_DIR_NAME,
)


STORAGE_DIR = pathlib.Path(__file__).parent.parent / "storage"
run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = STORAGE_DIR / run_time
os.makedirs(run_dir, exist_ok=True)
logging.basicConfig(
    filename=run_dir / "run.log",
    level=logging.INFO,
    format="[%(levelname)s] (%(asctime)s) %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Configs
    ITERATIONS_NUM = 10
    ITERATION_STOPPING_PATIENCE = 2
    # We want the first player to be able to force a win.
    # Doc: https://en.wikipedia.org/wiki/M,n,k-game#Specific_results.
    m, n, k = 4, 4, 3
    # If we want the first player to be unable to always force a win, but still have a reasonable chance of winning,
    # consider choosing: m, n, k = (5, 5, 4)
    seed = int(time.time() * 1000)
    logger.info(f"seed={seed}, run_time={run_time}")
    random.seed(seed)
    rngs = nnx.Rngs(seed)

    # Initialize the game.
    c_puct = 2.0  # TODO: Tune this hyperparameter

    def _calc_temperature(move_index: int) -> float:
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
    mnk_model_config = ModelConfig(calc_mcts_simulations_num=lambda: m * n * 5)  # TODO: Tune this hyperparameter
    baseline_model_config = ModelConfig(calc_mcts_simulations_num=lambda: m * n * 5)
    dummy_model_config = ModelConfig(calc_mcts_simulations_num=lambda: random.randint(m * n * 5, m * n * 20))
    generation_config = GenerationConfig(
        self_plays_num=250,
        game=mnk_game,
        play_config=PlayConfig(
            c_puct=c_puct,
            calc_temperature=_calc_temperature,  # TODO: Tune this hyperparameter
        ),
    )
    dummy_generation_config = copy.deepcopy(generation_config)
    dummy_generation_config.self_plays_num = 50
    evaluation_config = EvaluationConfig(
        competitions_num=250,
        competition_margin=0.1,
        game=mnk_game,
        play_config=PlayConfig(
            c_puct=c_puct,
            # Keep a high temperature for the first two moves (one move per turn for each player),
            # then lower the temperature for the rest to achieve stronger play.
            calc_temperature=lambda i: 1.0 if i <= 1 else 0.1,
        ),
        log_competition=log_competition,
    )
    mnk_training_config = MnkTrainingConfig(
        learning_rate=0.0005,
        epochs_num=100,
        batch_size=128,
        stopping_patience=5,
    )
    mnk_network = MnkNetwork(m, n, rngs)
    replay_buffer = ReplayBuffer()
    dummy_model = DummyModel(m, n)
    with open(run_dir / "config.log", "w") as config_file:
        config_file.write(f"seed={seed}\n")
        config_file.write(
            # Remove ANSI color codes.
            "model=" + re.sub(r"\x1b\[[0-9;]*m", "", str(mnk_network.get_best_model())) + "\n"
        )

    # Training and evaluating.
    evaluate(
        (dummy_model, baseline_model_config),
        (mnk_network.get_best_model(), mnk_model_config),
        evaluation_config,
        output_dir=run_dir / "initial" / EVALUATION_DUMMY_BEST_DIR_NAME,
    )
    i = 0
    iteration_patience_count = 0
    aggregation_excluded_model_names = set([dummy_model.name])
    while True:
        logger.info(f"iteration_index={i:0{len(str(ITERATIONS_NUM))}d}")
        output_dir = run_dir / f"iteration_{i:0{len(str(ITERATIONS_NUM))}d}"
        data_output_dir = output_dir / "data"
        os.makedirs(data_output_dir, exist_ok=True)
        j = 0
        replay_buffer.enqueue_new_buffer()  # Store the data from each iteration in separate buffers
        partner_model_choices = [
            # TODO: Tune the weights.
            (mnk_network.get_best_model(), mnk_model_config, 0.8),
            (dummy_model, baseline_model_config, 0.2),
        ]

        # Generate only last moves from a dummy player.
        with (
            open(data_output_dir / "dummy_last_moves-00_original.txt", "w") as original_data_file,
            open(data_output_dir / "dummy_last_moves-01_vertical.txt", "w") as vertical_data_file,
            open(data_output_dir / "dummy_last_moves-02_horizontal.txt", "w") as horizontal_data_file,
            open(data_output_dir / "dummy_last_moves-03_central.txt", "w") as central_data_file,
        ):
            for _, data_list in generate_data((dummy_model, dummy_model_config), dummy_generation_config):
                original_data = data_list[-1]  # Last move
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
                    replay_buffer.append_to_newest_buffer(data)
                    state, probs, reward = data
                    # Prioritize the last action to have the highest probability.
                    highest_prob = max(probs)
                    highest_prob_count = probs.count(highest_prob)
                    probs = [1.0 / highest_prob_count if p == highest_prob else 0.0 for p in probs]
                    data_file.write(f"{state}\n")
                    data_file.write(f"Probabilities:\n{format_board(probs, m)}\n")
                    data_file.write(f"Reward: {reward:.2f}\n")
                    data_file.write("\n")
                    data_file.flush()
        # Generate data from self-play.
        for model_names, data_list in generate_data(
            (mnk_network.get_best_model(), mnk_model_config),
            generation_config,
            partner_model_choices=partner_model_choices,
            aggregation_excluded_model_names=aggregation_excluded_model_names,
        ):
            data_file_name = f"self_play-{j:0{len(str(generation_config.self_plays_num))}d}"
            # Store and log the training data.
            with (
                open(data_output_dir / f"{data_file_name}-00_original.txt", "w") as original_data_file,
                open(data_output_dir / f"{data_file_name}-01_vertical.txt", "w") as vertical_data_file,
                open(data_output_dir / f"{data_file_name}-02_horizontal.txt", "w") as horizontal_data_file,
                open(data_output_dir / f"{data_file_name}-03_central.txt", "w") as central_data_file,
            ):
                for data_file in [original_data_file, vertical_data_file, horizontal_data_file, central_data_file]:
                    detailed_model_names = [
                        f"{n} (excluded)" if n in aggregation_excluded_model_names else n for n in model_names
                    ]
                    data_file.write(f"Models: {', '.join(detailed_model_names)}\n\n")
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
                        replay_buffer.append_to_newest_buffer(data)
                        state, search_probs, reward = data
                        data_file.write(f"{state}\n")
                        data_file.write(f"Search probabilities:\n{format_board(search_probs, m)}\n")
                        data_file.write(f"Reward: {reward:.2f}\n")
                        data_file.write("\n")
                        data_file.flush()
            j += 1
        logger.info(f"replay_buffer_len={sum(len(b) for b in replay_buffer.buffer_queue)}")

        is_best_model_updated = mnk_network.train_and_evaluate(
            replay_buffer, mnk_training_config, evaluation_config, mnk_model_config, output_dir
        )
        if is_best_model_updated:
            evaluate(
                (dummy_model, baseline_model_config),
                (mnk_network.get_best_model(), mnk_model_config),
                evaluation_config,
                output_dir=output_dir / EVALUATION_DUMMY_BEST_DIR_NAME,
            )
            iteration_patience_count = 0
            # Discard the oldest buffers after updating the best model.
            replay_buffer.discard_oldest_buffers()
        else:
            iteration_patience_count += 1
        if i >= ITERATIONS_NUM - 1 or iteration_patience_count == ITERATION_STOPPING_PATIENCE:
            break
        # Next iteration.
        i += 1


if __name__ == "__main__":
    main()
