import argparse
import datetime
import logging
import math
import os
import pathlib
import random
import re
import time
from typing import Optional

import jax
import numpy as np
from flax import nnx

from alphazero.core.game import ReplayBuffer
from alphazero.core.model import ModelConfig
from alphazero.core.generator import (
    PlayConfig,
    EvaluationConfig,
    NoiseSession,
    GenerationConfig,
    evaluate,
    generate_data,
    execute_mcts_fresh,
)
from alphazero.games.mnk import (
    StoneColor,
    MnkAction,
    MnkState,
    MnkGame,
    MnkTrainingConfig,
    MnkModel,
    MnkNetwork,
    DummyModel,
    augment_data,
    format_board,
    log_competition,
    EVALUATION_DUMMY_BEST_DIR_NAME,
)


def main():
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args()
    # We want the first player to be able to force a win.
    # Doc: https://en.wikipedia.org/wiki/M,n,k-game#Specific_results.
    game_shape = (6, 5, 4)
    if args.mode == "train":
        _train(game_shape)
    elif args.mode == "play":
        _play(game_shape, args.checkpoint_dir)


def _train(game_shape: tuple[int, int, int]):
    """ """
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

    # FIXME: For some reason, putting this import at the top prevents "run.log" file from being created.
    import orbax.checkpoint as ocp

    m, n, k = game_shape
    # Configs
    ITERATIONS_NUM = 100
    ITERATION_STOPPING_PATIENCE = 2
    seed = int(time.time() * 1000)
    logger.info(f"seed={seed}, run_time={run_time}")
    random.seed(seed)
    rngs = nnx.Rngs(seed)
    noise_key = jax.random.PRNGKey(seed)

    # Initialize the game.
    c_puct = 2.0  # TODO: Tune this hyperparameter

    def _calc_temperature(move_index: int) -> float:
        """ """
        MAX_TEMPERATURE = 1.0
        MIN_TEMPERATURE = 0.3
        COOLDOWN_START_INDEX = 0
        COOLDOWN_END_INDEX = (k - 1) * 2
        if move_index < COOLDOWN_START_INDEX:
            return MAX_TEMPERATURE
        scaler = (move_index - COOLDOWN_START_INDEX) / (COOLDOWN_END_INDEX - COOLDOWN_START_INDEX)
        temperature = scaler * (MIN_TEMPERATURE - MAX_TEMPERATURE) + MAX_TEMPERATURE
        return max(temperature, MIN_TEMPERATURE)

    mnk_game = MnkGame(m, n, k, should_prefer_fast_win=True)
    lazy_model_config = ModelConfig(calc_mcts_simulations_num=None)
    mnk_model_config = ModelConfig(calc_mcts_simulations_num=lambda: m * n * 10)  # TODO: Tune this hyperparameter
    baseline_model_config = ModelConfig(calc_mcts_simulations_num=lambda: m * n * 10)
    generation_config = GenerationConfig(
        self_plays_num=1000,
        game=mnk_game,
        play_config=PlayConfig(
            c_puct=c_puct,
            calc_temperature=_calc_temperature,  # TODO: Tune this hyperparameter
        ),
        noise_session=NoiseSession(
            key=noise_key,
            dirichlet_alpha=12 / (m * n),  # TODO: Tune this hyperparameter
            fraction=0.25,
        ),
    )
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
        data_split_ratio=0.9,
        learning_rate=0.0005,
        epochs_num=100,
        batch_size=128,
        stopping_patience=5,
    )
    mnk_network = MnkNetwork(m, n, rngs)
    replay_buffer = ReplayBuffer(buffer_queue_size=2)
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
        (mnk_network.get_best_model(), lazy_model_config),
        evaluation_config,
        output_dir=run_dir / "initial" / EVALUATION_DUMMY_BEST_DIR_NAME,
    )
    i = 0
    iteration_patience_count = 0
    while True:
        logger.info(f"iteration_index={i:0{len(str(ITERATIONS_NUM))}d}")
        output_dir = run_dir / f"iteration_{i:0{len(str(ITERATIONS_NUM))}d}"
        data_output_dir = output_dir / "data"
        os.makedirs(data_output_dir, exist_ok=True)
        j = 0
        replay_buffer.enqueue_new_buffer()  # Store the data from each iteration in separate buffers
        for _, data_list in generate_data((mnk_network.get_best_model(), mnk_model_config), generation_config):
            data_file_name = f"self_play-{j:0{len(str(generation_config.self_plays_num))}d}"
            # Store and log the training data.
            with open(data_output_dir / f"{data_file_name}.log", "w") as data_file:
                for original_data in data_list:
                    augmented_data_list = augment_data(original_data)
                    for data in [
                        original_data,
                        augmented_data_list["vertical"],
                        augmented_data_list["horizontal"],
                        augmented_data_list["central"],
                    ]:
                        replay_buffer.append_to_newest_buffer(data)
                    state, search_probs, reward = original_data
                    data_file.write(f"{state}\n")
                    data_file.write(f"Search probabilities (%):\n{format_board(state, search_probs)}\n")
                    data_file.write(f"Reward: {reward:.2f}\n")
                    data_file.write("\n")
                    data_file.flush()
            j += 1
        logger.info(f"replay_buffer_len={sum(len(b) for b in replay_buffer.buffer_queue)}")
        is_best_model_updated = mnk_network.train_and_evaluate(
            replay_buffer, mnk_training_config, evaluation_config, lazy_model_config, output_dir
        )
        if is_best_model_updated:
            evaluate(
                (dummy_model, baseline_model_config),
                (mnk_network.get_best_model(), lazy_model_config),
                evaluation_config,
                output_dir=output_dir / EVALUATION_DUMMY_BEST_DIR_NAME,
            )
            iteration_patience_count = 0
            # Discard the oldest buffers after updating the best model.
            replay_buffer.discard_oldest_buffers()
            # Save the checkpoint.
            checkpoint_dir_name = "state"
            _, state = nnx.split(mnk_network.get_best_model())
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(output_dir / checkpoint_dir_name, state)
            checkpointer.wait_until_finished()
            abstract_model = nnx.eval_shape(lambda: MnkModel(m, n, nnx.Rngs(0)))
            _, abstract_state = nnx.split(abstract_model)
            restored_state = checkpointer.restore(output_dir / checkpoint_dir_name, abstract_state)
            jax.tree.map(np.testing.assert_array_equal, state, restored_state)
        else:
            iteration_patience_count += 1
        if i >= ITERATIONS_NUM - 1 or iteration_patience_count == ITERATION_STOPPING_PATIENCE:
            break
        # Next iteration.
        i += 1


def _play(game_shape: tuple[int, int, int], checkpoint_dir: Optional[str]):
    """ """
    # FIXME: For some reason, putting this import at the top prevents "run.log" file from being created.
    import orbax.checkpoint as ocp

    if checkpoint_dir is None:
        checkpoint_dir = pathlib.Path(__file__).parent.parent / "backup" / "best_model" / "state"
    else:
        checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()

    m, n, k = game_shape
    # Configs
    seed = int(time.time() * 1000)
    random.seed(seed)
    simulations_num = m * n  # Increasing simulations improves model strength
    temperature = 0.1  # Decreasing temperature concentrates the focus on high-probability actions
    # Load the checkpoint.
    checkpointer = ocp.StandardCheckpointer()
    abstract_model = nnx.eval_shape(lambda: MnkModel(m, n, nnx.Rngs(0)))
    graph_def, abstract_state = nnx.split(abstract_model)
    restored_state = checkpointer.restore(checkpoint_dir, abstract_state)
    best_model = nnx.merge(graph_def, restored_state)

    # Play a game.
    def _make_color(board_str: str) -> str:
        """ """
        red = StoneColor.RED.get_mark(is_last_stone=False)
        big_red = StoneColor.RED.get_mark(is_last_stone=True)
        green = StoneColor.GREEN.get_mark(is_last_stone=False)
        big_green = StoneColor.GREEN.get_mark(is_last_stone=True)
        board_str = board_str.replace(red, f"\033[0;31m{red}\033[0m")
        board_str = board_str.replace(big_red, f"\033[1;31m{big_red}\033[0m")
        board_str = board_str.replace(green, f"\033[0;32m{green}\033[0m")
        board_str = board_str.replace(big_green, f"\033[1;32m{big_green}\033[0m")
        return board_str

    def _parse_action(action_str: str) -> Optional[MnkAction]:
        """ """
        action_pos = action_str.split(" ")
        if len(action_pos) != 2:
            return None
        x, y = action_pos
        if not x.isdigit() or not y.isdigit():
            return None
        return MnkAction(int(x), int(y))

    def _ensemble_augmented_predictions(state: MnkState, probs: list[float]) -> list[float]:
        """ """
        count = 1
        avg_probs: list[float] = [p for p in probs]
        for augmented_state, augmented_probs, _ in augment_data((state, probs, 0.0)).values():  # Use a dummy `0` value
            if augmented_state == state:
                avg_probs = [sum(ps) for ps in zip(avg_probs, augmented_probs)]
                count += 1
        avg_probs = [p / count for p in avg_probs]
        return avg_probs

    print("AI's configurations:")
    print("- Number of MCTS simulations per move:", simulations_num)
    print("- Temperature:", temperature)
    first_player_str = input("Choose the first player ('Human'='h', 'AI'='a', or any other key for random): ").lower()
    is_in_human_turn: bool
    if first_player_str == "h":
        is_in_human_turn = True
    elif first_player_str == "a":
        is_in_human_turn = False
    else:
        is_in_human_turn = random.choice([False, True])
    mnk_game = MnkGame(m, n, k)
    all_actions = mnk_game.list_all_actions()
    state = mnk_game.begin()
    reward: Optional[float] = None
    while True:
        current_color = (
            StoneColor.RED
            if state.last_action is None
            or state.board[state.last_action.y][state.last_action.x].color == StoneColor.GREEN
            else StoneColor.GREEN
        )
        print("==================================================")
        print(_make_color(str(state)))
        print(
            f"{'Human' if is_in_human_turn else 'AI'}'s turn. Current color:",
            _make_color(current_color.get_mark()),
        )
        legal_actions = mnk_game.list_legal_actions(state)
        action: MnkAction
        if is_in_human_turn:
            is_quitting = False
            while True:
                action_str = input("Choose a legal action in the format 'x y', or 'q' to quit: ")
                action_str = action_str.strip()
                if action_str == "q":
                    is_quitting = True
                    break
                action = _parse_action(action_str)
                if action is None or action not in legal_actions:
                    continue
                break
            if is_quitting:
                print("Human resigned. Woo-hoo!")
                break
        else:
            legal_probs: dict[MnkAction, float]
            value: float
            if simulations_num == 0:
                prior_probs, value = best_model.predict_single(state.make_input_data())
                legal_probs = {a: p for a, p in prior_probs.items() if a in legal_actions}
            else:
                c_puct = 2.0
                legal_probs, value = execute_mcts_fresh(state, mnk_game, best_model, simulations_num, c_puct)
            all_probs = [legal_probs[a] if a in legal_probs else 0.0 for a in all_actions]  # Unscaled
            all_probs = _ensemble_augmented_predictions(state, all_probs)
            print("Unscaled Probabilities (%):")
            print(_make_color(format_board(state, all_probs)))
            print("Value:", f"{value:.2f}")
            scaled_all_probs = [math.pow(p, 1 / temperature) for p in all_probs]
            scaled_all_probs_sum = sum(scaled_all_probs)
            all_search_probs = [p / scaled_all_probs_sum for p in scaled_all_probs]
            action = random.choices(all_actions, weights=all_search_probs)[0]  # Next action
        state = mnk_game.simulate(state, action)
        reward = mnk_game.receive_reward_if_terminal(state)
        if reward is not None:
            break
        is_in_human_turn = not is_in_human_turn
    if reward is not None:
        print("==================================================")
        print(_make_color(str(state)))
        if reward == 0.0:
            print("Draw!")
        else:
            print("Human" if is_in_human_turn == (reward > 0) else "AI", "won!")


if __name__ == "__main__":
    main()
