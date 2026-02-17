## 1. How to run
Install packages and dependencies:
```bash
uv sync
```

Run the tests:
```bash
uv run pytest
```

Train the model:
```bash
source .venv/bin/activate
cd ../storage/
nohup uv run python ../minimal-alpha-zero/main.py train &
```

Play a game with the trained model:
```bash
uv run python ../minimal-alpha-zero/main.py play --checkpoint-dir ${CHECKPOINT_DIR}
```
- `--checkpoint-dir`: the path to a checkpoint directory (e.g., `../storage/${RUN_TIME}/iteration_${INDEX}/state/`). Defaults to [`../backup/best_model/state/`](../backup/best_model/state/).
