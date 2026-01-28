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
nohup uv run python ../minimal-alpha-zero/main.py &
```
