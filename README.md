# minimal-alpha-zero
Minimal implementation of the AlphaZero algorithm for easy board games

## 1. Local development
### 1-1. Set up the environment
Change to [`./infra-local/`](./infra-local/) directory:
```bash
cd ./infra-local/
```

Start the containers:
```bash
docker compose up --build --remove-orphans -d
```

### 1-2. Run the program
Get inside the container:
```bash
docker compose exec minimal-alpha-zero bash
```

Then follow [`./minimal-alpha-zero/README.md`](./minimal-alpha-zero/README.md) for further instructions.

## 2. References
I want to express my sincere thanks to:
- [A Simple Alpha(Go) Zero Tutorial](https://suragnair.github.io/posts/alphazero.html)
- [Mastering the game of Go without human knowledge](https://www.semanticscholar.org/paper/Mastering-the-game-of-Go-without-human-knowledge-Silver-Schrittwieser/c27db32efa8137cbf654902f8f728f338e55cd1c)
- [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404)
- And many more...
