# minimal-alpha-zero
Minimal implementation of the AlphaZero algorithm for easy board games

## 1. Introduction
This project is intended for educational purposes. While I try my best to keep everything simple and the comments as clear as possible, the code is not optimized, does not focus on performance improvements, and is only suitable for small games. However, it still serves its main goal: to demonstrate that anyone can learn AlphaZero - a masterpiece of humankind that masters ancient board games using modern techniques.

## 2. Local development
### 2-1. Set up the environment
Change to [`./infra-local/`](./infra-local/) directory:
```bash
cd ./infra-local/
```

Start the containers:
```bash
docker compose up --build --remove-orphans -d
```

### 2-2. Run the program
Get inside the container:
```bash
docker compose exec minimal-alpha-zero bash
```

Then follow [`./minimal-alpha-zero/README.md`](./minimal-alpha-zero/README.md) for further instructions.

## 3. References
I want to express my sincere thanks to:
- [A Simple Alpha(Go) Zero Tutorial](https://suragnair.github.io/posts/alphazero.html)
- [Mastering the game of Go without human knowledge](https://www.semanticscholar.org/paper/Mastering-the-game-of-Go-without-human-knowledge-Silver-Schrittwieser/c27db32efa8137cbf654902f8f728f338e55cd1c)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404)
- [AlphaZero.jl](https://jonathan-laurent.github.io/AlphaZero.jl/stable/)
- And many more...
