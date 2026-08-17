# Blackjack AI 🃏

> A modular, research-grade Blackjack AI system built with Python — from a clean game engine to reinforcement-learning agents.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**Blackjack AI** is a multi-phase project that builds a complete pipeline for training, evaluating, and serving an AI agent that plays Blackjack optimally.

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Project foundation — architecture, engine, scaffolding | ✅ Complete |
| 2 | Blackjack engine completion + full simulation | 🔜 Planned |
| 3 | ML model — reinforcement learning agent | 🔜 Planned |
| 4 | Backend API + serving layer | 🔜 Planned |
| 5 | Frontend UI + visualization | 🔜 Planned |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Git

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd blackjack-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install with dev dependencies
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work for development)
```

### 3. Validate setup

```bash
python scripts/setup_env.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Start the API

```bash
uvicorn backend.app.main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 6. Run a simulation

```bash
python scripts/run_simulation.py --hands 1000 --strategy basic
```

---

## Repository Structure

```
blackjack-ai/
│
├── blackjack/              # Pure game engine (no HTTP, no ML)
│   ├── cards/              # Card, Deck, Shoe, Hand
│   ├── rules/              # Rule variants (Vegas Strip, Downtown…)
│   ├── engine/             # GameEngine orchestrator
│   ├── simulation/         # Batch simulator → SimResult
│   └── strategies/         # Abstract Strategy + BasicStrategy + Random
│
├── ml/                     # ML layer (abstract interfaces in Phase 1)
│   ├── data/               # DataLoader interface
│   ├── features/           # FeatureExtractor interface
│   ├── models/             # BaseModel interface
│   ├── training/           # Trainer interface
│   └── evaluation/         # Evaluator interface
│
├── backend/                # FastAPI application
│   └── app/
│       ├── api/v1/         # Route handlers
│       ├── core/           # Config + logging
│       ├── schemas/        # Pydantic models
│       └── services/       # Business logic (wraps blackjack engine)
│
├── tests/
│   ├── unit/               # Fast, isolated tests
│   └── integration/        # Multi-component tests
│
├── scripts/                # CLI entry-points
├── configs/                # YAML configuration files
├── docs/                   # Architecture documentation
├── docker/                 # Dockerfile + docker-compose
├── data/                   # raw/ + processed/ (gitignored content)
├── artifacts/              # Model checkpoints, evaluation outputs
├── logs/                   # Runtime logs (gitignored)
└── frontend/               # UI (Phase 5)
```

---

## Architecture

Each layer has a **single responsibility** and communicates through well-defined interfaces:

```
frontend  →  backend/api  →  backend/services  →  blackjack/simulation
                                                          ↕
                                                   blackjack/engine
                                                          ↕
                                               blackjack/strategies
                                     ml/training  →  blackjack/simulation
```

See [`docs/architecture.md`](docs/architecture.md) for full diagrams.

---

## Development

### Running Tests

```bash
pytest tests/                          # all tests
pytest tests/unit/ -v                  # unit tests only
pytest tests/ --cov=blackjack          # with coverage
```

### Linting & Formatting

```bash
ruff check .                           # lint
ruff format .                          # format
mypy blackjack/ ml/ backend/           # type check
```

---

## Contributing

1. Branch from `main`
2. Write tests first
3. Ensure `pytest` and `ruff check .` both pass
4. Open a PR with a clear description

---

## License

MIT — see [LICENSE](LICENSE).
