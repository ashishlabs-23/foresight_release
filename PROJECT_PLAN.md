# Blackjack AI — Project Plan

## Vision

Build a complete, research-grade Blackjack AI system that progresses from a pure rule-based engine to a trained reinforcement-learning agent, served via a real-time API with a visualization frontend.

---

## Phases

### ✅ Phase 1 — Project Foundation (Current)

**Goal**: Establish a clean, scalable repository that every future phase can build on.

**Deliverables**:
- Modular Python monorepo with clear layer boundaries
- Pure Blackjack game engine (cards, deck, hand, rules, strategies)
- Abstract ML interfaces (no model code yet)
- FastAPI backend scaffolding (health + simulation stub endpoints)
- Pytest test suite (≥20 tests, all passing)
- YAML configuration system with environment overrides
- Structured logging (structlog)
- Docker scaffolding
- Zero hard-coded secrets, zero circular imports

**Definition of Done**:
- `pytest tests/ -v` → all green
- `python scripts/setup_env.py` → prints config summary, no errors
- `uvicorn backend.app.main:app` → server starts, `/health` returns 200

---

### 🔜 Phase 2 — Full Blackjack Engine + Simulation

**Goal**: Complete and battle-test the game engine. Run large-scale simulations.

**Deliverables**:
- Full multi-hand GameEngine with splits, double-down, surrender
- Dealer play-out logic
- Simulation runner: 1M hand runs, configurable strategies
- House edge calculator
- CSV / Parquet output for training data
- Extended test coverage (≥80%)

---

### 🔜 Phase 3 — Reinforcement Learning Agent

**Goal**: Train a neural network agent to play near-optimal Blackjack.

**Deliverables**:
- State representation (player hand, dealer upcard, count)
- Deep Q-Network (DQN) agent
- Training loop with experience replay
- Evaluation against BasicStrategy baseline
- Model serialization (ONNX or PyTorch)
- MLflow experiment tracking
- Comparison: RL agent vs Basic Strategy vs Random

---

### 🔜 Phase 4 — Backend API + Serving

**Goal**: Expose the game engine and ML model through a production-ready REST API.

**Deliverables**:
- FastAPI endpoints: `/simulate`, `/play`, `/model/predict`
- WebSocket support for real-time game sessions
- PostgreSQL database (async SQLAlchemy)
- Alembic migrations
- Redis caching for simulation results
- OpenAPI docs
- Docker Compose with all services
- Basic authentication (API key)

---

### 🔜 Phase 5 — Frontend + Visualization

**Goal**: Interactive UI showing the AI in action.

**Deliverables**:
- React/Next.js frontend
- Interactive Blackjack table (play vs AI)
- Strategy heatmap visualization
- Training curves + win-rate charts
- Real-time WebSocket game session

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Phase 5)                    │
│                    React + WebSocket client                   │
└─────────────────────────┬───────────────────────────────────┘
                          │  HTTP / WebSocket
┌─────────────────────────▼───────────────────────────────────┐
│                     Backend API (Phase 4)                    │
│              FastAPI  ·  PostgreSQL  ·  Redis                │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│    │  /simulate   │  │   /play      │  │ /model/pred  │     │
│    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└───────────┼─────────────────┼─────────────────┼─────────────┘
            │                 │                 │
┌───────────▼─────────────────▼───────┐  ┌──────▼──────────────┐
│        Blackjack Engine              │  │    ML Layer          │
│  cards · deck · hand · rules        │  │  model · features    │
│  engine · simulator · strategies    │  │  trainer · evaluator │
└──────────────────────────────────────┘  └─────────────────────┘
```

---

## Interface Contracts

| Caller | Callee | Interface |
|--------|--------|-----------|
| `backend/services` | `blackjack/simulation` | `Simulator.run(config) → SimResult` |
| `ml/training` | `blackjack/simulation` | Simulator produces `Episode` objects for training |
| `ml/models` | `ml/features` | `FeatureExtractor.extract(state) → np.ndarray` |
| `backend/api` | `backend/services` | Service layer never imports FastAPI directly |
| `scripts/*` | `blackjack/` + `backend/` | Scripts are thin CLI wrappers only |

**Key rules**:
1. `blackjack/` never imports from `ml/` or `backend/`
2. `ml/` never imports from `backend/`
3. `backend/` imports from `blackjack/` and `ml/` only through the service layer
4. No layer imports from `tests/` or `scripts/`

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.11+ | Type hints, match/case, performance |
| Web framework | FastAPI | Async, OpenAPI auto-docs, Pydantic native |
| Data validation | Pydantic v2 | Fast, strict, type-safe |
| Logging | structlog | Structured JSON logs, easy filtering |
| Configuration | pydantic-settings + YAML | Layered env override |
| ML framework | PyTorch (Phase 3) | Research flexibility |
| Testing | pytest + pytest-asyncio | Async-native test support |
| Linting | ruff | Extremely fast |
| Type checking | mypy | Correctness at scale |
| Containerisation | Docker + Compose | Reproducible environments |

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Circular imports as project grows | Medium | Enforced layer boundaries, CI import check |
| Slow simulation at scale | Medium | NumPy vectorisation in Phase 2 |
| RL agent fails to beat BasicStrategy | Low | Ablation study, hyperparameter search |
| Secret leakage | Low | `.env` in .gitignore, env template only |
