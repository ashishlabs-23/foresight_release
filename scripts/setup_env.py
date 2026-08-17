# -*- coding: utf-8 -*-
"""
scripts/setup_env.py
~~~~~~~~~~~~~~~~~~~~~
Environment validation and setup diagnostic script.

Run this after initial setup to verify everything is correctly configured:
    python scripts/setup_env.py

Checks:
  - Python version (>=3.11)
  - Required packages are importable
  - Environment variables / config loads correctly
  - No circular imports between packages
  - Logging works
"""
from __future__ import annotations

import sys
import importlib
from pathlib import Path

# Ensure the project root is on the path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def check_python_version() -> bool:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 11)
    status = "[OK]" if ok else "[FAIL]"
    print(f"  {status} Python {major}.{minor} (required: 3.11+)")
    return ok


def check_imports() -> bool:
    """Try importing each top-level package to catch circular import issues."""
    packages = [
        "blackjack",
        "blackjack.cards.card",
        "blackjack.cards.deck",
        "blackjack.cards.hand",
        "blackjack.rules.rules",
        "blackjack.strategies.base",
        "blackjack.strategies.basic",
        "blackjack.strategies.random_strategy",
        "blackjack.engine.game",
        "blackjack.simulation.simulator",
        "ml",
        "ml.data.dataset",
        "ml.features.extractor",
        "ml.models.base_model",
        "ml.training.trainer",
        "ml.evaluation.evaluator",
        "backend",
        "backend.app.core.config",
        "backend.app.core.logging",
        "backend.app.schemas.simulation",
        "backend.app.services.simulation_service",
    ]

    all_ok = True
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(f"  [OK] import {pkg}")
        except ImportError as e:
            print(f"  [FAIL] import {pkg}  ->  {e}")
            all_ok = False
    return all_ok


def check_config() -> bool:
    try:
        from backend.app.core.config import get_settings
        settings = get_settings()
        print(f"  [OK] Config loaded: {settings}")
        return True
    except Exception as e:
        print(f"  [FAIL] Config failed: {e}")
        return False


def check_logging() -> bool:
    try:
        from backend.app.core.logging import configure_logging, get_logger
        configure_logging(level="WARNING", format="console")
        logger = get_logger("setup_env")
        logger.debug("debug test")
        print("  [OK] Logging configured successfully")
        return True
    except Exception as e:
        print(f"  [FAIL] Logging failed: {e}")
        return False


def check_simulation() -> bool:
    try:
        from blackjack.simulation.simulator import SimConfig, Simulator
        cfg = SimConfig(num_hands=100, strategy_name="basic", seed=42)
        result = Simulator(cfg).run()
        print(
            f"  [OK] Simulation ran: {result.total_hands} hands | "
            f"house edge: {result.house_edge:.4%} | "
            f"speed: {result.hands_per_second:,.0f} h/s"
        )
        return True
    except Exception as e:
        print(f"  [FAIL] Simulation failed: {e}")
        return False


def check_no_circular_imports() -> bool:
    """Import all packages in one go to surface any circular dependency."""
    try:
        import blackjack  # noqa: F401
        import ml         # noqa: F401
        import backend    # noqa: F401
        print("  [OK] No circular imports detected")
        return True
    except ImportError as e:
        print(f"  [FAIL] Circular import detected: {e}")
        return False


def main() -> int:
    print("\n" + "=" * 60)
    print("  Blackjack AI — Environment Validation")
    print("=" * 60)

    checks = [
        ("Python version", check_python_version),
        ("Package imports", check_imports),
        ("No circular imports", check_no_circular_imports),
        ("Configuration", check_config),
        ("Logging", check_logging),
        ("Simulation smoke test", check_simulation),
    ]

    results = []
    for name, fn in checks:
        print(f"\n>> {name}")
        results.append(fn())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"  [PASS] ALL CHECKS PASSED ({passed}/{total})")
        print("  Ready for development!")
    else:
        failed = total - passed
        print(f"  [FAIL] {failed}/{total} checks FAILED")
        print('  Run: pip install -e ".[dev]" to install missing dependencies.')
    print("=" * 60 + "\n")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
