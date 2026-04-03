"""
Predefined Gymnasium environment for tabular RL (Option 1).

Default: FrozenLake-v1 — small discrete state/action space, fits DP / MC / TD.
Swap `ENV_ID` or pass kwargs to `make_env` if you switch environments later.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import Env

# Toy Text env recommended in the assignment; easy to change in one place.
ENV_ID = "FrozenLake-v1"


def make_env(
    env_id: str = ENV_ID,
    *,
    seed: int | None = None,
    render_mode: str | None = None,
    **kwargs: Any,
) -> Env:
    """Create a Gymnasium environment. Extra kwargs are passed to `gym.make`."""
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env


def tabular_sizes(env: Env) -> tuple[int, int]:
    """Return (n_states, n_actions) for fully discrete observation and action spaces."""
    obs = env.observation_space
    act = env.action_space
    if not hasattr(obs, "n") or not hasattr(act, "n"):
        raise TypeError(
            "tabular_sizes expects Discrete observation and action spaces; "
            f"got obs={type(obs).__name__}, act={type(act).__name__}"
        )
    return int(obs.n), int(act.n)


def describe_env(env: Env) -> str:
    """One-line summary for logging / notebook."""
    eid = getattr(env.spec, "id", "?") if env.spec else "?"
    n_s, n_a = tabular_sizes(env)
    return f"{eid} | states={n_s}, actions={n_a}"
