"""
Workgroup 3.2: DP experiments, traces for plots, and FrozenLake visualizations.

Uses `dp.py` + a tabular model (`env.unwrapped.P` for FrozenLake). This module is
the blueprint pattern for later MC/TD: algorithms in `*.py`, reporting/plots here
or in the notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env


def _track_states(track_states: Sequence[int] | None, n_states: int) -> list[int]:
    if track_states is not None and len(track_states) > 0:
        return list(track_states)
    return [0, min(5, n_states - 1)]

import dp
from utils import (
    PolicyMatrix,
    QTable,
    TransitionFn,
    ValueFunction,
    optimal_value_from_q,
    to_value_function,
    transition_fn_from_p_dict,
    validate_policy_matrix,
)


def transition_fn_from_env(env: Env) -> TransitionFn:
    """Toy-text style `P` dict (e.g. FrozenLake)."""
    p = getattr(env.unwrapped, "P", None)
    if p is None:
        raise TypeError(
            "Need env.unwrapped.P (FrozenLake-style). For other envs, build a TransitionFn yourself."
        )
    return transition_fn_from_p_dict(p)


def value_iteration_trace(
    transitions: TransitionFn,
    n_states: int,
    n_actions: int,
    gamma: float,
    *,
    max_delta: float = 1e-6,
    max_iterations: int = 10_000,
    track_states: Sequence[int] | None = None,
) -> tuple[QTable, PolicyMatrix, dict[str, float | int], list[dict[str, Any]]]:
    """Like `dp.value_iteration` but records V*(s)=max_a Q(s,a) after each sweep."""
    track = _track_states(track_states, n_states)
    q = QTable.zeros(n_states, n_actions)
    history: list[dict[str, Any]] = []
    safety_cap = 1_000_000
    it = 0
    final_delta = float("inf")

    while True:
        it += 1
        q_next = dp.bellman_optimality_operator(q, transitions, gamma)
        final_delta = float(np.max(np.abs(q_next.values - q.values)))
        q = q_next
        vo = optimal_value_from_q(q)
        history.append(
            {
                "iteration": it,
                "delta": final_delta,
                "v_star": {s: float(vo[s]) for s in track},
            }
        )
        if final_delta <= max_delta:
            break
        if it >= max_iterations:
            break
        if it >= safety_cap:
            break

    policy = dp.greedy_policy_wrt(q)
    meta: dict[str, float | int] = {"iterations": it, "delta": final_delta}
    return q, policy, meta, history


def policy_iteration_trace(
    transitions: TransitionFn,
    n_states: int,
    n_actions: int,
    gamma: float,
    *,
    policy_init: PolicyMatrix | None = None,
    max_outer_iterations: int = 1000,
    eval_max_delta: float | None = 1e-6,
    eval_max_iterations: int | None = 10_000,
    track_states: Sequence[int] | None = None,
) -> tuple[PolicyMatrix, QTable, dict[str, Any], list[dict[str, Any]]]:
    """Records V^π(s)=Σ_a π(a|s)Q(s,a) after each evaluation step (before improvement)."""
    track = _track_states(track_states, n_states)
    policy = (
        policy_init
        if policy_init is not None
        else np.full((n_states, n_actions), 1.0 / n_actions, dtype=np.float64)
    )
    validate_policy_matrix(policy)

    history: list[dict[str, Any]] = []
    q = QTable.zeros(n_states, n_actions)
    meta_eval: dict[str, float | int] = {}

    for outer in range(max_outer_iterations):
        q, meta_eval = dp.iterative_policy_evaluation(
            policy,
            transitions,
            n_states,
            n_actions,
            gamma,
            max_delta=eval_max_delta,
            max_iterations=eval_max_iterations,
        )
        v_pi = to_value_function(q, policy)
        policy_new = dp.greedy_policy_wrt(q)
        history.append(
            {
                "outer": outer + 1,
                "eval_delta": meta_eval["delta"],
                "eval_iterations": meta_eval["iterations"],
                "v_pi": {s: float(v_pi[s]) for s in track},
            }
        )
        if dp.policies_equal(policy_new, policy):
            meta: dict[str, Any] = {
                "outer_iterations": outer + 1,
                "policy_stable": True,
                "eval_iterations": meta_eval["iterations"],
                "eval_delta": meta_eval["delta"],
            }
            return policy_new, q, meta, history
        policy = policy_new

    meta = {
        "outer_iterations": max_outer_iterations,
        "policy_stable": False,
        "eval_iterations": meta_eval.get("iterations", -1),
        "eval_delta": meta_eval.get("delta", float("nan")),
    }
    return policy, q, meta, history


def iterative_policy_evaluation_trace(
    policy: PolicyMatrix,
    transitions: TransitionFn,
    n_states: int,
    n_actions: int,
    gamma: float,
    *,
    max_delta: float | None = 1e-6,
    max_iterations: int | None = 10_000,
    track_states: Sequence[int] | None = None,
) -> tuple[QTable, dict[str, float | int], list[dict[str, Any]]]:
    """IPE with history (e.g. uniform random policy as baseline)."""
    track = _track_states(track_states, n_states)
    validate_policy_matrix(policy)

    if max_delta is None and max_iterations is None:
        raise ValueError("provide max_delta and/or max_iterations")

    q = QTable.zeros(n_states, n_actions)
    history: list[dict[str, Any]] = []
    safety_cap = 1_000_000
    it = 0
    final_delta = float("inf")

    while True:
        it += 1
        q_next = dp.bellman_expectation_operator(q, policy, gamma, transitions)
        final_delta = float(np.max(np.abs(q_next.values - q.values)))
        q = q_next
        v_pi = to_value_function(q, policy)
        history.append(
            {
                "iteration": it,
                "delta": final_delta,
                "v_pi": {s: float(v_pi[s]) for s in track},
            }
        )
        if max_delta is not None and final_delta <= max_delta:
            break
        if max_iterations is not None and it >= max_iterations:
            break
        if max_iterations is None and it >= safety_cap:
            break

    meta = {"iterations": it, "delta": final_delta}
    return q, meta, history


def rollout_under_policy(
    transitions: TransitionFn,
    policy: PolicyMatrix,
    *,
    start_state: int = 0,
    gamma: float = 0.99,
    max_steps: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[list[float], float]:
    """
    Sample one episode from the model following π(a|s). Returns step rewards and
    total discounted return G from the start state.
    """
    rng = rng or np.random.default_rng()
    validate_policy_matrix(policy)
    s = start_state
    rewards: list[float] = []
    g = 0.0
    for t in range(max_steps):
        a = int(rng.choice(policy.shape[1], p=policy[s]))
        outcomes = transitions(s, a)
        probs = np.array([float(o[0]) for o in outcomes], dtype=np.float64)
        probs /= probs.sum()
        idx = int(rng.choice(len(outcomes), p=probs))
        _p, s_next, r, done = outcomes[idx]
        r = float(r)
        rewards.append(r)
        g += (gamma**t) * r
        s = s_next
        if done:
            break
    return rewards, g


# --- FrozenLake plotting (4x4 grid; generalizes if desc matches n_states) ---

ACTION_ARROW = {0: "←", 1: "↓", 2: "→", 3: "↑"}


def _map_cell_char(cell: Any) -> str:
    """FrozenLake `desc` cells are often `|S1` bytes; list-of-rows uses raw bytes."""
    if isinstance(cell, (bytes, np.bytes_)):
        return bytes(cell).decode("ascii")
    return chr(int(cell))


def policy_argmax_per_state(policy: PolicyMatrix) -> np.ndarray:
    return np.argmax(np.asarray(policy), axis=1)


def plot_frozenlake_policy(
    desc: list[bytes] | np.ndarray,
    policy: PolicyMatrix,
    *,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay greedy action arrows on the map (holes/goal: no arrow)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    d = np.asarray(desc)
    nrow, ncol = d.shape
    actions = policy_argmax_per_state(policy)
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    for r in range(nrow):
        for c in range(ncol):
            ch = _map_cell_char(d[r, c])
            s = r * ncol + c
            color = (0.85, 0.9, 1.0)
            if ch == "H":
                color = (0.4, 0.4, 0.45)
            elif ch == "G":
                color = (0.5, 0.85, 0.5)
            elif ch == "S":
                color = (0.95, 0.95, 0.7)
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="black"))
            ax.text(c + 0.5, r + 0.5, ch, ha="center", va="center", fontsize=11, fontweight="bold")
            if ch not in ("H", "G"):
                a = int(actions[s])
                ax.text(
                    c + 0.5,
                    r + 0.78,
                    ACTION_ARROW.get(a, "?"),
                    ha="center",
                    va="center",
                    fontsize=14,
                )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return ax


def plot_state_values_bars(
    v: ValueFunction,
    *,
    title: str = "V(s)",
    ax: plt.Axes | None = None,
    max_states: int = 32,
) -> plt.Axes:
    """Bar chart of V(s); cap `max_states` for large envs."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    n = min(len(v), max_states)
    ax.bar(np.arange(n), v[:n])
    ax.set_xlabel("state index s")
    ax.set_ylabel("value")
    ax.set_title(title)
    return ax


def plot_v_evolution_vi(
    history: list[dict[str, Any]],
    *,
    title: str = "V*(s) vs VI iteration (tracked states)",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """history from `value_iteration_trace`: keys v_star per iteration."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if not history:
        return ax
    states = sorted(history[0]["v_star"].keys())
    xs = [h["iteration"] for h in history]
    for s in states:
        ys = [h["v_star"][s] for h in history]
        ax.plot(xs, ys, marker="o", ms=2, label=f"s={s}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("max_a Q(s,a)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    return ax


def plot_v_evolution_pi(
    history: list[dict[str, Any]],
    *,
    title: str = "V^π(s) vs policy iteration outer step",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if not history:
        return ax
    states = sorted(history[0]["v_pi"].keys())
    xs = [h["outer"] for h in history]
    for s in states:
        ys = [h["v_pi"][s] for h in history]
        ax.plot(xs, ys, marker="o", ms=3, label=f"s={s}")
    ax.set_xlabel("policy iteration (outer)")
    ax.set_ylabel("V^π(s)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    return ax


def plot_v_evolution_ipe(
    history: list[dict[str, Any]],
    *,
    title: str = "V^π(s) vs IPE sweep (fixed π)",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """History from `iterative_policy_evaluation_trace` (keys: iteration, v_pi)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if not history:
        return ax
    states = sorted(history[0]["v_pi"].keys())
    xs = [h["iteration"] for h in history]
    for s in states:
        ys = [h["v_pi"][s] for h in history]
        ax.plot(xs, ys, marker="o", ms=2, label=f"s={s}")
    ax.set_xlabel("IPE iteration")
    ax.set_ylabel("V^π(s)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    return ax


def plot_rollout_returns(
    rewards: list[float],
    gamma: float,
    *,
    title: str = "Discounted return accumulated within one episode",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    if not rewards:
        return ax
    t = np.arange(len(rewards))
    disc_step = np.array([(gamma**i) * rewards[i] for i in range(len(rewards))])
    cum = np.cumsum(disc_step)
    ax.plot(t, cum, marker=".")
    ax.set_xlabel("time step")
    ax.set_ylabel("cumulative discounted return")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def sweep_theta_value_iteration(
    transitions: TransitionFn,
    n_states: int,
    n_actions: int,
    gamma: float,
    thetas: Iterable[float],
) -> list[dict[str, float | int]]:
    """How many VI sweeps until Δ < θ for each θ (same γ)."""
    rows: list[dict[str, float | int]] = []
    for theta in thetas:
        _q, _pi, meta = dp.value_iteration(
            transitions,
            n_states,
            n_actions,
            gamma,
            max_delta=theta,
            max_iterations=100_000,
        )
        rows.append({"theta": theta, "iterations": meta["iterations"], "final_delta": meta["delta"]})
    return rows


def run_frozenlake_dp_suite(
    env: Env,
    *,
    gamma: float = 0.99,
    theta: float = 1e-6,
    track_states: Sequence[int] | None = None,
    rng: np.random.Generator | None = None,
    save_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    End-to-end §3.2 demo: PI + VI, traces, comparison plots, rollout under optimal π.

    If `save_dir` is set, writes PNGs there.
    """
    rng = rng or np.random.default_rng(0)
    n_states, n_actions = int(env.observation_space.n), int(env.action_space.n)
    tr = transition_fn_from_env(env)
    track_states = list(track_states) if track_states is not None else [0, 5, 10, 14]

    policy_pi, q_pi, meta_pi, hist_pi = policy_iteration_trace(
        tr, n_states, n_actions, gamma, eval_max_delta=theta, track_states=track_states
    )
    q_vi, policy_vi, meta_vi, hist_vi = value_iteration_trace(
        tr, n_states, n_actions, gamma, max_delta=theta, track_states=track_states
    )

    same = dp.policies_equal(policy_pi, policy_vi)
    v_pi = optimal_value_from_q(q_pi)
    v_vi = optimal_value_from_q(q_vi)

    rewards, g = rollout_under_policy(tr, policy_vi, start_state=0, gamma=gamma, rng=rng)

    desc = env.unwrapped.desc

    fig_dir = Path(save_dir) if save_dir else None
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_save(name: str) -> None:
        if fig_dir is not None:
            plt.savefig(fig_dir / name, dpi=150, bbox_inches="tight")

    fig1, ax1 = plt.subplots(1, 2, figsize=(8, 4))
    plot_frozenlake_policy(desc, policy_pi, title="Policy iteration (final)", ax=ax1[0])
    plot_frozenlake_policy(desc, policy_vi, title="Value iteration (final)", ax=ax1[1])
    plt.tight_layout()
    _maybe_save("dp_policies_pi_vs_vi.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 3))
    plot_state_values_bars(v_pi, title="V*(s) from policy iteration Q", ax=ax2[0])
    plot_state_values_bars(v_vi, title="V*(s) from value iteration Q", ax=ax2[1])
    plt.tight_layout()
    _maybe_save("dp_state_values_bars.png")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(1, 2, figsize=(9, 4))
    plot_v_evolution_vi(hist_vi, ax=ax3[0])
    plot_v_evolution_pi(hist_pi, ax=ax3[1])
    plt.tight_layout()
    _maybe_save("dp_value_evolution.png")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(6, 3))
    plot_rollout_returns(rewards, gamma, ax=ax4)
    plt.tight_layout()
    _maybe_save("dp_rollout_cumulative_discounted.png")
    plt.close(fig4)

    return {
        "policy_pi": policy_pi,
        "policy_vi": policy_vi,
        "q_pi": q_pi,
        "q_vi": q_vi,
        "meta_pi": meta_pi,
        "meta_vi": meta_vi,
        "policies_match": same,
        "v_pi_star": v_pi,
        "v_vi_star": v_vi,
        "rollout_rewards": rewards,
        "rollout_G": g,
        "hist_pi": hist_pi,
        "hist_vi": hist_vi,
        "figures_dir": str(fig_dir) if fig_dir else None,
    }
