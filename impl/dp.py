"""Tabular Dynamic Programming (Policy Iteration, Value Iteration, etc.)."""

from __future__ import annotations

import numpy as np

from utils import PolicyMatrix, QTable, TransitionFn, validate_policy_matrix


# Bellman expectation → iterative_policy_evaluation; greedy policy_improvement.
# Bellman optimality → value_iteration; policy_iteration alternates eval + improvement.

def bellman_expectation_operator(
    q: QTable,
    policy: PolicyMatrix,
    gamma: float,
    transitions: TransitionFn,
) -> QTable:
    """
    One synchronous Bellman expectation backup for Q^π.

    For each (s, a):

        Q(s,a) ← Σ_{s',r} p(s',r|s,a) [ r + γ Σ_{a'} π(a'|s') Q(s',a') ]

    If the transition marks terminal `episode_done`, the bootstrap term is 0.

    Parameters
    ----------
    policy :
        Stochastic policy π(a|s), shape (n_states, n_actions); each row sums to 1.
    transitions :
        Callable (s, a) -> list of (prob, s_next, reward, episode_done).
        FrozenLake's `env.unwrapped.P[s][a]` matches this shape.
    """

    # During development:
    validate_policy_matrix(policy)

    n_s, n_a = q.n_states, q.n_actions
    q_old = q.values
    q_new = np.empty_like(q_old, dtype=np.float64)

    for s in range(n_s):
        for a in range(n_a):
            acc = 0.0
            for prob, s_next, reward, done in transitions(s, a):
                if done:
                    boot = 0.0
                else:
                    boot = gamma * float(np.dot(policy[s_next], q_old[s_next]))
                acc += prob * (reward + boot)
            q_new[s, a] = acc

    return QTable(q_new)

def bellman_optimality_operator(
    q: QTable,
    transitions: TransitionFn,
    gamma: float,
) -> QTable:
    """
    One synchronous Bellman optimality backup for Q*.

    For each (s, a):

        Q_new(s,a) ← Σ_{s',r} p(s',r|s,a) [ r + γ max_{a'} Q(s',a') ]

    If the transition marks terminal `episode_done`, the max over next actions is omitted
    (bootstrap term 0).
    """
    n_s, n_a = q.n_states, q.n_actions
    q_old = q.values
    q_new = np.empty_like(q_old, dtype=np.float64)
    for s in range(n_s):
        for a in range(n_a):
            acc = 0.0
            for prob, s_next, reward, done in transitions(s, a):
                if done:
                    boot = 0.0
                else:
                    boot = gamma * float(np.max(q_old[s_next]))
                acc += prob * (reward + boot)
            q_new[s, a] = acc
    return QTable(q_new)

def iterative_policy_evaluation(
    policy: PolicyMatrix,
    transitions: TransitionFn,
    n_states: int,
    n_actions: int,
    gamma: float,
    *,
    q_init: QTable | None = None,
    max_delta: float | None = 1e-6,
    max_iterations: int | None = 10_000,
) -> tuple[QTable, dict[str, float | int]]:
    """
    Repeatedly apply `bellman_expectation_operator` until convergence or iteration cap.

    Stops when max_s,a |Q_{k+1}(s,a) - Q_k(s,a)| ≤ max_delta (if max_delta is set),
    or after max_iterations sweeps (if set), whichever comes first.

    At least one of `max_delta` or `max_iterations` must be finite: if `max_delta`
    is None, runs exactly `max_iterations` steps; if `max_iterations` is None,
    only the delta criterion is used (with a hard safety cap of 10**6 sweeps).
    """

    # During development:
    validate_policy_matrix(policy)

    if max_delta is None and max_iterations is None:
        raise ValueError("provide max_delta and/or max_iterations")

    q = q_init if q_init is not None else QTable.zeros(n_states, n_actions)
    if q.n_states != n_states or q.n_actions != n_actions:
        raise ValueError("q_init dimensions do not match n_states / n_actions")

    safety_cap = 1_000_000
    iterations_run = 0
    final_delta = float("inf")

    while True:
        iterations_run += 1
        q_next = bellman_expectation_operator(q, policy, gamma, transitions)
        final_delta = float(np.max(np.abs(q_next.values - q.values)))
        q = q_next

        if max_delta is not None and final_delta <= max_delta:
            break
        if max_iterations is not None and iterations_run >= max_iterations:
            break
        if max_iterations is None and iterations_run >= safety_cap:
            break

    meta: dict[str, float | int] = {
        "iterations": iterations_run,
        "delta": final_delta,
    }
    return q, meta


def greedy_policy_wrt(q: QTable) -> PolicyMatrix:
    """
    Greedy deterministic policy: π(s) puts all mass on argmax_a Q(s,a).
    Ties break toward the smallest action index (NumPy `argmax` default).
    """
    n_s, n_a = q.n_states, q.n_actions
    policy = np.zeros((n_s, n_a), dtype=np.float64)
    for s in range(n_s):
        a_star = int(np.argmax(q.values[s]))
        policy[s, a_star] = 1.0
    return policy


def policies_equal(pi: PolicyMatrix, pi2: PolicyMatrix) -> bool:
    """Exact equality (suitable for tie-broken greedy one-hot policies)."""
    return bool(np.array_equal(pi, pi2))


def policy_iteration(
    transitions: TransitionFn,
    n_states: int,
    n_actions: int,
    gamma: float,
    *,
    policy_init: PolicyMatrix | None = None,
    max_outer_iterations: int = 1000,
    eval_max_delta: float | None = 1e-6,
    eval_max_iterations: int | None = 10_000,
) -> tuple[PolicyMatrix, QTable, dict[str, float | int | bool]]:
    """
    Policy iteration: alternate (1) evaluation of Q^π and (2) greedy improvement
    until the policy stops changing.
    """
    policy = (
        policy_init
        if policy_init is not None
        else np.full((n_states, n_actions), 1.0 / n_actions, dtype=np.float64)
    )
    validate_policy_matrix(policy)

    q = QTable.zeros(n_states, n_actions)
    meta_eval: dict[str, float | int] = {}

    for outer in range(max_outer_iterations):
        q, meta_eval = iterative_policy_evaluation(
            policy,
            transitions,
            n_states,
            n_actions,
            gamma,
            max_delta=eval_max_delta,
            max_iterations=eval_max_iterations,
        )
        policy_new = greedy_policy_wrt(q)
        if policies_equal(policy_new, policy):
            return policy_new, q, {
                "outer_iterations": outer + 1,
                "policy_stable": True,
                "eval_iterations": meta_eval["iterations"],
                "eval_delta": meta_eval["delta"],
            }
        policy = policy_new

    return policy, q, {
        "outer_iterations": max_outer_iterations,
        "policy_stable": False,
        "eval_iterations": meta_eval.get("iterations", -1),
        "eval_delta": meta_eval.get("delta", float("nan")),
    }


def value_iteration(
    transitions: TransitionFn,
    n_states: int,
    n_actions: int,
    gamma: float,
    *,
    q_init: QTable | None = None,
    max_delta: float | None = 1e-6,
    max_iterations: int | None = 10_000,
) -> tuple[QTable, PolicyMatrix, dict[str, float | int]]:
    """
    Value iteration in Q-space: repeated Bellman optimality backups until convergence,
    then a greedy policy w.r.t. the final Q.
    """
    if max_delta is None and max_iterations is None:
        raise ValueError("provide max_delta and/or max_iterations")

    q = q_init if q_init is not None else QTable.zeros(n_states, n_actions)
    if q.n_states != n_states or q.n_actions != n_actions:
        raise ValueError("q_init dimensions do not match n_states / n_actions")

    safety_cap = 1_000_000
    iterations_run = 0
    final_delta = float("inf")

    while True:
        iterations_run += 1
        q_next = bellman_optimality_operator(q, transitions, gamma)
        final_delta = float(np.max(np.abs(q_next.values - q.values)))
        q = q_next

        if max_delta is not None and final_delta <= max_delta:
            break
        if max_iterations is not None and iterations_run >= max_iterations:
            break
        if max_iterations is None and iterations_run >= safety_cap:
            break

    policy = greedy_policy_wrt(q)
    meta: dict[str, float | int] = {
        "iterations": iterations_run,
        "delta": final_delta,
    }
    return q, policy, meta