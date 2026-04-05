"""
Shared tabular representations for state–action values Q(s, a).

Used by DP (policy / value iteration), MC, and TD. Dynamic programming here is
expressed in Q-space: iterative policy evaluation applies the Bellman *expectation*
backup for Q^π under a fixed policy π and a known transition model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# Stochastic tabular policy π(a|s): rows sum to 1, shape (n_states, n_actions).
PolicyMatrix = NDArray[np.float64]


def validate_policy_matrix(policy: PolicyMatrix, atol: float = 1e-6) -> None:
    """
    Validates that a policy matrix has shape (n_states, n_actions) and each row sums to 1.
    Raises AssertionError if any row does not sum to 1 within a specified tolerance.
    """
    if policy.ndim != 2:
        raise ValueError("PolicyMatrix must be 2D (n_states, n_actions)")
    row_sums = policy.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise AssertionError(
            f"Each policy row must sum to 1 (±{atol}). Row sums: {row_sums}"
        )

@dataclass(frozen=True)
class QTable:
    """Dense Q(s, a) table; `values` has shape (n_states, n_actions)."""

    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("values must be 2D (n_states, n_actions)")

    @property
    def n_states(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.values.shape[1])

    @classmethod
    def zeros(cls, n_states: int, n_actions: int, dtype: np.dtype = np.float64) -> QTable:
        return cls(np.zeros((n_states, n_actions), dtype=dtype))

    def copy(self) -> QTable:
        return QTable(self.values.copy())

    def __getitem__(self, key: tuple[int, int] | int) -> np.ndarray | float:
        return self.values[key]

    def values_for_state(self, s: int) -> Mapping[int, float]:
        """Q(·|s) as an action -> value mapping (read-only view via dict)."""
        row = self.values[s]
        return {a: float(row[a]) for a in range(self.n_actions)}

    def iter_state_action_values(self) -> Iterator[tuple[int, int, float]]:
        for s in range(self.n_states):
            for a in range(self.n_actions):
                yield s, a, float(self.values[s, a])

# Deterministic π(s)∈{0..n_a-1} is often stored as length-n_states int vector;
# convert to one-hot rows (or build PolicyMatrix) before DP helpers that expect π(a|s).

# (probability, next_state, reward, episode_done)
# `episode_done` matches Gymnasium/FrozenLake: if True, no bootstrapping from s'.
TransitionOutcome = Tuple[float, int, float, bool]
TransitionFn = Callable[[int, int], List[TransitionOutcome]]

def transition_fn_from_p_dict(
    p: Mapping[int, Mapping[int, Sequence[TransitionOutcome]]],
) -> TransitionFn:
    """
    Wrap FrozenLake-style dynamics: `env.unwrapped.P` is
    `P[state][action] -> [(prob, next_state, reward, terminated), ...]`.
    """
    return lambda s, a: list(p[s][a])


# Value function V(s): shape (n_states,).
ValueFunction = NDArray[np.float64]


def to_value_function(q: QTable, policy: PolicyMatrix) -> ValueFunction:
    """
    On-policy value from Q and π: V^π(s) = Σ_a π(a|s) Q(s,a).
    """
    return np.sum(policy * q.values, axis=1).astype(np.float64)


def optimal_value_from_q(q: QTable) -> ValueFunction:
    """V*(s) = max_a Q(s,a) (greedy value surface; use after optimal Q)."""
    return np.max(q.values, axis=1).astype(np.float64)