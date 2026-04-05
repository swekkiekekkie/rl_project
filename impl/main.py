# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reinforcement Learning assignment
#
# Full course instructions: see `project_description.md` in the repo root (and the Brightspace assignment PDF).
#
# Please remove any **instructions in italic** from this file before submitting.

# %%

import importlib
import numpy as np
import gymnasium as gym

import dp
import dp_experiments
import env
import mc
import td
importlib.reload(env)
importlib.reload(dp)
importlib.reload(dp_experiments)
importlib.reload(mc)
importlib.reload(td)
from dp import *
from mc import *
from td import *
# Add more imports if needed. Add these into the requirements.txt file.

# %% [markdown]
# ## Deliverables checklist (course project)
#
# - **Deadline:** April 5th, 23:59 (see project description).
# - **Code:** three Python modules — `dp.py` (Dynamic Programming), `mc.py` (Monte Carlo), `td.py` (Temporal Difference); implement algorithms there, call them from this notebook.
# - **Dependencies:** maintain `requirements.txt` (do not hand in a virtual environment).
# - **Notebook:** this file (Jupytext) / paired `.ipynb` — environment setup, algorithm runs, plots, and report text (~1000–1500 words).

# %%
# Shared hyperparameters (tune for your environment; sweep γ, θ for DP; ε / schedules for MC & TD).
RNG_SEED = 42
np.random.seed(RNG_SEED)

GAMMA = 0.99  # discount factor γ
THETA = 1e-6  # convergence threshold θ (policy / value iteration)
EPSILON = 0.1  # ε-greedy exploration (MC / TD); consider decaying ε over episodes for bonus

N_EPISODES_MC = 10_000
N_EPISODES_TD = 10_000
MAX_STEPS_PER_EPISODE = 500

# %% [markdown]
# # Abstract

# %% [markdown]
# _insert your abstract here_

# %% [markdown]
# # 0. Environment
#
# Use a **tabular** (finite discrete state and action) **episodic** environment so that DP, MC, and TD from the lectures apply. The assignment suggests toy Gymnasium envs (e.g. FrozenLake, CliffWalking, Taxi) or a **custom** `gym.Env` with `spaces.Discrete` for both spaces; keep \( |S| \cdot |A| \) modest (guideline in the project description: \( |S| \cdot |A| < 600 \)).
#
# Implement or choose an environment that satisfies the MDP assumptions (Markov transitions, well-defined rewards, terminal state(s)). If you use a readymade Gymnasium environment, note that **environment originality points** in the rubric apply to custom envs; algorithms are still implemented in `dp.py` / `mc.py` / `td.py`.

# %%
# Predefined Gymnasium env (FrozenLake-v1 by default); see env.py to change ENV_ID or pass kwargs.
# Use render_mode="human" for visual demos; None avoids opening a window (better for long runs).
rl_env = env.make_env(seed=RNG_SEED, render_mode=None)
print(env.describe_env(rl_env))
n_states, n_actions = env.tabular_sizes(rl_env)
print(f"tabular: |S|={n_states}, |A|={n_actions}")

# %% [markdown]
# ## 0.1 Random baseline (optional sanity check)
#
# The project asks for a **random agent** alongside DP, MC, and TD. This loop samples actions uniformly — useful to verify rewards, episode length, and that `reset` / `step` behave as expected before relying on learning code.

# %%
def run_random_episode(env: gym.Env, *, max_steps: int = MAX_STEPS_PER_EPISODE, seed: int | None = None):
    """One episode with uniformly random actions; returns total reward and length."""
    obs, _ = env.reset(seed=seed)
    total_r = 0.0
    for t in range(max_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        total_r += float(r)
        if term or trunc:
            break
    return total_r, t + 1


total_reward, n_steps = run_random_episode(rl_env, seed=RNG_SEED)
print(f"random baseline: return={total_reward:.4g}, steps={n_steps}")

# %% [markdown]
# # 1. Introduction

# %% [markdown]
# _insert your introduction in this cell_
#
# _describe your environment and the problem the agent has to solve_
#
# _describe the objective of the report (e.g. comparing various RL algorithms) and how you are going to accomplish this (research question)_
#
# _don't forget to add plots/images of the environment, can be done via code cells, but also by inserting .png files into the jupyter notebook_

# %% [markdown]
# # 2. Dynamic Programming algorithms
#
# Core solvers live in **`dp.py`**; **`dp_experiments.py`** builds the §3.2 deliverable: traces for plots, FrozenLake policy maps, value curves, rollouts, and optional figure export. Algorithms use **`env.unwrapped.P`** (FrozenLake toy text) as the tabular model.
#
# - **Policy iteration:** iterative policy evaluation (Bellman expectation for \(Q^\pi\)) + greedy improvement until stable.
# - **Value iteration:** Bellman optimality backups on \(Q\) until \(\Delta < \theta\), then greedy policy.
#
# Experiment with **γ** and **θ**; plots below cover final policies, \(V(s)\) bars, value evolution over iterations, and cumulative discounted return in one rollout.

# %% [markdown]
# ## 2.1 Run PI / VI, traces, and figures (FrozenLake)
#
# Uses `dp_experiments.run_frozenlake_dp_suite` — set `save_dir` to write PNGs (e.g. for the report).

# %%
import matplotlib.pyplot as plt

dp_results = dp_experiments.run_frozenlake_dp_suite(
    rl_env,
    gamma=GAMMA,
    theta=THETA,
    track_states=[0, 5, 10, 14],
    save_dir=None,  # e.g. "figures_dp" to save PNGs next to the notebook
)
print("PI meta:", dp_results["meta_pi"])
print("VI meta:", dp_results["meta_vi"])
print("PI vs VI greedy policies match:", dp_results["policies_match"])
print("Example rollout G from s=0:", dp_results["rollout_G"])

# %%
# Re-display key plots in the notebook (suite also closes figures when save_dir is set)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
dp_experiments.plot_frozenlake_policy(
    rl_env.unwrapped.desc, dp_results["policy_pi"], title="Policy iteration", ax=axes[0]
)
dp_experiments.plot_frozenlake_policy(
    rl_env.unwrapped.desc, dp_results["policy_vi"], title="Value iteration", ax=axes[1]
)
plt.tight_layout()
plt.show()

# %%
fig2, axes2 = plt.subplots(1, 2, figsize=(9, 3))
from utils import optimal_value_from_q

dp_experiments.plot_state_values_bars(
    optimal_value_from_q(dp_results["q_pi"]), title="V*(s) from PI final Q", ax=axes2[0]
)
dp_experiments.plot_state_values_bars(
    optimal_value_from_q(dp_results["q_vi"]), title="V*(s) from VI final Q", ax=axes2[1]
)
plt.tight_layout()
plt.show()

# %%
fig3, axes3 = plt.subplots(1, 2, figsize=(9, 4))
dp_experiments.plot_v_evolution_vi(dp_results["hist_vi"], ax=axes3[0])
dp_experiments.plot_v_evolution_pi(dp_results["hist_pi"], ax=axes3[1])
plt.tight_layout()
plt.show()

# %%
fig4, ax4 = plt.subplots(figsize=(6, 3))
dp_experiments.plot_rollout_returns(dp_results["rollout_rewards"], GAMMA, ax=ax4)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2.2 θ sweep (value iteration iteration counts)
#
# See how the stopping threshold affects the number of VI sweeps (fixed γ).

# %%
for row in dp_experiments.sweep_theta_value_iteration(
    dp_experiments.transition_fn_from_env(rl_env),
    n_states,
    n_actions,
    gamma=GAMMA,
    thetas=(1e-4, 1e-6, 1e-8),
):
    print(row)

# %% [markdown]
# ## 2.3 Iterative policy evaluation (uniform policy) — optional
#
# Ground-truth \(V^\pi(s)\) for a **fixed** random policy (here uniform random).

# %%
uniform = np.full((n_states, n_actions), 1.0 / n_actions)
q_uni, meta_uni, hist_uni = dp_experiments.iterative_policy_evaluation_trace(
    uniform,
    dp_experiments.transition_fn_from_env(rl_env),
    n_states,
    n_actions,
    GAMMA,
    max_delta=THETA,
    track_states=[0, 5, 10, 14],
)
print("IPE (uniform) meta:", meta_uni)

# %%
fig5, ax5 = plt.subplots(figsize=(6, 4))
dp_experiments.plot_v_evolution_ipe(hist_uni, ax=ax5)
plt.tight_layout()
plt.show()

# %% [markdown]
# _Report: relate policy evaluation, improvement, PI, and VI; discuss γ, θ, and the figures above._

# %% [markdown]
# # 3. Monte Carlo algorithms
#
# Implement in **`mc.py`**. Choose **one** control setting (see textbook / lecture):
#
# - **Monte Carlo with exploring starts (ES):** first-visit MC for **action values** \(q(s,a)\); control via greedy improvement from exploring starts.
# - **MC without exploring starts:** first-visit MC prediction; control with **ε-greedy** exploration.
#
# Find hyperparameters and plot learning curves / policies as in the assignment.

# %% [markdown]
# ## 3.1 Monte Carlo prediction (action-value estimation)
#
# Estimate \(q_\pi\) or \(q_*\) from sampled episodes (first-visit).

# %%
# Q_mc = mc.run_mc_prediction(rl_env, gamma=GAMMA, n_episodes=N_EPISODES_MC, epsilon=EPSILON)

# %% [markdown]
# ## 3.2 Monte Carlo control
#
# Improve the policy from estimated **Q** (greedy or ε-greedy depending on your chosen method).

# %%
# policy_mc, Q_mc = mc.run_mc_control(rl_env, gamma=GAMMA, n_episodes=N_EPISODES_MC, epsilon=EPSILON)

# %% [markdown]
# _Report: describe your chosen MC variant (ES vs non-ES), how it differs from the other, and show plots / policies._

# %% [markdown]
# # 4. Temporal Difference algorithms
#
# Implement in **`td.py`**. Required ingredients:
#
# - **TD(0) prediction:** bootstrapped update of **action values** \(Q(s,a)\) under the current policy (on-policy).
# - **SARSA (on-policy control):** TD control with **ε-greedy** behavior.
# - **Q-learning (off-policy control):** max over next actions in the TD target; behavior still typically ε-greedy.
#
# Optional: **decay ε** over episodes (bonus). TD(0) / SARSA / Q-learning share hyperparameters like `alpha` (step size) — add them when you implement `td.py`.

# %% [markdown]
# ## 4.1 TD(0) prediction
#
# On-policy TD(0) updates to \(Q\) from single-step transitions.

# %%
# Q_td0 = td.run_td0_prediction(rl_env, gamma=GAMMA, alpha=0.1, n_episodes=N_EPISODES_TD, epsilon=EPSILON)

# %% [markdown]
# ## 4.2 SARSA (on-policy)
#
# On-policy TD control: update using the **actual** next action taken.

# %%
# policy_sarsa, Q_sarsa = td.run_sarsa(rl_env, gamma=GAMMA, alpha=0.1, n_episodes=N_EPISODES_TD, epsilon=EPSILON)

# %% [markdown]
# ## 4.3 Q-learning (off-policy)
#
# Off-policy: TD target uses **max** over actions at the next state.

# %%
# policy_ql, Q_ql = td.run_q_learning(rl_env, gamma=GAMMA, alpha=0.1, n_episodes=N_EPISODES_TD, epsilon=EPSILON)

# %% [markdown]
# _Report: describe TD(0), SARSA, and Q-learning; include plots and optional discussion of ε schedules._

# %% [markdown]
# # 5. Comparison and discussion
#
# Compare **MC and TD** with plots (DP is not an online learning algorithm, so you do not need to plot it next to learning curves — but **DP optimal \(V\) or \(Q\)** from Section 2 is ideal **ground truth** for RMSE vs episode).
#
# Possible metrics (choose and justify):
#
# - **Cumulative return** per episode (or smoothed) as learning progresses.
# - **RMSE** of estimated \(Q\) (or \(V\)) vs DP **optimal** values, averaged over states or state–action pairs, vs episode.
# - **Sample efficiency:** episodes (or steps) to reach a performance threshold or near-optimal policy.
# - Any other fair comparison you define.
#
# **Discussion:** strengths and limitations of each family (DP vs sample-based; MC vs TD; on-policy vs off-policy). Add code cells for figures and tables.

# %%
# Example directions (implement helpers in dp/mc/td or here after imports):
# - V_dp, Q_dp from DP for reference
# - rmse_mc = compare_rmse(Q_mc, Q_dp)
# - plot_learning_curves(...)

# %% [markdown]
# _Add as many text cells as you like_

# %%
# Add as many code cells as you like

# %% [markdown]
# # 6. Conclusion
#
# Conclude your project.

# %% [markdown]
# _Add as many text cells as you like_
