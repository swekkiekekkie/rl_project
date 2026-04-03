# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reinforcement Learning assignment
#
# For the full instructions, please see the assignment pdf file!
#
# Please remove any **instructions in italic** from this file before submitting.

# %%

import importlib
import numpy as np
import gymnasium as gym

import dp
import env
import mc
import td
importlib.reload(env)
importlib.reload(dp)
importlib.reload(mc)
importlib.reload(td)
from dp import *
from mc import *
from td import *
# Add more imports if needed. Add these into the requirements.txt file.

# %% [markdown]
# # Abstract

# %% [markdown]
# _insert your abstract here_

# %% [markdown]
# # 0. Environment
#
# Declare your environment here. Feel free to use add any code cells.

# %%
# Predefined Gymnasium env (FrozenLake-v1 by default); see env.py to change.
rl_env = env.make_env(seed=42, render_mode="human")
print(env.describe_env(rl_env))

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
# First run the Dynamic Programming algorithms (Policy Iteration and Value Iteration) by calling functions from the separate `dp.py` file, and create plots. Then, fill in the cell completing your 'report' within this file.

# %%
# Call Policy Iteration algorithm

# %%
# Call Value Iteration algorithm

# %% [markdown]
# _In this cell, describe how the algorithms work, how the algorithms differ, plot results and/or policies (add more code cells!), etc._

# %% [markdown]
# # 3. Monte Carlo algorithms
#
# First run the Monte Carlo algorithm (Monte Carlo Exploring Starts or Monte Carlo without Exploring Starts (with $\epsilon$-greedy strategy)) by calling functions from the separate `mc.py` file, and create plots. Then, fill in the cell completing your 'report' within this file.

# %%
# Call Monte Carlo algorithm

# %% [markdown]
# _In this cell, describe how the algorithms work, how the algorithms differ (compared to the one you did not code), plot results and/or policies (add more code cells!), etc._

# %% [markdown]
# # 4. Temporal Difference algorithms
#
# First run the Temporal Difference algorithms (SARSA and Q-learning) by calling functions from the separate `td.py` file, and create plots. Then, fill in the cell completing your 'report' within this file.

# %%
# Call SARSA algorithm

# %%
# Call Q-learning algorithm

# %% [markdown]
# # 5. Comparison and discussion
#
# Compare different algorithms (MC and TD with plots). You don’t need to plot DP alongside MC and TD since DP is not a learning algorithm. However, DP can provide the ground truth for optimal state or action values, which can serve as a reference when evaluating MC and TD. You can choose to plot any of the following: cumulative reward, root mean squared error, sample efficiency, or any other metric you think is a fair comparison.
#
# Include a discussion: what can you conclude by comparing different RL algorithms? Do they have certain strengths or limitations?

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
