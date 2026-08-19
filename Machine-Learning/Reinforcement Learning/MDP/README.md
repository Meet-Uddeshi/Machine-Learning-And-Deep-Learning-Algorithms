# Reinforcement Learning - Markov Decision Process (MDP)

> Reinforcement Learning | Dynamic Programming & Optimal Control Framework

---

## Table of Contents

1. [What is a Markov Decision Process (MDP)?](#1-what-is-a-markov-decision-process-mdp)
2. [Theoretical Explanation](#2-theoretical-explanation)
3. [Mathematical Operations](#3-mathematical-operations)
4. [Real-World Example](#4-real-world-example)
5. [Worked MDP Sum (Step-by-Step)](#5-worked-mdp-sum-step-by-step)
6. [Program Flowchart](#6-program-flowchart)
7. [Module Responsibility Map](#7-module-responsibility-map)
8. [Configuration](#8-configuration)

---

## 1. What is a Markov Decision Process (MDP)?

A **Markov Decision Process (MDP)** provides a mathematical framework for modeling sequential decision-making under uncertainty, where outcomes are partly random and partly under the control of a decision-making agent.

In Reinforcement Learning (RL), an MDP models the environment. When the environment dynamics (transition probabilities and rewards) are fully known, **Dynamic Programming (DP)** algorithms—such as **Value Iteration** and **Policy Iteration**—compute the exact optimal state values $V^*(s)$ and optimal policy $\pi^*(s)$.

### Key Characteristics

| Property           | Description                                                          |
|--------------------|----------------------------------------------------------------------|
| Task type          | Reinforcement Learning / Model-Based Dynamic Programming             |
| Core Formalism     | 5-tuple $(S, A, P, R, \gamma)$                                       |
| Markov Property    | Future state depends ONLY on current state and action                |
| Primary Solvers    | Value Iteration (Bellman Optimality) & Policy Iteration              |
| Key Outputs        | Optimal State Values $V^*(s)$, Optimal Policy Mapping $\pi^*(s)$      |

---

## 2. Theoretical Explanation

### 1. The Markov Property
A stochastic process satisfies the **Markov Property** if the conditional probability distribution of future states depends solely upon the present state and action, independent of the past history of states and actions:

$$\mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a, S_{t-1} = s_{t-1}, A_{t-1} = a_{t-1}, \dots) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$$

---

### 2. Components of an MDP $(S, A, P, R, \gamma)$

1. **State Space ($S$)**: Set of all valid environment states.
2. **Action Space ($A$)**: Set of all valid agent decisions/moves.
3. **Transition Probability Model ($P$)**:
   $$P(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$$
4. **Reward Function ($R$)**:
   $$R(s, a, s') = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s']$$
5. **Discount Factor ($\gamma \in [0, 1)$)**: Determines the present value of future rewards. Ensures convergence of infinite-horizon returns.

---

### 3. Dynamic Programming Algorithms

#### A. Value Iteration
Value Iteration iteratively applies the **Bellman Optimality Operator** directly to update state values until convergence:

$$V_{k+1}(s) = \max_{a \in A} \sum_{s' \in S} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V_k(s') \right]$$

Convergence is guaranteed when the maximum Bellman residual error drops below threshold $\theta$:

$$\max_{s \in S} |V_{k+1}(s) - V_k(s)| < \theta$$

#### B. Policy Iteration
Policy Iteration alternates between two distinct phases:
1. **Policy Evaluation**: Iteratively evaluate $V^\pi(s)$ for fixed policy $\pi$:
   $$V_{k+1}^\pi(s) = \sum_{s' \in S} P(s' \mid s, \pi(s)) \left[ R(s, \pi(s), s') + \gamma V_k^\pi(s') \right]$$
2. **Policy Improvement**: Update policy greedily with respect to $V^\pi(s)$:
   $$\pi_{new}(s) = \arg\max_{a \in A} \sum_{s' \in S} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$
Repeat evaluation and improvement until policy $\pi(s)$ stabilizes ($\pi_{new} = \pi_{old}$).

---

## 3. Mathematical Operations

### 1. Expected Return $G_t$
The discounted cumulative return from step $t$ onwards is defined as:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

---

### 2. State-Value Function $V^\pi(s)$
Expected return starting from state $s$ following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$

**Bellman Expectation Equation**:

$$V^\pi(s) = \sum_{a \in A} \pi(a \mid s) \sum_{s' \in S} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

---

### 3. Action-Value Function $Q^\pi(s, a)$
Expected return starting from state $s$, taking action $a$, and thereafter following policy $\pi$:

$$Q^\pi(s, a) = \sum_{s' \in S} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

---

### 4. Bellman Optimality Equations

**Optimal State-Value Function $V^*(s)$**:

$$V^*(s) = \max_{a \in A} \sum_{s' \in S} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

**Optimal Action-Value Function $Q^*(s, a)$**:

$$Q^*(s, a) = \sum_{s' \in S} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right]$$

---

### 5. Optimal Policy Extraction $\pi^*(s)$

$$\pi^*(s) = \arg\max_{a \in A} Q^*(s, a) = \arg\max_{a \in A} \sum_{s' \in S} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

---

## 4. Real-World Example

### Autonomous Robot Navigation in Stochastic Gridworld
The pipeline models a robot navigating a $5 \times 5$ grid environment with hazards, wall obstacles, and wind slip noise.

- **State Space**: 24 valid grid cells $(r, c)$ (1 cell is an impassable wall obstacle at $(2, 2)$).
- **Actions**: $A = \{\text{UP}, \text{RIGHT}, \text{DOWN}, \text{LEFT}\}$.
- **Rewards**:
  - Goal state $(0, 4)$: $+1.0$ (terminal)
  - Trap states $(1, 3)$ and $(3, 1)$: $-1.0$ (terminal hazards)
  - Step living reward: $-0.04$ (encourages shortest path)
- **Stochastic Noise**:
  - Intended direction probability = $0.8$
  - Perpendicular lateral slip probability = $0.1$ left, $0.1$ right

---

## 5. Worked MDP Sum (Step-by-Step)

Let us trace **Value Iteration** manually on a $1 \times 3$ grid:
- $S = \{s_0, s_1, s_2\}$ where $s_2$ is GOAL ($+1.0$), $s_0$ and $s_1$ are non-terminal.
- Actions: $A = \{\text{RIGHT}\}$. Deterministic transitions $P(s_{i+1} \mid s_i, \text{RIGHT}) = 1.0$.
- Rewards: $R(s_0 \to s_1) = -0.04$, $R(s_1 \to s_2) = +1.0$.
- Hyperparameters: $\gamma = 0.9$, initial $V_0(s) = 0$.

---

### Iteration 0: Initialisation
$$V_0(s_0) = 0, \quad V_0(s_1) = 0, \quad V_0(s_2) = 1.0$$

---

### Iteration 1: Update Values

1. **State $s_1$**:
   $$V_1(s_1) = R(s_1 \to s_2) + \gamma V_0(s_2) = 1.0 + 0.9(1.0) = 1.90$$
2. **State $s_0$**:
   $$V_1(s_0) = R(s_0 \to s_1) + \gamma V_0(s_1) = -0.04 + 0.9(0) = -0.04$$

State Values after Iteration 1:
$$V_1 = [-0.04, 1.90, 1.00]$$

---

### Iteration 2: Update Values

1. **State $s_1$**:
   $$V_2(s_1) = 1.0 + 0.9(1.0) = 1.90 \quad (\Delta = 0)$$
2. **State $s_0$**:
   $$V_2(s_0) = -0.04 + 0.9 V_1(s_1) = -0.04 + 0.9(1.90) = -0.04 + 1.71 = 1.67$$

State Values after Iteration 2:
$$V_2 = [1.67, 1.90, 1.00]$$

Optimal Policy: $\pi^*(s_0) = \text{RIGHT}, \quad \pi^*(s_1) = \text{RIGHT}$.

---

## 6. Program Flowchart

```
+-----------------------------------------------------+
|               START: main.py runs                   |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 1: Load PipelineConfig                        |
|  - PathConfig   (output directories)                |
|  - EnvConfig    (grid 5x5, goal, traps, noise)     |
|  - ModelConfig  (value_iteration, gamma=0.95)       |
|  - LoggingConfig (console-only logs)                |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 2: Initialize Console Logger                  |
|  - Output formatted to sys.stdout (No local files)  |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 3: Instantiate GridworldMDP(env_config)       |
|  - Build state space S, actions A, walls            |
|  - Compute stochastic transitions P(s'|s,a) & R     |
+-----------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|  Step 4: MDPSolverService.solve_and_evaluate()      |
+-----------------------------------------------------+
                           |
           +---------------+-------------------+
           |                                   |
           v                                   v
+---------------------------+       +---------------------------+
|  _value_iteration()       |       |  _policy_iteration()      |
|  Apply Bellman Optimality |  OR   |  Iterative Eval + Greedy  |
|  Operator iteratively     |       |  Policy Improvement       |
+---------------------------+       +---------------------------+
           |                                   |
           +---------------+-------------------+
                           |
                           v
+--------------------------------------------------+
|  _extract_optimal_policy()                       |
|  pi*(s) = argmax_a sum P(s'|s,a)[R + gamma V*(s')]|
+--------------------------------------------------+
                           |
                           v
+--------------------------------------------------+
|  Save Output Artifacts to output/                |
|  - Write mdp_results.txt                         |
|  - Write mdp_analysis.md report                  |
|  - Generate value_heatmap.png                    |
|  - Generate policy_grid.png                      |
|  - Generate convergence_curve.png                |
+--------------------------------------------------+
                           |
                           v
+-----------------------------------------------------+
|               END: Pipeline Complete               |
+-----------------------------------------------------+
```

---

## 7. Module Responsibility Map

```
main.py
  |
  +-- config.py           (PipelineConfig, PathConfig, EnvConfig,
  |                        ModelConfig, LoggingConfig)
  |
  +-- logger.py           (LoggerFactory - console stream log setup)
  |
  +-- mdp_environment.py  (GridworldMDP - states, actions, stochastic
  |                        transitions P(s'|s,a), rewards R(s,a,s'))
  |
  +-- mdp_solver.py       (MDPSolverService - Value Iteration, Policy Iteration,
                           optimal policy extraction, heatmaps, quiver grids, reporting)
```

---

## 8. Configuration

All parameters are configured in `src/config.py`.

| Parameter        | Location      | Default             | Description                                      |
|------------------|---------------|---------------------|--------------------------------------------------|
| `algorithm_type` | `ModelConfig` | `'value_iteration'` | Dynamic programming algorithm choice             |
| `gamma`          | `ModelConfig` | `0.95`              | Discount factor for future rewards               |
| `theta`          | `ModelConfig` | `1e-6`              | Bellman error convergence threshold              |
| `grid_rows`      | `EnvConfig`   | `5`                 | Number of grid rows                              |
| `grid_cols`      | `EnvConfig`   | `5`                 | Number of grid columns                           |
| `step_reward`    | `EnvConfig`   | `-0.04`             | Cost/reward per transition step                  |
| `success_prob`   | `EnvConfig`   | `0.8`               | Probability of moving in intended direction      |
| `slip_prob`      | `EnvConfig`   | `0.1`               | Perpendicular lateral slip probability           |
