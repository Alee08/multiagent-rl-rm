# Reinforcement Learning with Value Iteration Benchmarking and Early Stopping

The goal is to evaluate the performance of RL algorithms against optimal policies derived from VI, utilizing statistical methods to determine when training can be safely terminated early.

---

## Table of Contents

1. [Calculating and Using the p-value](#calculating-and-using-the-p-value)
2. [Creating the Markov Decision Process (MDP) for VI](#creating-the-markov-decision-process-mdp-for-vi)
3. [Generated Graphs and Visualizations](#generated-graphs-and-visualizations)
4. [Example Run](#example-run)
5. [Command-Line Arguments](#command-line-arguments)

---

## Calculating and Using the p-value

During the training process, at regular intervals or upon meeting specific conditions (e.g., receiving a positive reward), the training loop pauses to **test** two distinct policies:

1. **VI Policy**  
   Obtained theoretically through **Value Iteration** by modeling the environment's MDP.

2. **RL Policy**  
   The policy learned by the RL algorithm (e.g., **QRMAX**, **QRM**, **QL**, **RMAX**...) up to that point.

For each policy, multiple **test episodes** (e.g., 100) are run to collect the final rewards achieved in each episode. A **t-test** is then performed to compare the distributions of rewards between the VI and RL policies. The resulting **p-value** indicates whether there is a statistically significant difference between the two policies.

- **p-value > 0.1** (with `--early_stop` enabled in stochastic runs):  
  Indicates that there is **no significant evidence** to suggest that the RL policy performs worse than the VI policy. In this case, **early stopping** is triggered to halt further training, as the RL policy is considered sufficiently close to the optimal VI policy.

- **p-value ≤ 0.1**:  
  Suggests that the RL policy may still be significantly different (typically worse) than the VI policy, and training continues to allow the RL algorithm to improve.

**Why Use the p-value?**

- **Validation**: Ensures that the RL policy has reached a performance level that is not statistically worse than the optimal VI policy.
- **Efficiency**: Allows the training process to terminate early when additional training is unlikely to yield significant performance improvements, saving computational resources.

---

## Creating the Markov Decision Process (MDP) for VI

To perform **Value Iteration**, an MDP model of the environment is constructed with the following components:

- **States**: Combinations of the agent's position `(x, y)` and the state of the **Reward Machine**.  
  Example: `(position_x, position_y, reward_machine_state)`

- **Actions**: The set of possible actions the agent can take, such as `[up, down, left, right]`.

- **Transition Probabilities**:  
  - **Deterministic Environment**: Each action leads to a specific next state with probability 1.
  - **Stochastic Environment**: Each action may result in different outcomes with associated probabilities (e.g., attempting to move left might succeed with 0.8 probability and result in staying in place with 0.2 probability).

- **Rewards**:  
  - Penalties for encountering obstacles like walls or plants.
  - Rewards defined by the Reward Machine for reaching specific goal states.

The transition model `P[s][a]` is a matrix where each entry is a list of tuples representing possible outcomes:  
`P[s][a] = [(probability, next_state, reward, done), ...]`

**Value Iteration Process:**

1. **Initialization**: Start with arbitrary value estimates for each state.
2. **Update**: Iteratively update the value of each state based on the Bellman equation:
$$
V(s) = \max_{a} \sum_{s', r} P(s'|s,a) [r + \gamma V(s')]
$$
3. **Policy Extraction**: Once values converge, derive the optimal policy by selecting actions that maximize the expected value.

---

## Generated Graphs and Visualizations

The project generates several types of **heatmaps** and **graphs** to visualize and analyze the policies and their performance:

1. **Value Function Heatmap**:  
   Displays the value \( V(s) \) for each state in the grid, illustrating the desirability of each position.

2. **Policy Heatmap**:  
   Shows the optimal action determined by VI for each state, providing a visual representation of the policy.

3. **Reward Comparison Graphs**:  
   - **Mean Reward Comparison**: Bar charts comparing the average rewards of VI and RL policies.
   - **Standard Deviation Comparison**: Bar charts showing the variability in rewards for both policies.
   - **T-test Results**: Bar charts displaying the t-test statistic and p-value for the comparison between VI and RL rewards.

4. **Training Logs**:  
   Logs metrics such as training rewards, total steps, success rates, and evaluation results over time.

All visualizations can be viewed locally or integrated with **Weights & Biases (wandb)** for online tracking and analysis.

---

## Example Run

To execute a training session in a **stochastic environment** with early stopping enabled, use the following command:

```bash
python multiagent_rlrm/environments/office_world/office_main.py \
  --map map1 --experiment exp1 --stoc --alg QRMAX --steps 1e7 --wandb --early_stop
```

### Explanation of Arguments

- `--map map1`: Selects the first map layout (`map1`) for the experiment.
- `--experiment exp1`: Chooses the first experiment configuration (`exp1`), which defines specific scenarios and Reward Machines.
- `--stoc`: Enables stochasticity in the environment, introducing probabilistic outcomes for actions (shorthand for `--stochastic`).
- `--alg QRMAX`: Specifies the use of the QRMAX RL algorithm (shorthand for `--algorithm`).
- `--steps 1e7`: Sets the maximum number of training steps to 10 million.
- `--wandb`: Activates logging to Weights & Biases, allowing for real-time monitoring of training metrics and visualizations.
- `--early_stop`: Enables early stopping; in stochastic runs, a p-value > 0.1 can halt training when the RL policy is not statistically worse than the VI policy.

#### What Happens with `--early_stop`

During training, the algorithm periodically performs a t-test comparing the RL policy's performance against the VI policy. If `--early_stop` is enabled and the run is stochastic, a p-value > 0.1 triggers an early stop to terminate training prematurely. Some algorithms can also signal termination via `update_policy(...)`, which is only honored when `--early_stop` is set.

## Command-Line Arguments

For the full list, run:

```bash
python multiagent_rlrm/environments/office_world/office_main.py --help
```

Argparse accepts unique prefixes, so `--stoc` and `--alg` are shorthand for `--stochastic` and `--algorithm`.

Common flags:

- `--map`, `--experiment`, `--algorithm`
- `--stochastic`, `--highprob`, `--all-slip`
- `--steps`, `--eval`, `--early_stop`
- `--wandb`, `--render`, `--generate_heatmap`, `--vi_cache`
- `--rm-spec`, `--complete-missing-transitions`, `--default-reward`, `--terminal-self-loop`, `--terminal-reward-must-be-zero`
- `--gamma`, `--Kthreshold`, `--learning_rate`, `--VIdelta`, `--VIdeltarel`
