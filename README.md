# Multi-Agent RLRM

## Introduction

The Multi-Agent RLRM (Reinforcement Learning with Reward Machines) Framework is a library designed to facilitate the formulation of multi-agent problems and solve them through reinforcement learning. The framework supports the integration of Reward Machines (RMs), providing a modular and flexible structure for defining complex tasks through a set of objectives and rules.

## Installation

To install the framework, follow these steps:

1. Clone the GitHub repository:

   ```bash
   git clone https://github.com/Alee08/multi-agent-rl-rm.git

2. Navigate to the project directory and install the dependencies (if applicable):
    ```bash 
    cd multi-agent-rl-rm
    pip install -r requirements.txt
3. Install the package in development mode to make changes to the code and have them immediately reflected without needing to reinstall the package:
    ```bash 
   pip install -e .


## Installation with docker

Follow instructions in the docker folder.


## Usage 
Below is a compact end-to-end example for two agents in the Frozen Lake environment, each with its own Reward Machine (RM) and tabular Q-learning.

### Step 1: Environment Setup
First, import the necessary modules and initialize the `MultiAgentFrozenLake` environment with desired parameters such as grid size and hole locations.
In the above code, `holes` represents the coordinates of obstacles within the grid that the agents must avoid. This setup provides a simple yet challenging environment for agents to learn navigation strategies.
```python
from multiagent_rlrm.environments.frozen_lake.ma_frozen_lake import MultiAgentFrozenLake

W, H = 10, 10
holes = [(2,3), (2,4), (7,0), (7,1), (7,2), (7,3), (7,4), (7,8)]
env = MultiAgentFrozenLake(width=W, height=H, holes=holes)
env.frozen_lake = True      # slip/stochastic dynamics
env.penalty_amount = 0      # penalty when falling into a hole
env.delay_action = False    # optional "wait" bias if True
```

### Step 2: Define Agents and Actions
Create agent instances, set their initial positions, and define possible actions along with their preconditions and effects.

Below is an example of defining the 'move_up' action, demonstrating how to set up additional actions for the agents. 
Each action is defined by its preconditions (when the action is applicable) and its effects (the outcome of performing the action).

```python
from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagent_rlrm.multi_agent.action_rl import ActionRL
from multiagent_rlrm.environments.frozen_lake.state_encoder_frozen_lake import StateEncoderFrozenLake

a1, a2 = AgentRL("a1", env), AgentRL("a2", env)
a1.set_initial_position(4, 0)
a2.set_initial_position(6, 2)

for a in (a1, a2):
    a.add_state_encoder(StateEncoderFrozenLake(a))

def can_up(a):    return a.get_position()[1] > 0
def can_down(a):  return a.get_position()[1] < H-1
def can_left(a):  return a.get_position()[0] > 0
def can_right(a): return a.get_position()[0] < W-1

def eff_up(a):    a.set_position(a.get_position()[0], a.get_position()[1]-1)
def eff_down(a):  a.set_position(a.get_position()[0], a.get_position()[1]+1)
def eff_left(a):  a.set_position(a.get_position()[0]-1, a.get_position()[1])
def eff_right(a): a.set_position(a.get_position()[0]+1, a.get_position()[1])

actions = [
    ActionRL("up",    [can_up],    [eff_up]),
    ActionRL("down",  [can_down],  [eff_down]),
    ActionRL("left",  [can_left],  [eff_left]),
    ActionRL("right", [can_right], [eff_right]),
]
for a in (a1, a2):
    for act in actions: a.add_action(act)
```



### Step 3: Define Reward Machines (one per agent)
You define the task as a small automaton (the Reward Machine). The `PositionEventDetector` turns grid visits into events; here, reaching (4,4) triggers a transition from q0→q1 (+0), then reaching (0,0) triggers q1→qf (+1, final). Each agent gets its own RM (rm1, rm2), so progress and rewards are tracked independently even in the same environment. This cleanly separates what should be achieved (waypoints/sequence) from how the agent moves in a stochastic world, and you can extend it by adding more waypoints, branches, or different detectors.

```python
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.environments.frozen_lake.detect_event import PositionEventDetector

# visit cells in sequence to progress and collect rewards
e1, e2 = (4,4), (0,0)
transitions = {
    ("q0", e1): ("q1", 10),
    ("q1", e2): ("qf", 100),  # final RM state
}
detector = PositionEventDetector({e1, e2})

rm1 = RewardMachine(transitions, detector)
rm2 = RewardMachine(transitions, detector)
a1.set_reward_machine(rm1)
a2.set_reward_machine(rm2)
```


### Step4: Wrap env with RM and set learners
Wrap the base environment with `RMEnvironmentWrapper` so RM logic is applied automatically at every step: it detects events, updates each agent’s RM state, and merges env reward + RM reward (and termination). The learner’s state size must include RM states `(W*H*rm.numbers_state())`, because policies depend on both position and RM progress. Assign a separate Q-learning instance per agent. Optional knobs: use_qrm=True for counterfactual RM updates and `use_rsh=True` for potential-based shaping.

```python
from multiagent_rlrm.multi_agent.wrappers.rm_environment_wrapper import RMEnvironmentWrapper
from multiagent_rlrm.learning_algorithms.qlearning import QLearning

rm_env = RMEnvironmentWrapper(env, [a1, a2])

def make_ql(rm):  # state size includes RM states
    return QLearning(
        state_space_size=W * H * rm.numbers_state(),
        action_space_size=4,
        learning_rate=0.2,
        gamma=0.99,
        action_selection="greedy",
        epsilon_start=0.01, epsilon_end=0.01, epsilon_decay=0.9995,
        # use_qrm=True, use_rsh=True  # optional: counterfactuals & RM shaping
    )

a1.set_learning_algorithm(make_ql(rm1))
a2.set_learning_algorithm(make_ql(rm2))
```

### Step5: Training Loop
Standard episodic loop. On each episode, reset initializes env + each agent’s RM state. Every step: each agent picks an action from the raw env state; the wrapped env executes them, detects events, and returns env+RM rewards plus per-agent termination flags. Then each agent calls update_policy(...) to learn from `(s, a, r, s')` (the learner/encoder handle RM progress internally). The loop stops when all agents are done (hole/time-limit or final RM state).

```python
import copy

EPISODES = 1000
for ep in range(EPISODES):
    obs, infos = rm_env.reset(seed=123 + ep)
    done = {ag.name: False for ag in rm_env.agents}

    while not all(done.values()):
        actions = {}
        for ag in rm_env.agents:
            s = rm_env.env.get_state(ag)          # raw env state for the agent
            actions[ag.name] = ag.select_action(s)

        next_obs, rewards, terms, truncs, infos = rm_env.step(actions)

        for ag in rm_env.agents:
            terminated = terms[ag.name] or truncs[ag.name]
            ag.update_policy(
                state=obs[ag.name],
                action=actions[ag.name],
                reward=rewards[ag.name],           # env + RM reward
                next_state=next_obs[ag.name],
                terminated=terminated,
                infos=infos[ag.name],              # includes RM fields
            )
            done[ag.name] = terminated

        obs = copy.deepcopy(next_obs)
```
In this loop, agents continuously assess their environment, make decisions, and act accordingly. The env.step(actions) method encapsulates the agents' interactions with the environment, including executing actions, receiving new observations, calculating rewards, and updating the agents' policies based on the results. This streamlined process simplifies the learning loop and focuses on the essential elements of agent-environment interaction.




