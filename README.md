# Multi-Agent RLRM
[![PyPI version](https://img.shields.io/pypi/v/multiagent-rl-rm.svg)](https://pypi.org/project/multiagent-rl-rm/)
[![CI]( https://github.com/alee08/multiagent-rl-rm/actions/workflows/ci.yml/badge.svg)](https://github.com/alee08/multiagent-rl-rm/actions/workflows/ci.yml)

## Introduction

The Multi-Agent RLRM (Reinforcement Learning with Reward Machines) Framework is a library designed to facilitate the formulation of multi-agent problems and solve them through reinforcement learning. The framework supports the integration of Reward Machines (RMs), providing a modular and flexible structure for defining complex tasks through a set of objectives and rules.

## Installation

### Option A — PyPI (recommended)

```bash
pip install multiagent-rl-rm
```
import path: multiagent_rlrm (underscore).

### Option B — From source (for development)

To install the framework, follow these steps:

```bash
git clone https://github.com/Alee08/multi-agent-rl-rm.git
cd multi-agent-rl-rm
pip install -r requirements.txt
pip install -e .
```


## Installation with docker

Build the container image from the repository root:

```bash
docker build -f docker/Dockerfile -t multiagent-rlrm .
docker run --rm -it multiagent-rlrm python
```

More details (compose, examples, troubleshooting) are available in `docker/README.md`.

## Quickstart

Train the multi-agent FrozenLake example (built-in RM: `A -> B -> C`):
```bash
pip install -e .
python -m multiagent_rlrm.environments.frozen_lake.frozen_lake_main \
  --map map1 --num-episodes 2000 --render-every 0
```

## Reward Machine generation (`rmgen`)

`rmgen` turns text (NL or a compact RM description) into a validated RM spec (`.json`/`.yaml`).
You can then load the spec from experiment scripts via `--rm-spec` (e.g., OfficeWorld and FrozenLake entrypoints).

<details>
<summary>Offline RM generation (mock provider)</summary>

Generate and validate a Reward Machine offline using the mock provider and fixtures:
```bash
pip install -e .
python -m multiagent_rlrm.cli.rmgen --provider mock \
  --mock-fixture tests/fixtures/officeworld_simple.json \
  --task "go to A then G" \
  --output /tmp/rm.json
```
If you point `--mock-fixture` to a non-deterministic spec (e.g., `tests/fixtures/nondeterministic_rm.json`), the command exits with code 1 and prints a validation error.

</details>

<details>
<summary>LLM local via Ollama (OpenAI-compatible)</summary>

1) Install and start Ollama (see https://ollama.com/docs/installation), then run the server:
```bash
ollama serve  # or ensure the Ollama service is running
```
2) Pull a model that exposes the OpenAI-compatible endpoint, e.g.:
```bash
ollama pull llama3.1:8b
```
3) Generate a Reward Machine via the OpenAI-compatible provider:
```bash
python -m multiagent_rlrm.cli.rmgen --provider openai_compat \
  --base-url http://localhost:11434/v1 \
  --model llama3.1:8b \
  --task "go to A, then B, then goal" \
  --output /tmp/rm.yaml
```
You can also provide a short prompt with only key transitions and ask the tool to complete the cartesian product with self-loops:
```bash
python -m multiagent_rlrm.cli.rmgen --provider openai_compat \
  --base-url http://localhost:11434/v1 \
  --model llama3.1:8b \
  --task "q0 at(A)->q1, q1 at(G)->q2 reward 1, default self-loop" \
  --complete-missing-transitions --default-reward 0.0 \
  --output /tmp/rm.yaml
```

</details>

<details>
<summary>NL→RM→Train (OfficeWorld / FrozenLake)</summary>

Generate an RM spec with Ollama and use it directly in an experiment script via `--rm-spec`.

When using a map-derived context (e.g., `--context officeworld` / `--context frozenlake`), `rmgen` also applies safe defaults to make the short command reliable:
- `temperature=0` (unless explicitly overridden)
- `--complete-missing-transitions` enabled
- `--max-positive-reward-transitions 1`
`env_id` is enforced to the selected context (`officeworld`/`frozenlake`) even if the model outputs a typo.

### OfficeWorld
1) Generate `/tmp/rm.yaml` (JSON or YAML; `.yaml` works even if the content is JSON):
```bash
python -m multiagent_rlrm.cli.rmgen --provider openai_compat \
  --base-url http://localhost:11434/v1 \
  --model llama3.1:8b \
  --context officeworld --map map1 \
  --task "A -> C -> B -> D, reward on D" \
  --output /tmp/rm.yaml
```
This command auto-injects the **allowed events** derived from the selected OfficeWorld map into the LLM prompt and then normalizes events (e.g., `A` → `at(A)`, `at(office)` → `at(O)`) before validation.
If the model mistakenly puts a non-zero reward on an outgoing transition from a terminal state, OfficeWorld context auto-repairs the spec (by redirecting into a new terminal sink state) when `terminal_reward_must_be_zero` is enabled.

To disable these safe defaults (legacy behavior), add `--no-safe-defaults`:
```bash
python -m multiagent_rlrm.cli.rmgen --provider openai_compat \
  --base-url http://localhost:11434/v1 \
  --model llama3.1:8b \
  --context officeworld --map map1 --no-safe-defaults \
  --task "A -> C -> B -> D, reward on D" \
  --output /tmp/rm.yaml
```

You can also provide a more explicit textual RM:
```bash
python -m multiagent_rlrm.cli.rmgen --provider openai_compat \
  --base-url http://localhost:11434/v1 \
  --model llama3.1:8b \
  --task "OfficeWorld: q0 at(A)->q1, q1 at(B)->q2, q2 at(C)->q3, q3 at(D)->q4 reward 1, default self-loop" \
  --complete-missing-transitions --default-reward 0.0 \
  --output /tmp/rm.yaml
```

2) Train using the generated spec (OfficeWorld entrypoint):
```bash
python multiagent_rlrm/environments/office_world/office_main.py \
  --map map1 --experiment exp4 --algorithm QL \
  --rm-spec /tmp/rm.yaml
```

### FrozenLake
FrozenLake context injects the allowed goal events derived from the selected emoji map (holes are excluded from the event vocabulary) and normalizes bare symbols (e.g., `A` → `at(A)`):
```bash
python -m multiagent_rlrm.cli.rmgen --provider openai_compat \
  --base-url http://localhost:11434/v1 \
  --model llama3.1:8b \
  --context frozenlake --map map1 \
  --task "B -> A -> C (exact order), reward 1 on C" \
  --output /tmp/rm_frozenlake.json
```

Train with the generated spec (FrozenLake entrypoint):
```bash
python -m multiagent_rlrm.environments.frozen_lake.frozen_lake_main \
  --map map1 --rm-spec /tmp/rm_frozenlake.json
```

You can also use different RMs per agent:
```bash
python -m multiagent_rlrm.environments.frozen_lake.frozen_lake_main \
  --map map1 --rm-spec-a1 /tmp/a1.json --rm-spec-a2 /tmp/a2.json
```

</details>

## Usage (Python API)
This repository includes several environments (Frozen Lake, Office World, Pickup & Delivery, etc.). Below is a compact end-to-end example for two agents in the Frozen Lake environment, each with its own Reward Machine (RM) and tabular Q-learning.

<details open>
<summary>End-to-end FrozenLake example (two agents)</summary>

### Step 1: Environment Setup
First, import the necessary modules and initialize the `MultiAgentFrozenLake` environment. You can derive width/height, holes and goals from an emoji layout via `parse_map_emoji` (as in `multiagent_rlrm/environments/frozen_lake/frozen_lake_main.py`).
```python
from multiagent_rlrm.environments.frozen_lake.ma_frozen_lake import MultiAgentFrozenLake
from multiagent_rlrm.environments.frozen_lake.action_encoder_frozen_lake import ActionEncoderFrozenLake
from multiagent_rlrm.utils.utils import parse_map_emoji

MAP_LAYOUT = """
  B 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 ⛔ ⛔ 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 A  🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🟩 ⛔ ⛔
 🟩 🟩 🟩 🟩  C 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
"""

holes, goals, (W, H) = parse_map_emoji(MAP_LAYOUT)
env = MultiAgentFrozenLake(width=W, height=H, holes=holes)
env.frozen_lake_stochastic = False      # deterministic here; set True to enable slip/stochastic dynamics
env.penalty_amount = 0                  # penalty when falling into a hole
env.delay_action = False                # optional "wait" bias if True
```



### Step 2: Define Agents and Action/State Encoders

Create agent instances, set their initial positions, and attach domain-specific encoders
for state and actions. In Frozen Lake, the `StateEncoderFrozenLake` maps grid positions
(and RM state) to tabular indices, while `ActionEncoderFrozenLake` registers the
discrete actions (`up`, `down`, `left`, `right`) for each agent. Finally, register the
agents with the environment so `reset`/`step` include them. Coordinates below match `multiagent_rlrm/environments/frozen_lake/frozen_lake_main.py`.
```python
from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagent_rlrm.multi_agent.action_rl import ActionRL
from multiagent_rlrm.environments.frozen_lake.state_encoder_frozen_lake import StateEncoderFrozenLake

a1, a2 = AgentRL("a1", env), AgentRL("a2", env)
a1.set_initial_position(5, 0)
a2.set_initial_position(0, 0)

for ag in (a1, a2):
    ag.add_state_encoder(StateEncoderFrozenLake(ag))
    ag.add_action_encoder(ActionEncoderFrozenLake(ag))

env.add_agent(a1)
env.add_agent(a2)
```



### Step 3: Define Reward Machines (one per agent)
You define the task as a small automaton (the Reward Machine). The `PositionEventDetector` turns grid visits into events; here we mirror `multiagent_rlrm/environments/frozen_lake/frozen_lake_main.py`: reach A then B then C. Each agent gets its own RM (rm1, rm2), so progress and rewards are tracked independently even in the same environment.

```python
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.environments.frozen_lake.detect_event import PositionEventDetector

# Events are the goal positions from the parsed map
event_detector = PositionEventDetector(set(goals.values()))
# transitions = {(current_state, event): (next_state, reward)}
transitions = {
    ("state0", goals["A"]): ("state1", 0),
    ("state1", goals["B"]): ("state2", 0),
    ("state2", goals["C"]): ("state3", 1),
}

rm1 = RewardMachine(transitions, event_detector)
rm2 = RewardMachine(transitions, event_detector)
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
        use_qrm=True, use_rsh=False  # optional: counterfactuals & RM shaping
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

### Frozen Lake layout & sample episode

Sample run saved at episode 1000 (deterministic dynamics, `frozen_lake_stochastic=False`). Run `python -m multiagent_rlrm.environments.frozen_lake.frozen_lake_main --render-every 100` to generate the files; they are saved under `./episodes/episode_1000.*` (relative to your working directory). Rendering uses pygame with a side panel showing RM state, events, and per-agent steps.


<p align="center">
  <img src="multiagent_rlrm/environments/frozen_lake/episodes/episode_1000.gif"
       alt="Frozen Lake Episode 1000"
       width="800">
</p>
</details>

## Implemented learning algorithms

All algorithms live in `multiagent_rlrm/learning_algorithms` and expose a common interface via `choose_action(...)` and `update(...)`.

| Algorithm    | Type                         | Short description                                                                                                                                                                                                                                                                                                                              |
|-------------|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `QLearning` | Model-free, tabular          | Standard tabular Q-learning with ε-greedy or softmax exploration; supports Reward Machines via QRM-style counterfactual updates and optional potential-based reward shaping.                                                                                                                                                                   |
| `QLambda`| Model-free, tabular, eligibility traces | Q-learning with eligibility traces (λ): propagates TD errors backwards along recent state–action pairs, enabling faster credit assignment over multi-step trajectories and often speeding up learning in sparse-reward settings. |
| `QRM`       | Model-free, RM-aware         | Q-learning over Reward Machines: augments the state with the RM automaton state and uses counterfactual updates across compatible automaton states to reuse experience under non-Markovian rewards.                                                                                                                                            |
| `RMax`      | Model-based, optimistic      | Classic R-Max algorithm: learns an explicit tabular transition/reward model, treats unknown state–action pairs as maximally rewarding, and plans via value iteration to drive directed exploration.                                                                                                                                            |
| `RMaxRM`    | Model-based, RM-aware        | R-Max on the product space S×Q): uses the Reward Machine to augment the MDP state but does **not** factorise environment and automaton dynamics; serves as a RM-aware model-based baseline.                                                                                                                                               |
| `QRMax`     | Model-based, factored, RM-aware | R-Max-style model-based RL for non-Markovian rewards via Reward Machines; factorises environment dynamics and RM dynamics, reuses each learned environment transition across RM states, and preserves PAC-style sample-efficiency guarantees. The algorithm only requires the current RM state and reward signal, not the full RM description. |
| `QRMaxRM`   | Model-based, RM-aware (extra RM experience) | Extension of `QRMax` that also leverages additional experience generated from the known Reward Machine, applying the same factorised updates to both real and counterfactual transitions to further improve sample efficiency.                                                                                                                 |
| `PSRL`      | Model-based, posterior sampling | Posterior Sampling for RL (Thompson sampling over MDPs): maintains Bayesian posteriors over transitions and rewards, samples an MDP each episode, and follows its optimal policy.                                                                                                                                                              |
| `OPSRL`     | Model-based, optimistic posterior sampling | Optimistic PSRL variant with Dirichlet/Beta priors and optimistic treatment of under-explored transitions, encouraging exploration by biasing sampled models toward rewarding but uncertain dynamics.                                                                                                                                          |
| `UCBVI`     | Model-based, UCB-style (base class) | Base implementation of tabular UCB Value Iteration for finite-horizon MDPs: empirical models plus step-wise exploration bonuses and backward value iteration. Concrete variants differ only in the bonus definition.                                                                                                                           |
| `UCBVI-sB`  | Model-based, UCBVI (simplified Bernstein) | UCBVI with simplified Bernstein bonuses as in Azar et al. (2017), trading off tightness of confidence intervals and implementation simplicity.                                                                                                                                                                                                 |
| `UCBVI-B`   | Model-based, UCBVI (Bernstein) | UCBVI variant using full Bernstein-style bonuses, yielding tighter confidence bounds and typically stronger theoretical guarantees.                                                                                                                                                                                                            |
| `UCBVI-H`   | Model-based, UCBVI (Hoeffding) | UCBVI variant with Hoeffding-style bonuses, using simpler but more conservative confidence intervals.                                                                                                                                                                                                                                          |

## License

Multi-Agent RLRM is released under the **Apache 2.0 License**.  
See the `LICENSE` file for details.
