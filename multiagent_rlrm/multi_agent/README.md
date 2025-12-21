# `multi_agent`: core multi-agent RL + Reward Machines

This package contains the **core building blocks** used by the environments in `multiagent_rlrm/environments/*`:

- a minimal multi-agent environment base (`BaseEnvironment`)
- an RL-capable agent abstraction (`AgentRL`)
- Reward Machines (`RewardMachine`) driven by pluggable event detectors (`EventDetector`)
- encoder interfaces (`StateEncoder`, `ActionEncoder`)
- a wrapper that “plugs” Reward Machines into an environment step loop (`RMEnvironmentWrapper`)

## Core abstractions

### `BaseEnvironment` (`base_environment.py`)

`BaseEnvironment` subclasses PettingZoo’s `ParallelEnv` and Unified-Planning’s `MultiAgentProblem`.
Child environments typically override `reset(...)` and `step(...)` to implement domain dynamics.

Expected step API (PettingZoo-style):

- `reset(seed=None, options=None) -> (observations, infos)`
- `step(actions) -> (observations, rewards, terminations, truncations, infos)`

Where:

- `actions` is a dict `{agent_name: action_obj}` (often an `ActionRL` instance).
- `observations`, `rewards`, `terminations`, `truncations`, `infos` are dicts keyed by agent name.

### `AgentRL` (`agent_rl.py`)

`AgentRL` extends `unified_planning.model.multi_agent.Agent` with:

- a per-agent `RewardMachine`
- a tabular learning algorithm (`set_learning_algorithm(...)`)
- a `StateEncoder` to map `(env_state, rm_state)` to an integer index
- an `ActionEncoder` to register the available `ActionRL`s for the agent

Main methods used by the training scripts:

- `select_action(state: dict, best: bool=False) -> ActionRL`
- `update_policy(state, action, reward, next_state, terminated, infos=...)`

### Actions & encoders

- `ActionRL` (`action_rl.py`): lightweight action object (`name`, optional preconditions/effects).
- `ActionEncoder` (`action_encoder.py`): abstract interface to register the agent’s actions via `build_actions()`.
- `StateEncoder` (`state_encoder.py`): abstract interface to encode a state; includes `encode_rm_state(...)` helper that converts RM state labels to indices.

Environment-specific encoders live in the environment folders (e.g., `environments/office_world/state_encoder_office.py`).

### Reward Machines & events

- `EventDetector` (`event_detector.py`): abstract interface `detect_event(current_state) -> event|None`.
- `RewardMachine` (`reward_machine.py`): automaton driven by detected events.

`RewardMachine` is initialized with:

- `transitions`: `{(rm_state, event): (next_rm_state, reward)}`
- `event_detector`: an `EventDetector` implementation

At runtime, the main loop calls `rm.step(env_state)`, which:

1. calls `event_detector.detect_event(env_state)`
2. applies the `(current_rm_state, event)` transition (if present)
3. returns the transition reward (otherwise `0`)

Note: `get_final_state()` is inferred from the last transition target in the transition dictionary.

## Integrating RMs into an environment loop (`RMEnvironmentWrapper`)

`wrappers/rm_environment_wrapper.py` wraps an environment and updates Reward Machines on every step:

- resets each agent’s RM on `reset(...)`
- calls `rm.step(...)` on each step and adds RM reward to the environment reward
- merges environment termination with “RM reached final state”
- injects RM-related fields into `infos[agent_name]` (e.g., `prev_q`, `q`, `RQ`, `reward_machine`)
- optionally generates “QRM experiences” (`qrm_experience`) when the learning algorithm exposes `use_qrm=True`

This wrapper is what makes Reward Machines “plug-and-play” for the training scripts.

## Minimal usage pattern

The concrete details (map parsing, event detectors, encoders) are environment-specific, but the structure is always:

```python
env = ...  # BaseEnvironment subclass

agent = AgentRL("a1", env)
agent.add_state_encoder(...)   # StateEncoder subclass
agent.add_action_encoder(...)  # ActionEncoder subclass (registers ActionRLs)
agent.set_reward_machine(...)  # RewardMachine
agent.set_learning_algorithm(...)  # e.g., QLearning / RMax / ...

env.add_agent(agent)
rm_env = RMEnvironmentWrapper(env, [agent])

obs, infos = rm_env.reset(seed=123)
done = {agent.name: False}
while not all(done.values()):
    act = agent.select_action(rm_env.env.get_state(agent))
    obs, rew, done, trunc, infos = rm_env.step({agent.name: act})
    agent.update_policy(..., infos=infos[agent.name])
```

For complete working examples, see:

- OfficeWorld: `../environments/office_world/README.md`
- FrozenLake: `../environments/frozen_lake/README.md`


## Related docs

- CLI (`rmgen`): `../cli/README.md`
- RM generation library: `../rmgen/README.md`

