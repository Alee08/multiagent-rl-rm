# FrozenLake (multi-agent) + Reward Machines

This folder contains the **MultiAgentFrozenLake** environment and an example training script (`frozen_lake_main.py`) that trains **two agents** with **tabular Q-learning** and a **Reward Machine (RM)**.

## Key files

- `ma_frozen_lake.py`: `MultiAgentFrozenLake` environment (grid, holes, action dynamics).
- `frozen_lake_main.py`: training entrypoint (2 agents, RM, logging, optional rendering).
- `state_encoder_frozen_lake.py`: encodes `(x, y, q_rm) -> int` for tabular algorithms.
- `action_encoder_frozen_lake.py`: registers symbolic actions `up|down|left|right`.
- `detect_event.py`: `PositionEventDetector` (event = reached position).
- `event_context.py`: builds the FrozenLake “guardrail” context for `rmgen` (allowed/normalized events).

## Map format & parser (`parse_map_emoji`)

FrozenLake maps are defined in `config_frozen_lake.py` as emoji layouts. The parser `multiagent_rlrm.utils.utils.parse_map_emoji(layout)` interprets a multiline string where:

- `⛔` = hole
- letters/digits = goals (they define the RM event vocabulary, e.g. `A`, `B`, `1`, …)
- anything else (e.g. `🟩`) = free floor
- spaces/indentation are ignored

It returns:

- `holes`: list of `(x, y)`
- `goals`: dict `{label -> (x, y)}`
- `dims`: `(width, height)`

### Coordinate system and actions

- `(0, 0)` is the **top-left** cell of the layout; `x` increases to the right, `y` increases downward.
- In `ma_frozen_lake.py`: `up` decrements `y`, `down` increments `y`.

## Reward Machines (RMs)

In this project, an RM can be authored in two “human-friendly” ways:

- **Python**: build a transition dictionary and instantiate `RewardMachine(...)` directly (this is what the built-in example does).
- **Natural language**: describe the task in plain English and use `rmgen` to generate a JSON/YAML spec, then load it with `--rm-spec*`.

Both workflows end up with the same runtime object: a `RewardMachine` attached to each agent.

By default, `frozen_lake_main.py` uses a small hard-coded RM (one per agent) with the sequence **A → B → C**:

- `state0 --at(A)--> state1`
- `state1 --at(B)--> state2`
- `state2 --at(C)--> state3`

Events are grid positions `(x, y)` detected by `PositionEventDetector`. When using a spec file, textual events (`A`, `at(A)`, …) are mapped to positions via the goal dictionary.

### Writing an RM in Python (hard-coded)

Minimal pattern (see `frozen_lake_main.py`):

```python
transitions = {
    ("state0", goals["A"]): ("state1", 10),
    ("state1", goals["B"]): ("state2", 15),
    ("state2", goals["C"]): ("state3", 20),
}
event_detector = PositionEventDetector(set(goals.values()))
rm = RewardMachine(transitions, event_detector)
```

### Loading an RM spec from file (`--rm-spec*`)

You can replace the default RM with a JSON/YAML spec (e.g., generated via `multiagent_rlrm.cli.rmgen`):

- `--rm-spec PATH`: shared spec for both agents
- `--rm-spec-a1 PATH` and `--rm-spec-a2 PATH`: one spec per agent

You can also auto-complete missing transitions:

- `--complete-missing-transitions --default-reward 0.0`

Allowed/normalized events for FrozenLake are derived from the selected emoji map in `event_context.py` (holes are intentionally excluded from the RM event vocabulary).

## Generating an RM from natural language (rmgen)

You can generate a valid RM spec directly from a natural-language description using the `rmgen` CLI. Example (FrozenLake, map1):

```bash
python -m multiagent_rlrm.cli.rmgen \
  --provider openai_compat --base-url http://localhost:11434/v1 --model llama3.1:8b \
  --context frozenlake --map map1 \
  --task "B then A then C (exact order), reward 1 on C" \
  --output /tmp/rm_frozenlake.json
```

Then train with it:

```bash
python -m multiagent_rlrm.environments.frozen_lake.frozen_lake_main \
  --map map1 --rm-spec /tmp/rm_frozenlake.json \
  --complete-missing-transitions --default-reward 0.0
```

Note: `rmgen` needs an LLM backend (any OpenAI-compatible endpoint or similar). If you don't have one, write the JSON/YAML spec by hand and pass it via `--rm-spec*`.

## Running

From the repository root:

```bash
python -m multiagent_rlrm.environments.frozen_lake.frozen_lake_main --map map1
```

Per-agent specs:

```bash
python -m multiagent_rlrm.environments.frozen_lake.frozen_lake_main \
  --map map1 --rm-spec-a1 /tmp/a1.json --rm-spec-a2 /tmp/a2.json
```

## CLI arguments

For the authoritative list, run:

```bash
python -m multiagent_rlrm.environments.frozen_lake.frozen_lake_main --help
```

Common flags:

| Flag | Default | Description |
|---|---:|---|
| `--map` | `map1` | Map name from `config_frozen_lake.py`. |
| `--num-episodes` | `30000` | Number of training episodes. |
| `--render-every` | `100` | Record an episode every N episodes (use `0` to disable). |
| `--seed` | `111` | Seed passed to `env.reset(...)`. |
| `--rm-spec` | (none) | RM spec file (`.json/.yaml`) shared by both agents (overrides built-in RM). |
| `--rm-spec-a1` | (none) | RM spec file for agent `a1` (overrides `--rm-spec`). |
| `--rm-spec-a2` | (none) | RM spec file for agent `a2` (overrides `--rm-spec`). |
| `--complete-missing-transitions` | `False` | Auto-complete missing spec transitions (only with `--rm-spec*`). |
| `--default-reward` | `0.0` | Reward used for auto-completed transitions (only with `--rm-spec*`). |

Notes:

- Provide either `--rm-spec` (shared) or **both** `--rm-spec-a1` and `--rm-spec-a2` (per-agent); mixed configurations are rejected.

## Outputs

- `episodes/`: recorded videos/GIFs (when `--render-every > 0`).
- `data/q_tables.npz`: Q-tables saved at the end of training (keys `q_table_<agent_name>`).
