# rmgen (Reward Machine generation utilities)

`multiagent_rlrm.rmgen` is a small library that supports:

- **RMSpec**: a JSON/YAML-friendly schema for Reward Machines.
- **Validation**: structural checks + optional semantic constraints.
- **Normalization**: map/environment-aware event normalization (e.g., `A` → `at(A)`).
- **Completion**: auto-fill missing transitions with default self-loops.
- **Compilation**: compile an `RMSpec` into a runtime `RewardMachine`.
- **Text → RM**: providers + a pipeline for generating specs from natural language.

## Key modules

- `spec.py`: `RMSpec` and `TransitionSpec` dataclasses.
- `validator.py`: `validate_schema`, `ensure_deterministic`, `validate_semantics`.
- `normalize.py`: event normalization + OfficeWorld guardrails/autofix helpers.
- `completion.py`: `complete_missing_transitions(...)`.
- `io.py`: `load_rmspec(...)` and `compile_reward_machine(...)` convenience entrypoints.
- `pipeline.py`: `RMGenerationPipeline` (provider → spec → normalize → complete → validate → export).
- `providers.py`: provider implementations (`mock`, `openai_compat`, `openai`).
- `exporter.py`: JSON export and a generic `PassthroughEventDetector`.
- `summary.py`: formatting helpers for human-readable summaries.

## RMSpec schema (what a spec contains)

An `RMSpec` is a top-level object with (at least) these fields:

- `name`: string
- `env_id`: string (normalized to lowercase)
- `version`: string (e.g., `"1.0"`)
- `states`: list of unique state names (strings)
- `initial_state`: must be one of `states`
- `terminal_states`: subset of `states`
- `event_vocabulary`: list of unique event tokens (strings)
- `transitions`: list of `{from_state, event, to_state, reward}`
- `notes`: optional string

The runtime `RewardMachine` expects transitions as a mapping:
`{(from_state, event): (to_state, reward)}`.

## Loading specs (JSON/YAML)

Use `load_rmspec` to read a spec from disk:

```python
from multiagent_rlrm.rmgen.io import load_rmspec

spec = load_rmspec("rm.json")  # .yaml/.yml supported if PyYAML is installed
```

Notes:

- YAML parsing requires `pyyaml` (`pip install pyyaml`).
- JSON-with-`.yaml` extension is accepted because the loader tries JSON first.

## Compiling specs into a RewardMachine

`compile_reward_machine` validates the spec and returns a `RewardMachine`.

```python
from multiagent_rlrm.rmgen.io import compile_reward_machine, load_rmspec

spec = load_rmspec("rm.json")
rm = compile_reward_machine(spec)
```

If you do not provide an `event_detector`, compilation uses a generic
`PassthroughEventDetector` that expects environment states like `{"event": "at(A)"}`.

### Mapping spec events to environment events

Environments often detect events as *objects* (e.g., grid positions `(x, y)`).
`compile_reward_machine` supports an `event_mapping` from spec tokens to those objects:

```python
from multiagent_rlrm.rmgen.io import compile_reward_machine
from multiagent_rlrm.environments.office_world.detect_event import PositionEventDetector

event_mapping = {
    "at(A)": (1, 2),
    "at(coffee)": [(3, 4), (8, 6)],  # values may be lists/sets to expand one token
}
event_detector = PositionEventDetector({(1, 2), (3, 4), (8, 6)})

rm = compile_reward_machine(spec, event_detector=event_detector, event_mapping=event_mapping)
```

## Validation & semantics

Validation is split into:

- **Schema checks** (`validate_schema`): non-empty states/events/transitions, references are valid, uniqueness.
- **Determinism** (`ensure_deterministic`): forbids conflicting transitions for the same `(from_state, event)`.
- **Optional semantics** (`validate_semantics`):
  - `max_positive_reward_transitions`: cap the number of transitions with reward > 0
  - `terminal_reward_must_be_zero`: require reward 0 for all outgoing transitions from terminal states

## Normalization contexts

For map-derived environments, you typically want to constrain/normalize events.
Contexts are built in the environment modules (examples):

- OfficeWorld: `multiagent_rlrm.environments.office_world.event_context.build_officeworld_context(map_name)`
- FrozenLake: `multiagent_rlrm.environments.frozen_lake.event_context.build_frozenlake_context(map_name)`

Then normalize a spec via:

```python
from multiagent_rlrm.rmgen.normalize import normalize_rmspec_events

spec = normalize_rmspec_events(spec, context)
```

OfficeWorld also includes guardrails such as:

- auto-adding missing states referenced by transitions
- optional rewrites for common “terminal reward” mistakes

## Text → spec pipeline

`RMGenerationPipeline` orchestrates provider output, normalization, completion and validation:

```python
from multiagent_rlrm.rmgen.pipeline import RMGenerationPipeline
from multiagent_rlrm.rmgen.providers import OpenAICompatLLMClient

provider = OpenAICompatLLMClient(base_url="http://localhost:11434/v1", model="llama3.1:8b")
pipeline = RMGenerationPipeline(provider)
spec, _rm = pipeline.run("Go to A then B, reward on B", output_path=None)
```

For the end-user interface, see the CLI wrapper: `../cli/README.md`.

## Extending rmgen

Add a new provider:

1. Implement a class with `generate(task: str) -> str` returning a JSON string.
2. Add it to `providers.py` and expose it in the CLI (`multiagent_rlrm/cli/rmgen.py`).

Add a new context:

1. Implement `build_<env>_context(map_name) -> dict` in the environment package.
2. Register it in `_get_supported_contexts()` in `multiagent_rlrm/cli/rmgen.py`.

