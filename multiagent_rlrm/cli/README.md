# CLI: Reward Machine Generator (`rmgen`)

This folder contains the `rmgen` command-line interface: a small tool that turns a **natural-language task description** into a validated **Reward Machine spec** (JSON) that you can load in the environments (e.g., OfficeWorld / FrozenLake).

Run:

```bash
python -m multiagent_rlrm.cli.rmgen --help
```

## Quickstart (OfficeWorld)

Generate a spec from text (using an OpenAI-compatible endpoint such as Ollama/vLLM):

```bash
python -m multiagent_rlrm.cli.rmgen \
  --provider openai_compat --base-url http://localhost:11434/v1 --model llama3.1:8b \
  --context officeworld --map map1 \
  --task "A -> C -> B -> D (in order), reward 1 on D" \
  --output /tmp/rm_office.json
```

Train with it:

```bash
python multiagent_rlrm/environments/office_world/office_main.py \
  --map map1 --experiment exp4 --algorithm QL \
  --rm-spec /tmp/rm_office.json
```

## Providers

`rmgen` supports these providers:

- `openai_compat`: any OpenAI-compatible `/chat/completions` server (defaults: `--base-url http://localhost:11434/v1`, `--model llama3.1:8b`).
- `openai`: OpenAI hosted API (requires `OPENAI_API_KEY` or `--api-key`).
- `mock`: offline mode for tests/debugging (requires `--mock-fixture` with a JSON file to return).

## Contexts (event guardrails)

Using `--context` makes generation **environment-aware**:

- It injects the **allowed event tokens** derived from the selected map into the prompt.
- It normalizes common aliases into canonical events (preferred form: `at(X)`).

Supported contexts are defined in `multiagent_rlrm/cli/rmgen.py` and currently include:

- `officeworld` (requires `--map <map_name>`)
- `frozenlake` (requires `--map <map_name>`)

If you omit `--context`, the spec is still validated structurally, but events may not match any environment.

## Safe defaults

When `--context` is set, `rmgen` enables some safe defaults unless you pass `--no-safe-defaults`:

- forces a deterministic setup (temperature defaults to `0.0` unless you override it)
- `--complete-missing-transitions` is enabled
- `--max-positive-reward-transitions` defaults to `1` (to avoid accidental reward spam)

## Output format

Use `--output` to write the generated spec to disk. The CLI writes **JSON** (even if the filename ends with `.yaml`/`.yml`).

The loaders in this repo accept JSON and (optionally) YAML; JSON-with-`.yaml` extension is also accepted because the loader tries JSON parsing first.

## Flag reference (high level)

Core:

- `--task TEXT`: natural-language description of the task.
- `--provider {mock,openai_compat,openai}`: required.
- `--output PATH`: save the validated spec (JSON) to a file.

Provider config:

- `--base-url URL`: OpenAI-compatible base URL (`openai_compat` only).
- `--model NAME`: model name.
- `--api-key KEY`: API key (optional for local gateways).
- `--temperature FLOAT`: sampling temperature (defaults to `0.0`).
- `--mock-fixture PATH`: fixture JSON for `mock`.

Context / validation:

- `--context {officeworld,frozenlake}` and `--map MAP_NAME`
- `--complete-missing-transitions`, `--default-reward FLOAT`
- `--terminal-self-loop` / `--no-terminal-self-loop`
- `--terminal-reward-must-be-zero` / `--no-terminal-reward-must-be-zero`
- `--max-positive-reward-transitions N`
- `--no-safe-defaults`

## See also

- Library internals: `../rmgen/README.md`
- OfficeWorld environment: `../environments/office_world/README.md`
- FrozenLake environment: `../environments/frozen_lake/README.md`

