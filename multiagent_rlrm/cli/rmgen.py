import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from multiagent_rlrm.rmgen.pipeline import RMGenerationPipeline
from multiagent_rlrm.rmgen.providers import MockLLMClient
from multiagent_rlrm.rmgen.validator import ValidationError


@dataclass(frozen=True)
class ContextConfig:
    label: str
    env_id: str
    build_context: Callable[[str], Mapping[str, object]]
    map_required: bool = True


def _get_supported_contexts() -> Mapping[str, ContextConfig]:
    # Import lazily to keep rmgen CLI import costs low and to avoid importing
    # environment modules when rmgen is used without contexts.
    from multiagent_rlrm.environments.frozen_lake.event_context import (
        build_frozenlake_context,
    )
    from multiagent_rlrm.environments.office_world.event_context import (
        build_officeworld_context,
    )

    return {
        "officeworld": ContextConfig(
            label="OfficeWorld",
            env_id="officeworld",
            build_context=build_officeworld_context,
        ),
        "frozenlake": ContextConfig(
            label="FrozenLake",
            env_id="frozenlake",
            build_context=build_frozenlake_context,
        ),
    }


def parse_args(argv):
    supported_contexts = _get_supported_contexts()
    parser = argparse.ArgumentParser(
        description="Generate a Reward Machine from text using a selected provider."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Text description of the task to turn into a Reward Machine.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "openai_compat", "openai"],
        required=True,
        help="Provider to use for generation.",
    )
    parser.add_argument(
        "--mock-fixture",
        type=Path,
        help="Path to a fixture JSON to return when using the mock provider.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434/v1",
        help="Base URL for OpenAI-compatible endpoints (ignored for mock).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for OpenAI/OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for OpenAI/OpenAI-compatible providers (optional for local gateways).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for LLM providers.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        choices=sorted(supported_contexts.keys()),
        help="Optional generation context that injects environment-derived guardrails.",
    )
    parser.add_argument(
        "--map",
        type=str,
        default=None,
        help="Map name for the selected context (required for --context officeworld).",
    )
    parser.add_argument(
        "--no-safe-defaults",
        action="store_true",
        help="Disable OfficeWorld safe defaults (keeps the legacy behavior).",
    )
    parser.add_argument(
        "--complete-missing-transitions",
        action="store_true",
        help="Auto-complete missing transitions with self-loops and default reward.",
    )
    parser.add_argument(
        "--default-reward",
        type=float,
        default=0.0,
        help="Reward value for auto-completed transitions.",
    )
    parser.add_argument(
        "--terminal-self-loop",
        dest="terminal_self_loop",
        action="store_true",
        default=True,
        help="When completing missing transitions, add self-loops for terminal states.",
    )
    parser.add_argument(
        "--no-terminal-self-loop",
        dest="terminal_self_loop",
        action="store_false",
        help="Skip auto-completion for terminal states.",
    )
    parser.add_argument(
        "--max-positive-reward-transitions",
        type=int,
        default=None,
        help="If set, fail if the number of transitions with reward>0 exceeds this value.",
    )
    parser.add_argument(
        "--terminal-reward-must-be-zero",
        dest="terminal_reward_must_be_zero",
        action="store_true",
        default=True,
        help="Require terminal states to have only reward 0 transitions (default).",
    )
    parser.add_argument(
        "--no-terminal-reward-must-be-zero",
        dest="terminal_reward_must_be_zero",
        action="store_false",
        help="Allow non-zero rewards from terminal states.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save the validated RM spec JSON. If omitted, no file is written.",
    )
    return parser.parse_args(argv)


def build_provider(args):
    if args.provider == "mock":
        if not args.mock_fixture:
            raise SystemExit("--mock-fixture is required for provider 'mock'")
        return MockLLMClient(fixture_path=args.mock_fixture)
    if args.provider == "openai_compat":
        from multiagent_rlrm.rmgen.providers import OpenAICompatLLMClient

        model = args.model or "llama3.1:8b"
        return OpenAICompatLLMClient(
            base_url=args.base_url,
            model=model,
            api_key=args.api_key,
            temperature=args.temperature,
        )
    if args.provider == "openai":
        from multiagent_rlrm.rmgen.providers import OpenAILLMClient

        model = args.model or "gpt-4o-mini"
        return OpenAILLMClient(
            model=model,
            api_key=args.api_key,
            temperature=args.temperature,
        )
    raise SystemExit(f"Unsupported provider: {args.provider}")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    args = parse_args(argv)
    supported_contexts = _get_supported_contexts()

    temperature_provided = args.temperature is not None
    if args.temperature is None:
        # Backward-compatible default.
        args.temperature = 0.0

    if args.context in supported_contexts and not args.no_safe_defaults:
        # Safe defaults for map-derived contexts to reduce user flags and increase robustness.
        if not temperature_provided:
            args.temperature = 0.0
        args.complete_missing_transitions = True
        if args.max_positive_reward_transitions is None:
            args.max_positive_reward_transitions = 1

    provider = build_provider(args)
    pipeline = RMGenerationPipeline(provider)

    normalize_context = None
    enforce_env_id = None
    task = args.task
    if args.context in supported_contexts:
        ctx = supported_contexts[args.context]
        if ctx.map_required and not args.map:
            raise SystemExit(f"--map is required when using --context {args.context}")

        normalize_context = ctx.build_context(args.map)
        enforce_env_id = ctx.env_id
        allowed_events = normalize_context["allowed_events"]
        allowed_block = "\n".join(f"- {ev}" for ev in allowed_events)
        task = (
            f"{task}\n\n"
            f"{ctx.label} map: {args.map}\n"
            "Allowed events (use ONLY these tokens as events; do not invent new events):\n"
            f"{allowed_block}\n"
            "Rules:\n"
            "- Use ONLY the allowed event tokens above.\n"
            "- Prefer the canonical form `at(X)`.\n"
            f"- Set env_id to '{ctx.env_id}'.\n"
            "- If you use terminal_states, all outgoing transitions from terminal states must have reward 0.\n"
        )

    try:
        spec, _rm = pipeline.run(
            task,
            output_path=args.output,
            complete=args.complete_missing_transitions,
            default_reward=args.default_reward,
            terminal_self_loop=args.terminal_self_loop,
            normalize_context=normalize_context,
            enforce_env_id=enforce_env_id,
            max_positive_reward_transitions=args.max_positive_reward_transitions,
            terminal_reward_must_be_zero=args.terminal_reward_must_be_zero,
        )
    except ValidationError as exc:
        sys.stderr.write(f"Validation failed: {exc}\n")
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 1

    sys.stdout.write(f"Generated RM spec '{spec.name}' (env={spec.env_id})\n")
    if args.complete_missing_transitions and spec.event_vocabulary and spec.states:
        total = len(spec.states) * len(spec.event_vocabulary)
        sys.stdout.write(
            f"Completed missing transitions to reach {total} transitions "
            f"(default_reward={args.default_reward})\n"
        )
    if args.output:
        sys.stdout.write(f"Saved to {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
