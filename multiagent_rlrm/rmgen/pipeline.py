import json
from pathlib import Path
from typing import Mapping, Optional, Tuple

from multiagent_rlrm.rmgen.exporter import build_reward_machine, export_spec_to_file
from multiagent_rlrm.rmgen.completion import complete_missing_transitions
from multiagent_rlrm.rmgen.spec import RMSpec
from multiagent_rlrm.rmgen.validator import (
    ValidationError,
    ensure_deterministic,
    validate_schema,
    validate_semantics,
)


class RMGenerationPipeline:
    """
    Orchestrates text -> RM generation with validation and export.
    """

    def __init__(self, provider):
        self.provider = provider

    def _load_json(self, raw: str, task: str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if hasattr(self.provider, "repair"):
                repaired = self.provider.repair(task, raw)
                return json.loads(repaired)
            raise

    def run(
        self,
        task: str,
        output_path: Optional[Path] = None,
        *,
        complete: bool = False,
        default_reward: float = 0.0,
        terminal_self_loop: bool = True,
        normalize_context: Optional[Mapping[str, object]] = None,
        enforce_env_id: Optional[str] = None,
        max_positive_reward_transitions: int = None,
        terminal_reward_must_be_zero: bool = True,
    ) -> Tuple[RMSpec, object]:
        raw = self.provider.generate(task)
        try:
            payload = self._load_json(raw, task)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"Provider returned invalid JSON after repair: {exc}"
            ) from exc

        if enforce_env_id is not None:
            expected = str(enforce_env_id).strip().lower()
            raw_env_id = payload.get("env_id")
            raw_env_id_norm = (
                str(raw_env_id).strip().lower() if raw_env_id is not None else None
            )
            if raw_env_id is None or raw_env_id_norm != expected:
                old_display = raw_env_id if raw_env_id is not None else "<missing>"
                print(
                    f"Warning: overriding env_id from '{old_display}' to '{expected}' "
                    f"because --context {expected} is set"
                )
                payload["env_id"] = expected

        spec = RMSpec.from_dict(payload)
        if normalize_context is not None:
            from multiagent_rlrm.rmgen.normalize import normalize_rmspec_events

            spec = normalize_rmspec_events(spec, normalize_context)
        if (
            enforce_env_id is not None
            and str(enforce_env_id).strip().lower() == "officeworld"
        ):
            from multiagent_rlrm.rmgen.normalize import (
                autofix_rmspec_states_for_officeworld,
                autofix_terminal_reward_violations_for_officeworld,
            )

            spec = autofix_rmspec_states_for_officeworld(spec)
            if terminal_reward_must_be_zero:
                spec = autofix_terminal_reward_violations_for_officeworld(spec)
        if complete:
            spec, _report = complete_missing_transitions(
                spec,
                default_reward=default_reward,
                terminal_self_loop=terminal_self_loop,
            )
        validate_schema(spec)
        ensure_deterministic(spec)
        validate_semantics(
            spec,
            max_positive_reward_transitions=max_positive_reward_transitions,
            terminal_reward_must_be_zero=terminal_reward_must_be_zero,
        )

        if output_path:
            export_spec_to_file(spec, output_path)

        rm = build_reward_machine(spec)
        return spec, rm
