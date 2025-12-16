from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TransitionSpec:
    from_state: str
    event: str
    to_state: str
    reward: float

    @staticmethod
    def _coerce_reward(raw: Any, data: Dict[str, Any]) -> float:
        """
        Accept numeric rewards or strings with optional leading 'r'.
        Examples: 1, 1.0, "1", "r1", "r0.5".
        """
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            trimmed = raw.strip()
            if trimmed.lower().startswith("r"):
                trimmed = trimmed[1:]
            try:
                return float(trimmed)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid reward value '{raw}' in transition {data}"
                ) from exc
        raise ValueError(f"Invalid reward type '{type(raw)}' in transition {data}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransitionSpec":
        return cls(
            from_state=data["from_state"],
            event=data["event"],
            to_state=data["to_state"],
            reward=cls._coerce_reward(data["reward"], data),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state,
            "event": self.event,
            "to_state": self.to_state,
            "reward": self.reward,
        }


@dataclass
class RMSpec:
    name: str
    env_id: str
    version: str
    states: List[str]
    initial_state: str
    terminal_states: List[str]
    event_vocabulary: List[str]
    transitions: List[TransitionSpec]
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RMSpec":
        transitions = [TransitionSpec.from_dict(t) for t in data.get("transitions", [])]
        env_id = data["env_id"].strip().lower()
        return cls(
            name=data["name"],
            env_id=env_id,
            version=data["version"],
            states=list(data["states"]),
            initial_state=data["initial_state"],
            terminal_states=list(data.get("terminal_states", [])),
            event_vocabulary=list(data["event_vocabulary"]),
            transitions=transitions,
            notes=data.get("notes"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "env_id": self.env_id,
            "version": self.version,
            "states": self.states,
            "initial_state": self.initial_state,
            "terminal_states": self.terminal_states,
            "event_vocabulary": self.event_vocabulary,
            "transitions": [t.to_dict() for t in self.transitions],
            "notes": self.notes,
        }

    def as_transition_map(self) -> Dict[tuple, tuple]:
        """
        Build the mapping {(state, event): (next_state, reward)} expected by RewardMachine.
        """
        transition_map = {}
        for t in self.transitions:
            transition_map[(t.from_state, t.event)] = (t.to_state, t.reward)
        return transition_map
