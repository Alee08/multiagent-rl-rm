import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Optional
from multiagent_rlrm.rmgen.validator import ValidationError


class MockLLMClient:
    """
    Mock LLM provider that returns a JSON RM spec from fixture files.

    - If fixture_path is provided, it is always used.
    - Otherwise, task_fixtures maps task text -> fixture path.
    """

    def __init__(
        self,
        fixture_path: Optional[Path] = None,
        task_fixtures: Optional[Dict[str, Path]] = None,
    ):
        self.fixture_path = Path(fixture_path) if fixture_path else None
        self.task_fixtures: Dict[str, Path] = {
            k: Path(v) for k, v in (task_fixtures or {}).items()
        }

    def generate(self, task: str):
        """
        Returns the raw JSON string for the task.
        """
        path = self.fixture_path or self.task_fixtures.get(task)
        if not path:
            raise ValueError(f"No fixture configured for task '{task}'")
        text = Path(path).read_text(encoding="utf-8")
        # Validate JSON loads early to catch malformed fixtures.
        json.loads(text)
        return text


class _ChatClient:
    """
    Minimal OpenAI-compatible chat client using urllib from stdlib.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def _post(self, payload: Dict[str, object]) -> Dict[str, object]:
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:  # type: ignore[arg-type]
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8") if hasattr(exc, "read") else ""
            raise ValidationError(f"HTTP error from provider: {exc.code} {text}")

    def chat(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        try:
            return self._post(payload)
        except ValidationError as err:
            # Fallback: retry without response_format for backends that do not support it.
            if "response_format" in str(err):
                payload.pop("response_format", None)
                return self._post(payload)
            raise


SCHEMA_INSTRUCTIONS = """You are a function that returns ONLY JSON for a Reward Machine specification.
Required top-level keys:
- name (string)
- env_id (string)
- version (string)
- states (list of unique strings)
- initial_state (string, must be in states)
- terminal_states (list of strings, subset of states)
- event_vocabulary (list of unique strings)
- transitions (list of objects with keys: from_state, event, to_state, reward (number))
- notes (string, optional)
The JSON must be valid and contain no extra text.
"""

EXAMPLE_JSON = {
    "name": "example_task",
    "env_id": "gridworld",
    "version": "1.0",
    "states": ["q0", "q1", "q2"],
    "initial_state": "q0",
    "terminal_states": ["q2"],
    "event_vocabulary": ["at(A)", "at(G)"],
    "transitions": [
        {"from_state": "q0", "event": "at(A)", "to_state": "q1", "reward": 0.0},
        {"from_state": "q1", "event": "at(G)", "to_state": "q2", "reward": 1.0},
    ],
    "notes": "go to A then goal",
}


def _build_prompt(task: str) -> list:
    example_text = json.dumps(EXAMPLE_JSON, ensure_ascii=True)
    system = (
        SCHEMA_INSTRUCTIONS
        + "Return only JSON. Include only events that appear in transitions. Example:\n"
        + example_text
    )
    user = f"Task description:\n{task}\nReturn only the JSON spec."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class OpenAICompatLLMClient:
    """
    OpenAI-compatible provider (e.g., Ollama / vLLM / local gateways).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llama3.1:8b",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.client = _ChatClient(
            base_url=base_url, model=model, api_key=api_key, temperature=temperature
        )

    def generate(self, task: str):
        messages = _build_prompt(task)
        response = self.client.chat(messages)
        if "error" in response:
            raise ValidationError(f"Provider error: {response['error']}")
        content = response["choices"][0]["message"]["content"]
        return extract_json_object(content)

    def repair(self, task: str, previous_output: str):
        prompt = (
            "The previous output was not valid JSON. Return ONLY valid JSON that conforms "
            "to the RMSpec schema. Fix the issues and do not add explanations. Previous output:\n"
            f"{previous_output}"
        )
        messages = _build_prompt(task)
        messages.append({"role": "assistant", "content": previous_output})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat(messages)
        if "error" in response:
            raise ValidationError(f"Provider error: {response['error']}")
        return extract_json_object(response["choices"][0]["message"]["content"])


class OpenAILLMClient(OpenAICompatLLMClient):
    """
    OpenAI hosted provider; defaults to OpenAI base URL and requires API key.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for provider 'openai'")
        super().__init__(
            base_url="https://api.openai.com/v1",
            model=model,
            api_key=key,
            temperature=temperature,
        )


def extract_json_object(text: str) -> str:
    """
    Extract the first JSON object found in text. Strips code fences and
    returns substring from first '{' to last '}'.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove fences like ```json ... ``` or ``` ...
        cleaned = cleaned.strip("`")
        parts = cleaned.split("\n", 1)
        cleaned = parts[1] if len(parts) > 1 else ""
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        snippet = cleaned[:200]
        raise ValidationError(f"Could not find JSON object in content: {snippet}")
    return cleaned[start : end + 1]
