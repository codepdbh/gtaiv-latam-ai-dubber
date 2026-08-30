from __future__ import annotations

import json
import urllib.error
import urllib.request


SYSTEM_PROMPT = """Eres un adaptador profesional de doblaje latinoamericano de videojuegos.
Traduce diálogos del inglés al español latinoamericano neutro y natural.
No traduzcas literalmente. Mantén intención, personalidad, emoción, humor,
sarcasmo, insultos, vulgaridad, lenguaje callejero, nombres propios y contexto.
La frase debe poder pronunciarse aproximadamente en TARGET_DURATION segundos.
No añadas explicaciones ni comillas. Devuelve únicamente la frase final doblable."""


class QwenAdapter:
    def __init__(self, config: dict, glossary: dict[str, str]):
        self.model = config.get("model", "qwen3:8b")
        self.host = config.get("host", "http://127.0.0.1:11434").rstrip("/")
        self.timeout = int(config.get("timeout_seconds", 180))
        self.glossary = glossary

    def _request(self, messages: list[dict]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "keep_alive": 0,
                "options": {"temperature": 0.25, "top_p": 0.9},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama/Qwen no disponible: {exc}") from exc
        text = str(body.get("message", {}).get("content", "")).strip()
        if not text:
            raise RuntimeError("Qwen devolvió una traducción vacía")
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        return text

    def adapt(
        self,
        current_line: str,
        target_duration: float,
        *,
        previous_line: str = "",
        next_line: str = "",
        relative_path: str = "",
        length_instruction: str = "",
    ) -> str:
        glossary = "\n".join(f"- {key}: {value}" for key, value in self.glossary.items())
        prompt = f"""TARGET_DURATION: {target_duration:.3f} segundos
ARCHIVO: {relative_path}
LÍNEA ANTERIOR: {previous_line or '(ninguna)'}
LÍNEA ACTUAL: {current_line}
LÍNEA SIGUIENTE: {next_line or '(ninguna)'}
GLOSARIO OBLIGATORIO:
{glossary or '(vacío)'}
{length_instruction}
Traduce solamente LÍNEA ACTUAL."""
        return self._request(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        )

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as response:
                body = json.loads(response.read().decode("utf-8"))
            names = {item.get("name", "") for item in body.get("models", [])}
            return self.model in names or any(name.startswith(self.model + ":") for name in names)
        except Exception:
            return False
