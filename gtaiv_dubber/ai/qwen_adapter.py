from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Eres un adaptador profesional de doblaje latinoamericano de videojuegos.
Traduce diálogos del inglés al español latinoamericano neutro y natural.
No traduzcas literalmente. Mantén intención, personalidad, emoción, humor,
sarcasmo, insultos, vulgaridad, lenguaje callejero, nombres propios y contexto.
La frase debe poder pronunciarse aproximadamente en TARGET_DURATION segundos.
No añadas explicaciones ni comillas. Devuelve únicamente la frase final doblable."""

SCENE_PROMPT = """La línea pertenece a una escena con varios personajes.
Mantén coherente a cada personaje durante toda la escena: el mismo género
gramatical al hablar de sí mismo, el mismo tratamiento (tú/usted) hacia cada
interlocutor y el mismo registro. Usa la ESCENA sólo como contexto: nunca la
traduzcas ni la repitas."""


def format_scene_context(
    entries: list[tuple[str, str]], current_index: int, window: int = 4
) -> str:
    """Diálogo vecino etiquetado por personaje, marcando la línea que toca traducir."""
    first = max(0, current_index - window)
    last = min(len(entries), current_index + window + 1)
    return "\n".join(
        f"{'>>' if index == current_index else '  '} {speaker}: {text}"
        for index, (speaker, text) in enumerate(entries[first:last], first)
    )


class QwenAdapter:
    def __init__(self, config: dict, glossary: dict[str, str]):
        self.model = config.get("model", "qwen3:8b")
        self.host = config.get("host", "http://127.0.0.1:11434").rstrip("/")
        self.timeout = int(config.get("timeout_seconds", 180))
        self.attempts = max(1, int(config.get("request_attempts", 3)))
        self.retry_pause = float(config.get("retry_pause_seconds", 5.0))
        self.max_tokens = int(config.get("max_output_tokens", 512))
        self.glossary = glossary

    def _request(self, messages: list[dict]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "keep_alive": 0,
                "options": {
                    "temperature": 0.25,
                    "top_p": 0.9,
                    # Una réplica doblada cabe de sobra aquí; el tope sólo muerde
                    # cuando el modelo se desboca y consume todo el contexto.
                    "num_predict": self.max_tokens,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=payload, headers={"Content-Type": "application/json"}
        )
        # Medido el 27/09/2026 en BR_3 segmento 23: las líneas normales tardan
        # 15-20 s y esa tardaba más de 180 s en los tres intentos, con el modelo
        # ya residente en VRAM. No es una recarga lenta ni contención de GPU: es
        # generación desbocada en una línea concreta. El reintento cubre el fallo
        # transitorio; num_predict acota el caso patológico.
        body = None
        for attempt in range(1, self.attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == self.attempts:
                    raise RuntimeError(
                        f"Ollama/Qwen no disponible tras {attempt} intentos: {exc}"
                    ) from exc
                logger.warning(
                    "[TRANSLATE] Ollama falló (intento %d de %d): %s; se reintenta",
                    attempt, self.attempts, exc,
                )
                time.sleep(self.retry_pause)
        generated = int(body.get("eval_count", 0) or 0)
        if generated >= self.max_tokens:
            logger.warning(
                "[TRANSLATE] la traducción agotó el tope de %d tokens "
                "(contexto %d); revisar la línea",
                self.max_tokens, int(body.get("prompt_eval_count", 0) or 0),
            )
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
        speaker_label: str = "",
        scene_context: str = "",
    ) -> str:
        glossary = "\n".join(f"- {key}: {value}" for key, value in self.glossary.items())
        scene: list[str] = []
        if scene_context:
            scene.append(f"ESCENA:\n{scene_context}")
        if speaker_label:
            scene.append(f"PERSONAJE ACTUAL: {speaker_label}")
        prompt = "\n".join(
            [
                f"TARGET_DURATION: {target_duration:.3f} segundos",
                f"ARCHIVO: {relative_path}",
                *scene,
                f"LÍNEA ANTERIOR: {previous_line or '(ninguna)'}",
                f"LÍNEA ACTUAL: {current_line}",
                f"LÍNEA SIGUIENTE: {next_line or '(ninguna)'}",
                "GLOSARIO OBLIGATORIO:",
                glossary or "(vacío)",
                length_instruction,
                "Traduce solamente LÍNEA ACTUAL.",
            ]
        )
        system = SYSTEM_PROMPT if not scene else f"{SYSTEM_PROMPT}\n{SCENE_PROMPT}"
        return self._request(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        )

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as response:
                body = json.loads(response.read().decode("utf-8"))
            names = {item.get("name", "") for item in body.get("models", [])}
            return self.model in names or any(name.startswith(self.model + ":") for name in names)
        except Exception:
            return False
