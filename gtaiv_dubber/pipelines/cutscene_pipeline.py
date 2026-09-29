from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from gtaiv_dubber.ai.demucs_adapter import load_vocal_stem
from gtaiv_dubber.ai.qwen_adapter import format_scene_context
from gtaiv_dubber.audio.alignment import fit_dialogue, fit_frame_count
from gtaiv_dubber.audio.diarization import (
    diarize_scene,
    neighbour_limits,
    speaker_profile_segments,
)
from gtaiv_dubber.audio.io import mono_mix, read_audio, write_pcm16
from gtaiv_dubber.audio.loudness import limit_peak, match_rms, style_hints
from gtaiv_dubber.audio.metadata import read_wav_metadata
from gtaiv_dubber.audio.mixing import render_dialogue_replacements
from gtaiv_dubber.audio.reference import build_speaker_reference, extract_reference
from gtaiv_dubber.audio.resampling import resample_audio
from gtaiv_dubber.audio.speakers import replace_overlapped_segments, simultaneous_source_segments
from gtaiv_dubber.audio.spatial import estimate_reference_weights
from gtaiv_dubber.audio.validation import validate_wav
from gtaiv_dubber.core.pipeline import BasePipeline, ProcessResult
from gtaiv_dubber.core.state import Status
from gtaiv_dubber.utils.atomic import atomic_copy_wav, atomic_publish
from gtaiv_dubber.utils.hashing import stable_key


# Risas, quejidos y muletillas: doblarlos no aporta nada y es donde peor sale.
# Qwen repitió una risa hasta agotar el contexto y CosyVoice la convirtió en
# ruido (BR_3, 27/09/2026). Se conserva la voz del actor original.
_VOCALIZATIONS = frozenset(
    {
        "ha", "hah", "haha", "heh", "hehe", "ah", "aah", "ahh", "oh", "ooh", "ohh",
        "uh", "uhh", "um", "umm", "hm", "hmm", "mm", "mmm", "argh", "ugh", "ow",
        "ouch", "whoa", "woah", "phew", "huh", "eh", "gah", "agh", "aw", "aww",
        "ew", "psst", "shh", "tsk", "hmph", "pfft",
    }
)


# Alucinación clásica de Whisper sobre música o silencio: "You You You You".
# Eran los "You" en huecos de 0 s que se comprimían hasta 32 veces (27/09/2026).
# Con comas o puntos entre medias ("No, no, no") es habla real y no se toca.
_REPEATED_YOU = re.compile(r"(?:\bYou\b\s*){3,}")


def strip_asr_hallucinations(text: str) -> str:
    """Quita la coletilla alucinada; devuelve "" si no queda nada real."""
    cleaned = _REPEATED_YOU.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.strip(" .!?,") in {"", "You"}:
        return ""
    return cleaned


def is_vocalization(text: str) -> bool:
    """Verdadero si la línea no tiene palabras que traducir."""
    words = re.findall(r"[a-z]+", text.lower())
    return all(word in _VOCALIZATIONS for word in words)


def runaway_translation(source: str, translation: str) -> bool:
    """La traducción se ha ido en bucle: muchas más palabras que el original.

    Qwen convirtió "It's not what you have to play" en "es lo que tienes que
    ser, es lo que tienes que hacer, es lo que..." (B_1, 27/09/2026). El
    español ocupa algo más que el inglés, pero no el doble y medio.
    """
    source_words = len(source.split())
    return len(translation.split()) > max(8, round(2.5 * source_words))


def same_words(first: str, second: str) -> bool:
    """La traducción repite el original: doblarla sólo cambiaría la voz."""
    def normalize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9áéíóúñü]+", text.lower())
    return normalize(first) == normalize(second)


def fit_badness(frames: int, slot: int, room: int, min_fill: float) -> float:
    """Cuánto habrá que deformar o cuánto sobra: menor es mejor.

    Comprimir se castiga el doble que quedarse corto, porque acelerar una voz
    se oye y un silencio al final de la frase casi no.
    """
    over = max(0.0, frames / max(room, 1) - 1.0)
    under = max(0.0, min_fill - frames / max(slot, 1))
    return 2.0 * over + under


def length_instruction(
    translation: str, spoken_seconds: float, slot_seconds: float, room_seconds: float
) -> str:
    """Pide el recorte o la ampliación con números, no con un "más corta".

    La instrucción anterior era sólo "Haz la frase más corta" y el modelo no
    tenía forma de saber cuánto.
    """
    words = max(1, len(translation.split()))
    if spoken_seconds > room_seconds:
        target = max(1, math.floor(words * room_seconds * 1.05 / max(spoken_seconds, 1e-6)))
        return (
            f"AJUSTE DE LONGITUD: tu versión anterior «{translation}» tiene {words} "
            f"palabras y al decirla dura {spoken_seconds:.1f} s, pero el personaje sólo "
            f"habla {room_seconds:.1f} s. Reescríbela con {target} palabras como máximo. "
            "Conserva la intención, el tono y los insultos; quita relleno y repeticiones."
        )
    target = max(words + 1, math.ceil(words * slot_seconds * 0.85 / max(spoken_seconds, 1e-6)))
    return (
        f"AJUSTE DE LONGITUD: tu versión anterior «{translation}» dura "
        f"{spoken_seconds:.1f} s al decirla, pero el personaje habla {slot_seconds:.1f} s. "
        f"Reescríbela más completa y natural, con unas {target} palabras, sin inventar "
        "información que no esté en el original."
    )


class CutscenePipeline(BasePipeline):
    mode = "cutscene"
    exact_frames = True

    def _segment_audio(
        self, vocal_mono: np.ndarray, speaker_sources: dict[int, np.ndarray]
    ):
        """La fuente separada del segmento cuando existe; si no, el stem vocal."""
        return lambda segment: speaker_sources.get(segment.get("speaker_source"), vocal_mono)

    def _segment_clips(
        self, segments: list[dict], audio_for_segment, sample_rate: int
    ) -> list[np.ndarray]:
        clips = []
        for segment in segments:
            audio = audio_for_segment(segment)
            first = max(0, round(float(segment["start"]) * sample_rate))
            last = min(len(audio), round(float(segment["end"]) * sample_rate))
            clips.append(np.asarray(audio[first:last], dtype=np.float32).reshape(-1))
        return clips

    def _diarize(
        self, relative_path: str, segments: list[dict], audio_for_segment, sample_rate: int
    ) -> tuple[list[dict], list[int], dict[int, float], dict[int, float]]:
        """Parte la escena donde cambia la voz y agrupa las líneas por personaje.

        Es información de apoyo: si el identificador falla, la escena se dobla
        igual con los segmentos de la ASR, que es el comportamiento anterior.
        """
        settings = self.config.get("diarization", {})
        if not segments or not bool(settings.get("enabled", True)):
            return segments, [0] * len(segments), {}, {}
        try:
            backend = self.models.speaker_embedding.prepare()
            ecapa = backend == "ecapa"
            threshold = float(
                settings.get("distance_threshold", 0.75) if ecapa
                else settings.get("spectral_distance_threshold", 0.32)
            )
            separation: dict[int, float] = {}
            split, labels, cohesion = diarize_scene(
                segments, audio_for_segment, sample_rate,
                separation_out=separation,
                embed=self.models.speaker_embedding.embed,
                threshold=threshold,
                change_threshold=float(settings.get("change_threshold", 0.70)) if ecapa else None,
                min_separation_ratio=(
                    None if ecapa else float(settings.get("spectral_separation_ratio", 8.0))
                ),
                max_speakers=int(settings.get("max_speakers", 8)),
                window_seconds=float(settings.get("window_seconds", 1.5)),
                hop_seconds=float(settings.get("hop_seconds", 0.75)),
                min_window_seconds=float(settings.get("min_window_seconds", 0.6)),
                min_turn_seconds=float(settings.get("min_turn_seconds", 0.6)),
                split_turns=bool(settings.get("split_turns", True)),
                min_cluster_seconds=float(settings.get("min_cluster_seconds", 0.0)),
                split_sentences=bool(settings.get("split_sentences", False)),
            )
            self.logger.info(
                "[SPEAKERS] %s: %d voces en %d intervenciones (%d tras cortar por voz, %s)",
                relative_path, len(set(labels)), len(segments), len(split), backend,
            )
            return split, labels, cohesion, separation
        except Exception as exc:
            self.logger.warning(
                "[SPEAKERS] %s: identificación de personajes no disponible: %s",
                relative_path, exc,
            )
            return segments, [0] * len(segments), {}, {}
        finally:
            self.models.unload_speaker_embedding()

    def _keep_original(self, file_id: int, index: int, row, reason: str) -> None:
        """No se dobla: sin pista nueva, la voz del actor original queda intacta."""
        self.logger.info(
            "[ALIGN] %s segmento %d: se conserva la voz original (%s)",
            row["relative_path"], index, reason,
        )
        self.db.update_segment(
            file_id, index, translation_es="", status=Status.DONE,
            duration_error_pct=0.0, error_message=f"voz original: {reason}",
        )

    def _speaker_profiles(
        self,
        key: str,
        segments: list[dict],
        labels: list[int],
        cohesion: dict[int, float],
        clips: list[np.ndarray],
        sample_rate: int,
        separation: dict[int, float] | None = None,
    ) -> tuple[dict[int, Path], dict[int, Path]]:
        """Una referencia vocal común por personaje, sólo cuando el grupo es fiable.

        Fiable significa que el grupo se parece a sí mismo claramente más que
        al personaje vecino más cercano. El umbral absoluto anterior (0,55) no
        servía: la escala de distancias cambia por escena (BR_1 vive en 0,6-0,8,
        BR_2 en 0,8-0,95) y rechazaba voces únicas bien separadas. Medido sobre
        20 grupos de BC_3, BR_1 y BR_2 (27/09/2026): los contrastes caen en
        -0,10..-0,06 (mezclas), 0,03..0,13 (dudosos) y 0,20..0,38 (una voz), con
        un hueco limpio entre 0,13 y 0,20.
        """
        settings = self.config.get("diarization", {})
        if not bool(settings.get("enabled", True)):
            return {}, {}
        target = float(settings.get("profile_target_seconds", 6.0))
        limit = float(settings.get("profile_max_intra_distance", 0.55))
        min_contrast = float(settings.get("profile_min_contrast", 0.15))
        separation = separation or {}
        chosen = speaker_profile_segments(
            segments, labels,
            target_seconds=target,
            max_windows=int(settings.get("profile_max_windows", 6)),
        )
        profiles: dict[int, Path] = {}
        spares: dict[int, Path] = {}
        for label, indexes in chosen.items():
            path = self.runtime["references"] / "cutscenes" / f"{key}_speaker{label:02d}.wav"
            try:
                built, pooled = build_speaker_reference(
                    [clips[index] for index in indexes], sample_rate, path,
                    target_seconds=target,
                    max_seconds=float(settings.get("profile_max_seconds", 12.0)),
                    minimum_seconds=float(settings.get("profile_min_seconds", 2.0)),
                )
            except ValueError as exc:
                self.logger.info(
                    "[SPEAKERS] PERSONAJE %d sin referencia común (%s); se usa cada línea",
                    label + 1, exc,
                )
                continue
            # Siempre se guarda como reserva: si un fragmento es demasiado corto
            # para su propia referencia, la voz del grupo es mejor que perder la línea.
            spares[label] = built
            spread = cohesion.get(label)
            # Dos motivos distintos para no agrupar, y conviene distinguirlos en
            # el log: una sola intervención no da promedio de cohesión que valga.
            if len(indexes) < 2:
                self.logger.info(
                    "[SPEAKERS] PERSONAJE %d: una sola intervención; "
                    "se clona desde su propio audio",
                    label + 1,
                )
                continue
            neighbour = separation.get(label)
            if spread is not None and neighbour is not None:
                contrast = neighbour - spread
                if contrast < min_contrast:
                    self.logger.info(
                        "[SPEAKERS] PERSONAJE %d: contraste %.2f < %.2f (dentro %.2f, "
                        "vecino %.2f); se clona cada línea por separado",
                        label + 1, contrast, min_contrast, spread, neighbour,
                    )
                    continue
            elif spread is not None and spread > limit:
                # Sin medida del vecino se mantiene el criterio antiguo.
                self.logger.info(
                    "[SPEAKERS] PERSONAJE %d: grupo poco cohesionado (%.2f > %.2f); "
                    "se clona cada línea por separado",
                    label + 1, spread, limit,
                )
                continue
            profiles[label] = built
            self.logger.info(
                "[SPEAKERS] PERSONAJE %d: referencia de %.1fs desde %d intervenciones",
                label + 1, pooled, len(indexes),
            )
        return profiles, spares

    def _process(self, row, source: Path, destination: Path, digest: str) -> ProcessResult:
        file_id = int(row["id"])
        self.db.update_file(file_id, status=Status.ANALYZING, error_message=None)
        metadata = read_wav_metadata(source)
        original, sample_rate = read_audio(source)
        analysis_mono = mono_mix(original)
        key = stable_key(self.mode, row["relative_path"], digest)[:24]
        analysis_path = self.runtime["tmp"] / f"analysis_{key}.wav"
        write_pcm16(analysis_path, analysis_mono, sample_rate)
        self.logger.info("[SEPARATE] %s", row["relative_path"])
        vocals_path, _background_path = self.models.demucs.separate(
            analysis_mono, sample_rate, f"{row['relative_path']}:{digest}"
        )
        vocal_mono = load_vocal_stem(vocals_path, sample_rate, metadata.frames)
        self.models.unload_cosyvoice()
        self.logger.info("[ASR] %s", row["relative_path"])
        transcript_segments = self.models.whisper.transcribe(analysis_path, word_timestamps=True)
        segments = [
            {
                "start": max(0.0, item.start),
                "end": min(metadata.duration, item.end),
                "text": item.text,
                "confidence": item.confidence,
                "words": item.words,
                "status": Status.TRANSCRIBED,
            }
            for item in transcript_segments if item.end > item.start
        ]
        for segment in segments:
            segment["text"] = strip_asr_hallucinations(segment["text"])
        # Sin texto real no hay nada que doblar: el audio original queda intacto.
        segments = [segment for segment in segments if segment["text"]]
        speaker_sources: dict[int, np.ndarray] = {}
        review_overlap = False
        separation_config = self.config.get("speaker_separation", {})
        if segments and bool(separation_config.get("enabled", True)):
            try:
                self.logger.info("[SPEAKERS] Buscando voces simultáneas en %s", row["relative_path"])
                separated = self.models.speaker_separator.separate(
                    vocal_mono, sample_rate, f"{row['relative_path']}:{digest}"
                )
                self.models.unload_speaker_separator()
                by_source: list[list[dict]] = []
                for source_index in range(min(2, separated.shape[1])):
                    separated_path = self.runtime["tmp"] / f"speakers_{key}_{source_index}.wav"
                    write_pcm16(separated_path, separated[:, source_index], sample_rate)
                    source_transcript = self.models.whisper.transcribe(separated_path)
                    separated_path.unlink(missing_ok=True)
                    by_source.append(
                        [
                            {
                                "start": max(0.0, item.start),
                                "end": min(metadata.duration, item.end),
                                "text": item.text,
                                "confidence": item.confidence,
                                "status": Status.TRANSCRIBED,
                            }
                            for item in source_transcript if item.end > item.start
                        ]
                    )
                simultaneous = simultaneous_source_segments(
                    by_source,
                    separated,
                    sample_rate,
                    min_overlap_seconds=float(separation_config.get("min_overlap_seconds", 0.12)),
                    min_source_rms_ratio=float(
                        separation_config.get("min_source_rms_ratio", 0.12)
                    ),
                    duplicate_similarity=float(
                        separation_config.get("duplicate_similarity", 0.72)
                    ),
                    activity_threshold=float(
                        separation_config.get("activity_threshold", 0.15)
                    ),
                    max_source_correlation=float(
                        separation_config.get("max_source_correlation", 0.25)
                    ),
                )
                if simultaneous:
                    segments = replace_overlapped_segments(segments, simultaneous)
                    speaker_sources = {
                        index: separated[:, index] for index in range(min(2, separated.shape[1]))
                    }
                    review_overlap = True
                    self.logger.warning(
                        "[SPEAKERS] %s: %d intervenciones simultáneas separadas",
                        row["relative_path"], len(simultaneous),
                    )
            except Exception as exc:
                self.logger.warning(
                    "[SPEAKERS] %s: separación no disponible; se conserva revisión manual: %s",
                    row["relative_path"], exc,
                )
            finally:
                self.models.unload_speaker_separator()
        self.models.unload_whisper()
        audio_for_segment = self._segment_audio(vocal_mono, speaker_sources)
        segments, labels, cohesion, separation = self._diarize(
            row["relative_path"], segments, audio_for_segment, sample_rate
        )
        clips = self._segment_clips(segments, audio_for_segment, sample_rate)
        speaker_names = [f"PERSONAJE {label + 1}" for label in labels]
        for segment, name in zip(segments, speaker_names):
            segment["speaker_label"] = name
        self.db.replace_segments(file_id, segments)
        transcript = " ".join(item["text"] for item in segments)
        self.db.update_file(file_id, transcript_en=transcript, status=Status.TRANSCRIBED)
        if not segments:
            original_meta = metadata

            def validator(path: Path):
                result = validate_wav(path, original_meta, exact_frames=True)
                if not result.valid:
                    raise ValueError(result.reason)

            atomic_copy_wav(source, destination, self.runtime["tmp"], validator)
            self.db.update_file(
                file_id, sha256=digest, config_hash=self.config_hash,
                status=Status.DONE, error_message="Sin diálogo detectable; original conservado",
            )
            return ProcessResult(Status.DONE, row["relative_path"], "Sin diálogo")
        diarization_config = self.config.get("diarization", {})
        limits = neighbour_limits(
            segments, float(diarization_config.get("neighbour_guard_seconds", 0.08))
        )
        profiles, spares = self._speaker_profiles(
            key, segments, labels, cohesion, clips, sample_rate, separation
        )
        reference_max = float(self.config["cutscenes"].get("reference_max_seconds", 20.0))
        multi_speaker = len(set(labels)) > 1
        scene_entries = list(zip(speaker_names, [item["text"] for item in segments]))
        context_lines = int(diarization_config.get("context_window_lines", 4))
        cut = self.config["cutscenes"]
        max_speedup = float(cut.get("max_speedup", 1.2))
        hard_max_speedup = float(cut.get("hard_max_speedup", 1.35))
        max_overrun = float(cut.get("max_overrun_seconds", 0.75))
        min_fill = float(cut.get("min_fill_ratio", 0.55))
        regenerations = int(cut.get("length_regeneration_attempts", 2))
        min_dub = float(cut.get("min_dub_seconds", 0.35))
        scene_seconds = metadata.frames / sample_rate
        failures = 0
        review_duration = False
        translations: list[str] = []
        dialogue_tracks: list[tuple[np.ndarray, int, int, np.ndarray]] = []
        for index, segment in enumerate(segments):
            if index and segment["start"] < segments[index - 1]["end"]:
                review_overlap = True
            try:
                previous = segments[index - 1]["text"] if index else ""
                following = segments[index + 1]["text"] if index + 1 < len(segments) else ""
                duration = segment["end"] - segment["start"]
                scene_context = (
                    format_scene_context(scene_entries, index, context_lines)
                    if multi_speaker else ""
                )
                speaker_label = speaker_names[index] if multi_speaker else ""
                if duration < min_dub:
                    self._keep_original(file_id, index, row, f"hueco de {duration:.2f} s")
                    translations.append(segment["text"])
                    continue
                if is_vocalization(segment["text"]):
                    self._keep_original(file_id, index, row, "vocalización sin palabras")
                    translations.append(segment["text"])
                    continue
                self.logger.info(
                    "[TRANSLATE] %s segmento %d (%s)",
                    row["relative_path"], index, speaker_names[index],
                )
                translation = self.models.qwen.adapt(
                    segment["text"], duration, previous_line=previous, next_line=following,
                    relative_path=row["relative_path"],
                    speaker_label=speaker_label, scene_context=scene_context,
                )
                if runaway_translation(segment["text"], translation):
                    self.logger.warning(
                        "[TRANSLATE] %s segmento %d: traducción desbocada (%d palabras "
                        "para %d); se pide otra vez",
                        row["relative_path"], index,
                        len(translation.split()), len(segment["text"].split()),
                    )
                    translation = self.models.qwen.adapt(
                        segment["text"], duration, previous_line=previous, next_line=following,
                        relative_path=row["relative_path"],
                        length_instruction=(
                            "Traduce solamente esta línea, una sola vez, sin repetir "
                            "frases ni continuar la conversación."
                        ),
                        speaker_label=speaker_label, scene_context=scene_context,
                    )
                    if runaway_translation(segment["text"], translation):
                        self._keep_original(file_id, index, row, "traducción desbocada")
                        translations.append(segment["text"])
                        continue
                if same_words(translation, segment["text"]):
                    self._keep_original(file_id, index, row, "la traducción repite el original")
                    translations.append(segment["text"])
                    continue
                translations.append(translation)
                speaker_source = segment.get("speaker_source")
                source_audio = speaker_sources.get(speaker_source, vocal_mono)
                reference_path = profiles.get(labels[index])
                if reference_path is None:
                    reference_path = (
                        self.runtime["references"] / "cutscenes" / f"{key}_{index:04d}.wav"
                    )
                    reference_padding = (
                        0.15 if speaker_source is not None
                        else float(self.config["cutscenes"].get("reference_padding_seconds", 0.75))
                    )
                    try:
                        extract_reference(
                            source_audio, sample_rate, segment["start"], segment["end"],
                            reference_padding,
                            reference_path,
                            limit_start=limits[index][0], limit_end=limits[index][1],
                            max_seconds=reference_max,
                        )
                    except ValueError:
                        # Tres escalones, de mejor a peor: la voz agrupada del
                        # mismo personaje conserva el timbre; si tampoco existe,
                        # se suelta el límite con los vecinos, que puede colar algo
                        # de la voz contigua pero salva la línea; perderla es peor.
                        spare = spares.get(labels[index])
                        if spare is not None:
                            self.logger.info(
                                "[SPEAKERS] %s segmento %d: fragmento muy corto; "
                                "se usa la voz agrupada de PERSONAJE %d",
                                row["relative_path"], index, labels[index] + 1,
                            )
                            reference_path = spare
                        else:
                            extract_reference(
                                source_audio, sample_rate, segment["start"], segment["end"],
                                reference_padding,
                                reference_path,
                                max_seconds=reference_max,
                            )
                            self.logger.warning(
                                "[SPEAKERS] %s segmento %d: sin voz agrupada; referencia "
                                "ampliada sobre los vecinos, revisar el timbre",
                                row["relative_path"], index,
                            )
                first = max(0, round(segment["start"] * sample_rate))
                last = min(metadata.frames, round(segment["end"] * sample_rate))
                original_region = original[first:last]
                vocal_region = vocal_mono[first:last]
                level_reference = source_audio[first:last]
                hints = style_hints(original_region, sample_rate)
                slot_frames = max(1, last - first)
                room_end = min(limits[index][1], segment["end"] + max_overrun, scene_seconds)
                room_seconds = max(duration, room_end - segment["start"])
                room_frames = max(slot_frames, round(room_seconds * sample_rate))

                def voice(text: str) -> np.ndarray:
                    produced, produced_rate = self.synthesize_with_oom_retry(
                        reference_path, text, duration, hints
                    )
                    return resample_audio(produced, produced_rate, sample_rate)

                generated = voice(translation)
                badness = fit_badness(len(generated), slot_frames, room_frames, min_fill)
                for attempt in range(regenerations):
                    too_long = len(generated) > room_frames * max_speedup
                    too_short = len(generated) < slot_frames * min_fill
                    if not (too_long or too_short):
                        break
                    candidate_text = self.models.qwen.adapt(
                        segment["text"], duration, previous_line=previous, next_line=following,
                        relative_path=row["relative_path"],
                        length_instruction=length_instruction(
                            translation, len(generated) / sample_rate, duration, room_seconds
                        ),
                        speaker_label=speaker_label, scene_context=scene_context,
                    )
                    candidate = voice(candidate_text)
                    candidate_badness = fit_badness(
                        len(candidate), slot_frames, room_frames, min_fill
                    )
                    self.logger.info(
                        "[ALIGN] %s segmento %d: reescritura %d, voz %.2fs -> %.2fs "
                        "(hueco %.2fs, margen %.2fs)",
                        row["relative_path"], index, attempt + 1,
                        len(generated) / sample_rate, len(candidate) / sample_rate,
                        duration, room_seconds,
                    )
                    if candidate_badness < badness:
                        translation, generated, badness = (
                            candidate_text, candidate, candidate_badness
                        )
                translations[-1] = translation
                # El volumen se iguala antes de encajar: el silencio de relleno
                # bajaría el RMS y match_rms subiría la voz hasta +12 dB.
                generated = match_rms(generated, level_reference)
                aligned, tempo, trimmed = fit_dialogue(
                    generated, sample_rate, duration, room_seconds,
                    self.runtime["tmp"] / f"{key}_{index}",
                    hard_max_speedup=hard_max_speedup,
                )
                self.logger.info(
                    "[ALIGN] %s segmento %d: voz %.2fs, hueco %.2fs, margen %.2fs -> tempo %.2f%s",
                    row["relative_path"], index, len(generated) / sample_rate,
                    duration, room_seconds, tempo, ", RECORTADA" if trimmed else "",
                )
                weights = estimate_reference_weights(original_region, vocal_region)
                dialogue_tracks.append((aligned[:, 0], first, first + len(aligned), weights))
                needs_review = trimmed or tempo > max_speedup
                segment_status = (
                    Status.REVIEW_DURATION if needs_review
                    else Status.REVIEW_OVERLAP if review_overlap else Status.DONE
                )
                review_duration = review_duration or needs_review
                # En cutscenes esta columna mide ahora la deformación aplicada
                # (0 = voz a su velocidad natural), que es lo que se oye.
                self.db.update_segment(
                    file_id, index, translation_es=translation,
                    reference_path=str(reference_path), generated_duration=len(generated) / sample_rate,
                    duration_error_pct=(tempo - 1.0) * 100.0, status=segment_status,
                    error_message="cola recortada" if trimmed else None,
                )
            except Exception as exc:
                failures += 1
                translations.append("")
                self.db.update_segment(
                    file_id, index, status=Status.REVIEW_AUDIO, error_message=str(exc)
                )
                self.logger.error("[REVIEW] %s segmento %d: %s", row["relative_path"], index, exc)
                continue
        output = render_dialogue_replacements(
            original,
            vocal_mono,
            dialogue_tracks,
            attenuation=float(self.config["cutscenes"].get("vocal_attenuation", 1.0)),
        )
        output = fit_frame_count(output, metadata.frames)
        output = limit_peak(output, self.config["audio"]["peak_limit_dbfs"])

        def producer(path: Path):
            write_pcm16(path, output, metadata.sample_rate)

        def validator(path: Path):
            result = validate_wav(path, metadata, exact_frames=True)
            if not result.valid:
                raise ValueError(result.reason)

        atomic_publish(destination, self.runtime["tmp"], producer, validator)
        if failures == len(segments):
            status = Status.FALLBACK
        elif failures:
            status = Status.REVIEW_AUDIO
        elif review_overlap:
            status = Status.REVIEW_OVERLAP
        elif review_duration:
            status = Status.REVIEW_DURATION
        else:
            status = Status.DONE
        self.db.update_file(
            file_id, sha256=digest, config_hash=self.config_hash,
            translation_es=" ".join(filter(None, translations)), status=status,
            error_message=f"{failures} segmentos conservados en inglés" if failures else None,
        )
        analysis_path.unlink(missing_ok=True)
        self.logger.info("[EXPORT] %s", row["relative_path"])
        return ProcessResult(status, row["relative_path"])
