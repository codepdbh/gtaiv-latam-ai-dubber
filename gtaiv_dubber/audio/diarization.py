from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _mel_filterbank(
    n_filters: int, n_fft: int, sample_rate: int, low_hz: float = 60.0, high_hz: float = 7600.0
) -> np.ndarray:
    def to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    high_hz = min(high_hz, sample_rate / 2.0)
    low_hz = min(low_hz, high_hz / 2.0)
    points = to_hz(np.linspace(to_mel(low_hz), to_mel(high_hz), n_filters + 2))
    bins = np.clip(np.floor((n_fft + 1) * points / sample_rate).astype(int), 0, n_fft // 2)
    filters = np.zeros((n_filters, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_filters):
        left, center, right = (int(bins[index]), int(bins[index + 1]), int(bins[index + 2]))
        center = max(center, left + 1)
        right = min(max(right, center + 1), n_fft // 2)
        if left >= center or center >= right:
            continue
        filters[index, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        filters[index, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
    return filters


def _framed(signal: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))
    count = 1 + (len(signal) - frame_length) // hop
    offsets = hop * np.arange(count)[:, None]
    return signal[offsets + np.arange(frame_length)[None, :]]


def spectral_embedding(clip: np.ndarray, sample_rate: int, n_filters: int = 40) -> np.ndarray:
    """Huella de timbre por estadísticas log-mel, sin depender de modelos externos.

    Es el respaldo cuando ECAPA no está disponible: distingue voces claramente
    distintas dentro de una misma escena, que es lo que necesita el agrupamiento.
    """
    signal = np.asarray(clip, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return np.zeros(2 * n_filters, dtype=np.float32)
    frame_length = 2 ** int(round(np.log2(max(256.0, 0.025 * sample_rate))))
    frame_length = min(frame_length, 2 ** int(np.log2(max(len(signal), 256))))
    hop = max(1, frame_length // 2)
    frames = _framed(signal, frame_length, hop) * np.hanning(frame_length).astype(np.float32)
    spectrum = np.square(np.abs(np.fft.rfft(frames, axis=1)))
    mel = np.log(spectrum @ _mel_filterbank(n_filters, frame_length, sample_rate).T + 1e-10)
    energy = mel.mean(axis=1)
    if len(energy) > 4:
        loud = energy >= (float(energy.max()) - 6.0)
        if loud.any():
            mel = mel[loud]
    return np.concatenate([mel.mean(axis=0), mel.std(axis=0)]).astype(np.float32)


def spectral_embeddings(clips: list[np.ndarray], sample_rate: int) -> np.ndarray:
    return np.stack([spectral_embedding(clip, sample_rate) for clip in clips])


def normalize_embeddings(embeddings: np.ndarray, *, center: bool = False) -> np.ndarray:
    """Normaliza en L2 para comparar por coseno.

    ``center`` resta la media de la escena. No se usa al diarizar: medido sobre
    audio real, centrar una escena de un solo personaje deja únicamente ruido y
    la parte en voces inexistentes. Sin centrar, una misma voz queda a distancia
    casi cero y dos voces distintas se separan igual de bien.
    """
    data = np.asarray(embeddings, dtype=np.float32)
    if data.ndim != 2 or len(data) == 0:
        return data.reshape(len(data), -1).astype(np.float32)
    if center:
        data = data - data.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return np.divide(data, np.maximum(norms, 1e-8)).astype(np.float32)


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    similarity = np.clip(np.asarray(embeddings, dtype=np.float32) @ embeddings.T, -1.0, 1.0)
    distances = 1.0 - similarity
    np.fill_diagonal(distances, 0.0)
    return distances


def has_multiple_voices(embeddings: np.ndarray, min_separation_ratio: float = 8.0) -> bool:
    """¿La escena es claramente bimodal, o suena a una sola voz?

    Red de seguridad para el respaldo espectral, cuyo umbral absoluto no está
    calibrado contra audio real: sólo deja separar si las distancias grandes son
    muy superiores a las pequeñas. ECAPA no la necesita y no la usa.
    """
    if len(embeddings) < 2:
        return False
    values = cosine_distance_matrix(embeddings)[np.triu_indices(len(embeddings), k=1)]
    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 90))
    return high >= min_separation_ratio * max(low, 1e-3)


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    threshold: float,
    max_speakers: int = 6,
    cannot_link: set[tuple[int, int]] | None = None,
) -> list[int]:
    """Agrupamiento aglomerativo (enlace promedio) con restricciones de exclusión.

    Sigue fusionando por encima del umbral mientras haya más grupos que
    ``max_speakers``, y nunca fusiona dos segmentos marcados como incompatibles.
    """
    count = len(embeddings)
    if count == 0:
        return []
    if count == 1:
        return [0]
    distances = cosine_distance_matrix(embeddings)
    blocked = {(min(a, b), max(a, b)) for a, b in (cannot_link or set())}
    clusters: list[list[int]] = [[index] for index in range(count)]

    def mergeable(left: list[int], right: list[int]) -> bool:
        return not any((min(a, b), max(a, b)) in blocked for a in left for b in right)

    while len(clusters) > 1:
        best: tuple[float, int, int] | None = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                if not mergeable(clusters[left], clusters[right]):
                    continue
                distance = float(np.mean(distances[np.ix_(clusters[left], clusters[right])]))
                if best is None or distance < best[0]:
                    best = (distance, left, right)
        if best is None:
            break
        distance, left, right = best
        if distance > threshold and len(clusters) <= max(1, max_speakers):
            break
        clusters[left] = clusters[left] + clusters.pop(right)
    labels = [0] * count
    for label, cluster in enumerate(sorted(clusters, key=min)):
        for index in cluster:
            labels[index] = label
    return labels


def overlap_constraints(
    segments: list[dict], min_overlap_seconds: float = 0.05
) -> set[tuple[int, int]]:
    """Dos intervenciones que suenan a la vez no pueden ser el mismo personaje."""
    blocked: set[tuple[int, int]] = set()
    for left in range(len(segments)):
        for right in range(left + 1, len(segments)):
            start = max(float(segments[left]["start"]), float(segments[right]["start"]))
            end = min(float(segments[left]["end"]), float(segments[right]["end"]))
            if end - start > min_overlap_seconds:
                blocked.add((left, right))
    return blocked


def speech_windows(
    start: float,
    end: float,
    *,
    window_seconds: float = 1.5,
    hop_seconds: float = 0.75,
    min_window_seconds: float = 0.6,
) -> list[tuple[float, float]]:
    """Ventanas solapadas que recorren una intervención para vigilar el timbre."""
    if end - start <= 0:
        return []
    if end - start < window_seconds + hop_seconds:
        return [(start, end)]
    windows: list[tuple[float, float]] = []
    position = start
    while position + min_window_seconds <= end:
        windows.append((position, min(position + window_seconds, end)))
        position += hop_seconds
    return windows or [(start, end)]


def smooth_labels(labels: list[int]) -> list[int]:
    """Quita parpadeos de una sola ventana, que casi siempre son ruido."""
    if len(labels) < 3:
        return list(labels)
    smoothed = list(labels)
    for index in range(1, len(labels) - 1):
        if labels[index - 1] == labels[index + 1] != labels[index]:
            smoothed[index] = labels[index - 1]
    return smoothed


def change_points(windows: list[tuple[float, float]], labels: list[int]) -> list[float]:
    """Instantes en los que cambia la voz, a mitad del solape entre ventanas."""
    times: list[float] = []
    for index in range(1, len(labels)):
        if labels[index] == labels[index - 1]:
            continue
        previous_end, current_start = windows[index - 1][1], windows[index][0]
        times.append((previous_end + current_start) / 2.0)
    return times


def _split_text(text: str, pieces: int) -> list[str]:
    words = text.split()
    if pieces <= 1 or not words:
        return [text] + [""] * (pieces - 1)
    size = max(1, round(len(words) / pieces))
    chunks = [" ".join(words[index * size : (index + 1) * size]) for index in range(pieces)]
    remainder = " ".join(words[pieces * size :])
    if remainder:
        chunks[-1] = f"{chunks[-1]} {remainder}".strip()
    return chunks


def split_segment(
    segment: dict, boundaries: list[float], *, min_turn_seconds: float = 0.6
) -> list[dict]:
    """Parte una intervención de la ASR donde cambia el personaje.

    Con marcas de palabra el texto se reparte por el instante de cada palabra;
    sin ellas se reparte proporcionalmente, que es peor pero nunca pierde texto.
    """
    start, end = float(segment["start"]), float(segment["end"])
    cuts = sorted(
        time
        for time in boundaries
        if start + min_turn_seconds <= time <= end - min_turn_seconds
    )
    if not cuts:
        return [segment]
    edges = [start, *cuts, end]
    words = tuple(segment.get("words") or ())
    texts: list[str]
    if words:
        texts = []
        for index in range(len(edges) - 1):
            left, right = edges[index], edges[index + 1]
            chosen = [
                word for word in words if left <= (float(word[0]) + float(word[1])) / 2.0 < right
            ]
            texts.append(" ".join(str(word[2]).strip() for word in chosen).strip())
    else:
        texts = _split_text(str(segment.get("text", "")), len(edges) - 1)
    pieces: list[dict] = []
    for index, text in enumerate(texts):
        if not text:
            continue
        piece = {**segment, "start": edges[index], "end": edges[index + 1], "text": text}
        piece.pop("words", None)
        pieces.append(piece)
    if not pieces:
        return [segment]
    pieces[0]["start"], pieces[-1]["end"] = start, end
    for index in range(1, len(pieces)):
        pieces[index]["start"] = pieces[index - 1]["end"]
    return pieces


def cluster_separation(embeddings: np.ndarray, labels: list[int]) -> dict[int, float]:
    """Distancia media al grupo vecino más cercano.

    La cohesión sola no dice si un grupo es una voz o una mezcla: sobre ventanas
    de 1,5 s las distancias son ruidosas y su escala cambia con la escena. Lo que
    distingue una voz de una mezcla es el contraste entre lo de dentro y lo de
    fuera, así que se mide también esto para poder compararlos.
    """
    distances = cosine_distance_matrix(embeddings)
    groups = sorted(set(labels))
    separation: dict[int, float] = {}
    for label in groups:
        members = [index for index, item in enumerate(labels) if item == label]
        nearest = float("inf")
        for other in groups:
            if other == label:
                continue
            outsiders = [index for index, item in enumerate(labels) if item == other]
            block = distances[np.ix_(members, outsiders)]
            nearest = min(nearest, float(block.mean()))
        separation[label] = nearest
    return separation


def cluster_cohesion(embeddings: np.ndarray, labels: list[int]) -> dict[int, float]:
    """Distancia media dentro de cada grupo: cuánto confiar en que sea una sola voz."""
    distances = cosine_distance_matrix(embeddings)
    cohesion: dict[int, float] = {}
    for label in sorted(set(labels)):
        members = [index for index, item in enumerate(labels) if item == label]
        if len(members) < 2:
            cohesion[label] = 0.0
            continue
        block = distances[np.ix_(members, members)]
        count = len(members)
        cohesion[label] = float((block.sum() - np.trace(block)) / (count * (count - 1)))
    return cohesion


def neighbour_limits(
    segments: list[dict], guard_seconds: float = 0.08
) -> list[tuple[float, float]]:
    """Hasta dónde puede estirarse la referencia de cada línea sin tocar a otra.

    Evita que el relleno de una réplica corta arrastre la voz del personaje que
    habla justo antes o justo después, que es lo que arruina el clonado.
    """
    limits: list[tuple[float, float]] = []
    for index, segment in enumerate(segments):
        start, end = float(segment["start"]), float(segment["end"])
        left, right = 0.0, float("inf")
        for other_index, other in enumerate(segments):
            if other_index == index:
                continue
            other_start, other_end = float(other["start"]), float(other["end"])
            if other_end <= start:
                left = max(left, other_end + guard_seconds)
            elif other_start >= end:
                right = min(right, other_start - guard_seconds)
            else:
                left, right = max(left, start), min(right, end)
        limits.append((min(left, start), max(right, end)))
    return limits


def sentence_boundaries(segment: dict) -> list[float]:
    """Instantes entre una palabra que cierra frase y la siguiente."""
    words = tuple(segment.get("words") or ())
    cuts: list[float] = []
    for current, following in zip(words, words[1:]):
        text = str(current[2]).strip().rstrip("\"'»”)")
        if text.endswith((".", "?", "!", "…")):
            cuts.append((float(current[1]) + float(following[0])) / 2.0)
    return cuts


def merge_same_speaker(
    pieces: list[dict], labels: list[int], parents: list[int]
) -> tuple[list[dict], list[int]]:
    """Une frases seguidas de la misma intervención si son del mismo personaje."""
    merged: list[dict] = []
    merged_labels: list[int] = []
    previous_parent = None
    for piece, label, parent in zip(pieces, labels, parents):
        if merged and parent == previous_parent and label == merged_labels[-1]:
            joined = f"{merged[-1]['text']} {piece['text']}".strip()
            merged[-1] = {**merged[-1], "end": piece["end"], "text": joined}
        else:
            merged.append(dict(piece))
            merged_labels.append(label)
        previous_parent = parent
    return merged, merged_labels


_SENTENCE_END = (".", "?", "!", "…")


def join_unfinished(
    pieces: list[dict],
    labels: list[int],
    *,
    max_gap_seconds: float = 0.7,
    contiguous_seconds: float = 0.15,
    unreliable_seconds: float = 1.0,
    max_seconds: float = 10.0,
) -> tuple[list[dict], list[int]]:
    """Une una frase que Whisper dejó a medias con su continuación.

    Whisper parte a veces una frase en dos intervenciones ("You are" |
    "stressed.") y Qwen traducía cada trozo por separado. Se unen si la primera
    no cierra frase, juntas no pasan de ``max_seconds`` y, además, o son del
    mismo personaje con una pausa de hasta ``max_gap_seconds`` (medidas en el
    monólogo de BC_3: 0,54-0,56 s), o van pegadas y uno de los dos dura menos
    de ``unreliable_seconds``: la huella de un trozo de 0,9 s no vale como
    prueba de voz, y una frase que sigue sin pausa sí. Con dos trozos largos
    manda la etiqueta, porque Whisper omite a menudo el punto final y unía
    "Goodbye, Florian. No" de Niko con la réplica de Bernie. Las de habla
    simultánea no se tocan.
    """
    if not pieces:
        return pieces, labels
    joined: list[dict] = [dict(pieces[0])]
    joined_labels: list[int] = [labels[0]]
    for piece, label in zip(pieces[1:], labels[1:]):
        last = joined[-1]
        text = str(last["text"]).strip().rstrip("\"'»”)")
        gap = float(piece["start"]) - float(last["end"])
        if (
            last.get("speaker_source") is None
            and piece.get("speaker_source") is None
            and not text.endswith(_SENTENCE_END)
            and float(piece["end"]) - float(last["start"]) <= max_seconds
            and (
                (label == joined_labels[-1] and 0.0 <= gap <= max_gap_seconds)
                or (
                    -0.05 <= gap <= contiguous_seconds
                    and min(
                        float(last["end"]) - float(last["start"]),
                        float(piece["end"]) - float(piece["start"]),
                    ) < unreliable_seconds
                )
            )
        ):
            joined[-1] = {
                **last,
                "end": piece["end"],
                "text": f"{last['text'].strip()} {piece['text'].strip()}".strip(),
            }
            joined[-1].pop("words", None)
            continue
        joined.append(dict(piece))
        joined_labels.append(label)
    return joined, joined_labels


def cluster_labels(
    voices: np.ndarray,
    pieces: list[dict],
    exclusive: set[tuple[int, int]],
    *,
    threshold: float,
    max_speakers: int,
    min_cluster_seconds: float = 0.0,
) -> list[int]:
    """Agrupa por personaje fundando grupos sólo con las réplicas fiables.

    Una réplica de menos de un segundo da una huella de voz ruidosa y acababa
    fundando un personaje propio ("Yeah." como PERSONAJE 4 en B_1). Aquí sólo
    las réplicas de al menos ``min_cluster_seconds`` crean grupos; las cortas
    se asignan al personaje más parecido que no tengan prohibido.
    """
    reliable = [
        index for index, piece in enumerate(pieces)
        if float(piece["end"]) - float(piece["start"]) >= min_cluster_seconds
    ]
    if min_cluster_seconds <= 0 or len(reliable) < 2 or len(reliable) == len(pieces):
        return cluster_embeddings(
            voices, threshold=threshold, max_speakers=max_speakers, cannot_link=exclusive
        )
    position = {index: order for order, index in enumerate(reliable)}
    core = cluster_embeddings(
        voices[reliable], threshold=threshold, max_speakers=max_speakers,
        cannot_link={
            (position[a], position[b]) for a, b in exclusive
            if a in position and b in position
        },
    )
    labels = [-1] * len(pieces)
    for order, index in enumerate(reliable):
        labels[index] = core[order]
    groups = sorted(set(core))
    centroids = {}
    for group in groups:
        members = [reliable[order] for order, label in enumerate(core) if label == group]
        centre = voices[members].mean(axis=0)
        centroids[group] = centre / max(float(np.linalg.norm(centre)), 1e-8)
    for index in range(len(pieces)):
        if labels[index] >= 0:
            continue
        banned = set()
        for a, b in exclusive:
            other = b if a == index else a if b == index else None
            if other is not None and labels[other] >= 0:
                banned.add(labels[other])
        options = [group for group in groups if group not in banned] or groups
        labels[index] = max(options, key=lambda group: float(voices[index] @ centroids[group]))
    return labels


def diarize_scene(
    segments: list[dict],
    audio_for_segment,
    sample_rate: int,
    *,
    embed=None,
    threshold: float = 0.75,
    change_threshold: float | None = None,
    max_speakers: int = 8,
    window_seconds: float = 1.5,
    hop_seconds: float = 0.75,
    min_window_seconds: float = 0.6,
    min_turn_seconds: float = 0.6,
    min_overlap_seconds: float = 0.05,
    min_separation_ratio: float | None = None,
    separation_out: dict[int, float] | None = None,
    split_turns: bool = True,
    min_cluster_seconds: float = 0.0,
    split_sentences: bool = False,
) -> tuple[list[dict], list[int], dict[int, float]]:
    """Recorre la escena por ventanas, corta donde cambia la voz y agrupa por personaje.

    Son dos decisiones distintas y por eso tienen umbrales distintos. Detectar
    que dentro de una intervención cambia el hablante es fácil y conviene ser
    sensible (``change_threshold``); decidir que dos turnos separados de la
    escena son el mismo personaje es difícil y conviene ser conservador
    (``threshold``), porque equivocarse ahí clona la voz de otro. Sin
    ``change_threshold`` explícito se usa el mismo valor, que es lo prudente
    cuando la escala de distancias del backend no está calibrada.

    Devuelve las intervenciones ya partidas, su etiqueta de personaje y la
    cohesión de cada grupo, que dice si merece una referencia vocal común.
    """
    if not segments:
        return [], [], {}
    embed = embed or spectral_embeddings
    windows: list[tuple[float, float]] = []
    owners: list[int] = []
    clips: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        audio = audio_for_segment(segment)
        for start, end in speech_windows(
            float(segment["start"]), float(segment["end"]),
            window_seconds=window_seconds, hop_seconds=hop_seconds,
            min_window_seconds=min_window_seconds,
        ):
            first = max(0, round(start * sample_rate))
            last = min(len(audio), round(end * sample_rate))
            if last <= first:
                continue
            windows.append((start, end))
            owners.append(index)
            clips.append(np.asarray(audio[first:last], dtype=np.float32).reshape(-1))
    if not clips:
        return segments, [0] * len(segments), {}
    embeddings = normalize_embeddings(embed(clips, sample_rate))
    simultaneous = overlap_constraints(segments, min_overlap_seconds)
    blocked = {
        (left, right)
        for left in range(len(windows))
        for right in range(left + 1, len(windows))
        if (min(owners[left], owners[right]), max(owners[left], owners[right])) in simultaneous
    }
    single_voice = (
        min_separation_ratio is not None
        and not blocked
        and not has_multiple_voices(embeddings, float(min_separation_ratio))
    )
    if single_voice:
        if separation_out is not None:
            separation_out[0] = float("inf")
        return segments, [0] * len(segments), {0: 0.0}
    split: list[dict] = []
    siblings: list[list[int]] = []
    parents: list[int] = []
    for index, segment in enumerate(segments):
        if not split_turns:
            # Cortar por cambio de voz en la cuadrícula de 0,75 s partía frases
            # y, como los trozos no podían reagruparse, inventaba personajes:
            # 20 de 22 escenas acababan en el tope de 8 voces (27/09/2026).
            # Aquí sólo se corta en finales de frase, que es donde cambia
            # quien habla, y sin prohibir que las frases compartan personaje:
            # las seguidas de la misma voz se vuelven a unir al final.
            pieces = (
                split_segment(
                    segment, sentence_boundaries(segment), min_turn_seconds=min_turn_seconds
                )
                if split_sentences else [segment]
            )
            siblings.extend([len(split) + order] for order in range(len(pieces)))
            parents.extend([index] * len(pieces))
            split.extend(pieces)
            continue
        own = [position for position, owner in enumerate(owners) if owner == index]
        local = smooth_labels(
            cluster_embeddings(
                embeddings[own],
                threshold=threshold if change_threshold is None else change_threshold,
                max_speakers=max_speakers,
            )
        )
        boundaries = change_points([windows[position] for position in own], local)
        pieces = split_segment(segment, boundaries, min_turn_seconds=min_turn_seconds)
        siblings.append(list(range(len(split), len(split) + len(pieces))))
        parents.extend([index] * len(pieces))
        split.extend(pieces)
    centres = [(start + end) / 2.0 for start, end in windows]
    pooled: list[np.ndarray] = []
    for piece, parent in zip(split, parents):
        start, end = float(piece["start"]), float(piece["end"])
        # Sólo ventanas de su propia intervención: en habla simultánea, las de
        # otro segmento caen en el mismo instante y contaminarían la huella.
        own = [position for position, owner in enumerate(owners) if owner == parent]
        inside = [position for position in own if start <= centres[position] < end]
        if not inside:
            middle = (start + end) / 2.0
            inside = [min(own, key=lambda p: abs(centres[p] - middle))]
        pooled.append(embeddings[inside].mean(axis=0))
    voices = normalize_embeddings(np.stack(pooled))
    # Un corte separa dos voces por definición: las mitades no pueden reagruparse.
    exclusive = {
        (group[order], group[order + 1])
        for group in siblings
        for order in range(len(group) - 1)
    }
    exclusive |= overlap_constraints(split, min_overlap_seconds)
    labels = cluster_labels(
        voices, split, exclusive,
        threshold=threshold, max_speakers=max_speakers,
        min_cluster_seconds=min_cluster_seconds,
    )
    cohesion = cluster_cohesion(voices, labels)
    separation = cluster_separation(voices, labels)
    if split_sentences and not split_turns:
        split, labels = merge_same_speaker(split, labels, parents)
        split, labels = join_unfinished(split, labels)
    if separation_out is not None:
        separation_out.update(separation)
    for label in sorted(cohesion):
        neighbour = separation.get(label, float("inf"))
        logger.info(
            "[SPEAKERS] medida PERSONAJE %d: dentro %.2f vecino %.2f contraste %.2f",
            label + 1, cohesion[label], neighbour,
            (neighbour - cohesion[label]) if neighbour != float("inf") else float("nan"),
        )
    return split, labels, cohesion


def speaker_profile_segments(
    segments: list[dict],
    labels: list[int],
    *,
    target_seconds: float = 6.0,
    max_windows: int = 6,
    min_window_seconds: float = 0.4,
) -> dict[int, list[int]]:
    """Índices de las intervenciones más largas de cada personaje, en orden temporal."""
    grouped: dict[int, list[int]] = {}
    for label in sorted(set(labels)):
        candidates = [
            index
            for index, (segment, item) in enumerate(zip(segments, labels))
            if item == label
            and float(segment["end"]) - float(segment["start"]) >= min_window_seconds
        ]
        candidates.sort(
            key=lambda index: float(segments[index]["end"]) - float(segments[index]["start"]),
            reverse=True,
        )
        chosen: list[int] = []
        pooled = 0.0
        for index in candidates[: max(1, max_windows)]:
            chosen.append(index)
            pooled += float(segments[index]["end"]) - float(segments[index]["start"])
            if pooled >= target_seconds:
                break
        if chosen:
            grouped[label] = sorted(chosen)
    return grouped
