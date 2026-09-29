<div align="center">

# 🎙️ GTA IV LATAM AI Dubber

**Pipeline local para doblar a español latino los WAV extraídos de Grand Theft Auto IV.**

[![Tests](https://github.com/codepdbh/gtaiv-latam-ai-dubber/actions/workflows/tests.yml/badge.svg)](https://github.com/codepdbh/gtaiv-latam-ai-dubber/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10–3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![WSL2](https://img.shields.io/badge/Windows_11-WSL2-0078D4?logo=windows11&logoColor=white)](https://learn.microsoft.com/windows/wsl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Creado por **Paulo Daniel Batuani Hurtado**<br>
📺 **[Visita y suscríbete a DanielMcDt](https://www.youtube.com/@DanielMcDt)**

</div>

> [!IMPORTANT]
> Este es un proyecto comunitario, no oficial y sin afiliación con Rockstar Games ni
> Take-Two Interactive. Debes poseer legalmente el juego y extraer tus propios audios.
> El repositorio **no contiene ni distribuye archivos de GTA IV, modelos de IA o RPF**.

## ¿Qué hace?

El pipeline trabaja completamente en tu PC:

```text
WAV original → Whisper → adaptación → clonación → ajuste temporal → WAV final
            ↘ Demucs (voz/fondo) → ECAPA (quién habla) ↗
```

- Transcribe el diálogo inglés con [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- Lo adapta a español latino neutral mediante [Ollama](https://ollama.com/) y Qwen.
- Conserva la identidad vocal con [CosyVoice 3](https://github.com/FunAudioLLM/CosyVoice).
- En cutscenes separa voz y fondo con [Demucs](https://github.com/facebookresearch/demucs).
- Identifica qué personaje habla en cada frase de una cutscene con
  [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).
- Opcionalmente separa dos voces simultáneas con
  [SpeechBrain SepFormer](https://huggingface.co/speechbrain/sepformer-wsj02mix); viene
  desactivado porque en las escenas del juego hacía más daño que bien (ver más abajo).
- Reanuda trabajos interrumpidos con SQLite y valida cada resultado antes de publicarlo.
- Incluye CLI, interfaz PySide6, canarios previos y reportes JSON/CSV.

### Garantías del audio

Los WAV finales conservan el **mismo nombre, ruta relativa, sample rate, número de
canales y duración exacta en frames** que el original. En cutscenes también se conserva
el orden físico de canales. Si falla una etapa, el sistema utiliza un *fallback* seguro:
jamás publica un archivo vacío o parcialmente escrito.

## Requisitos

- Windows 11 con WSL2 y Ubuntu.
- GPU NVIDIA visible dentro de WSL mediante `nvidia-smi`.
- Driver NVIDIA instalado en **Windows**; no instales un driver Linux dentro de WSL.
- Git y al menos 15 GB libres para entorno, cachés y modelos, además del espacio para audios.
- Una copia legal de GTA IV y sus WAV ya extraídos.

La configuración se probó con una GPU NVIDIA de 8 GB. El procesamiento es secuencial y
libera modelos entre etapas para reducir el uso de VRAM. Una GPU con más memoria permite
probar modelos de traducción mayores, pero siempre conviene validar primero con canarios.
El instalador actual está orientado a CUDA; CPU es una opción manual y será mucho más lenta.

## Instalación rápida

### 1. Instalar WSL2

En PowerShell como administrador:

```powershell
wsl --install -d Ubuntu
wsl --update
```

Reinicia si Windows lo solicita, abre Ubuntu y confirma que `nvidia-smi` muestra tu GPU.

### 2. Clonar e instalar

Dentro de Ubuntu/WSL:

```bash
git clone https://github.com/codepdbh/gtaiv-latam-ai-dubber.git
cd gtaiv-latam-ai-dubber
bash scripts/setup_wsl.sh --download-models
```

El instalador crea un entorno aislado en
`~/.local/share/gtaiv-latam-ai-dubber`, instala las dependencias y descarga Whisper,
Qwen y CosyVoice. Son varios gigabytes y el primer arranque puede tardar.

Si sólo quieres reinstalar el código y las dependencias sin volver a descargar modelos:

```bash
bash scripts/setup_wsl.sh
```

### 3. Preparar los audios

Coloca los WAV que tú extrajiste junto al repositorio o utiliza otra carpeta de proyecto:

```text
mi-proyecto/
├── Speech/       # originales: sólo lectura
└── Cutscenes/    # originales: sólo lectura
```

El programa creará `Speech_es/`, `Cutscenes_es/`, `work/` y `logs/`. Nunca escribe
dentro de las carpetas originales.

Una ruta de Windows como `D:\Juegos\audiosgtaiv` se ve en WSL como
`/mnt/d/Juegos/audiosgtaiv`.

## Primer doblaje y ejecución completa

Activa primero un canario real. Procesa una muestra de Speech y una cutscene, y detiene
la cola si una inferencia crítica falla:

```bash
source ~/.local/share/gtaiv-latam-ai-dubber/.venv/bin/activate
export PATH="$HOME/.local/share/gtaiv-latam-ai-dubber/bin:$PATH"
python -m gtaiv_dubber process /mnt/d/Juegos/audiosgtaiv --canary-only
```

Cuando escuches y apruebes esas muestras, inicia la cola persistente:

```bash
bash scripts/start_mass.sh /mnt/d/Juegos/audiosgtaiv
bash scripts/status_mass.sh
```

Puedes cerrar la terminal: el proceso continúa mediante `nohup`. Si Windows o WSL se
reinicia, ejecuta de nuevo `start_mass.sh`; el estado guardado permite reanudar sin repetir
los archivos `DONE` que sigan siendo válidos.

También puedes ejecutarlo en primer plano:

```bash
bash scripts/run.sh /mnt/d/Juegos/audiosgtaiv
```

O abrir la interfaz gráfica con WSLg:

```bash
bash scripts/gui.sh /mnt/d/Juegos/audiosgtaiv
```

Desde Windows están disponibles `RUN_GTAIV_DUBBER.bat`, `RUN_GTAIV_GUI.bat` y
`STATUS_GTAIV_DUBBER.bat`.

## Comandos útiles

```bash
python -m gtaiv_dubber env      /ruta/proyecto
python -m gtaiv_dubber scan     /ruta/proyecto
python -m gtaiv_dubber process  /ruta/proyecto --speech
python -m gtaiv_dubber process  /ruta/proyecto --cutscenes
python -m gtaiv_dubber status   /ruta/proyecto
python -m gtaiv_dubber verify   /ruta/proyecto
python -m gtaiv_dubber gui      /ruta/proyecto
```

Para probar individualmente los motores:

```bash
python scripts/smoke_models.py /ruta/proyecto qwen
python scripts/smoke_models.py /ruta/proyecto whisper
python scripts/smoke_models.py /ruta/proyecto cosyvoice
```

Los reportes quedan en `work/reports/` y los logs rotativos en `logs/`. Mientras la cola
no termine, `verify` marcará como ausentes los WAV todavía pendientes; no significa que
los originales estén dañados.

## Cambiar modelos

Copia primero la configuración de ejemplo:

```bash
cp config/local.example.yaml config/local.yaml
```

`config/local.yaml` es privado, está ignorado por Git y sólo necesita contener las
opciones que quieras cambiar. Después de cualquier cambio, ejecuta un canario: el hash de
configuración cambia y los resultados anteriores pueden necesitar regenerarse.

### Modelo de traducción/adaptación (Ollama)

Descarga el modelo deseado y usa exactamente su etiqueta:

```bash
ollama pull qwen3:14b
```

```yaml
ollama:
  model: qwen3:14b
```

Un modelo mayor puede adaptar mejor contexto y expresiones, pero requiere más VRAM/RAM y
tarda más. Si no cabe en GPU, Ollama puede repartir capas en memoria del sistema, con una
pérdida importante de velocidad. Conserva `qwen3:8b` en equipos de 8 GB salvo que tus
canarios demuestren una mejora estable. Consulta las etiquetas disponibles en la
[biblioteca oficial de Qwen3 para Ollama](https://ollama.com/library/qwen3).

### Modelo de transcripción (faster-whisper)

Modifica `whisper.model`; un nombre compatible se descargará en el primer uso:

```yaml
whisper:
  model: large-v3
  device: cuda
  compute_type: float16
```

`large-v3-turbo` ofrece un buen equilibrio. `large-v3` puede mejorar casos difíciles a
cambio de velocidad. Para experimentar con CPU usa `device: cpu` y un tipo de cómputo
compatible, como `int8`, aunque el resto del instalador sigue optimizado para CUDA.

### Modelo de voz (CosyVoice)

Puedes usar otro checkpoint únicamente si es compatible con la versión instalada de
CosyVoice, `AutoModel` y `inference_cross_lingual`. Descárgalo en otra carpeta y apunta
`model_dir` hacia ella:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("ORGANIZACION/MODELO-COMPATIBLE", local_dir="/ruta/al/modelo")
PY
```

```yaml
cosyvoice:
  model_id: ORGANIZACION/MODELO-COMPATIBLE
  model_dir: /ruta/al/modelo
```

Cambiar a un motor TTS arbitrario (XTTS, F5-TTS, etc.) no se logra sólo con YAML: requiere
implementar un adaptador que entregue audio compatible con el pipeline. El proyecto usa el
formato oficial de CosyVoice para que la instrucción interna no sea pronunciada. Escucha
siempre varios canarios antes de lanzar miles de archivos.

### Separación vocal (Demucs)

```yaml
demucs:
  model: htdemucs_ft
```

`htdemucs_ft` puede dar mejores separaciones en algunos casos, pero el proyecto oficial
advierte que tarda aproximadamente cuatro veces más que `htdemucs`. Sólo afecta cutscenes.

## Cosas importantes antes de procesar miles de WAV

- Escucha canarios de personajes distintos y al menos una cutscene multicanal.
- Comprueba que el audio diga la traducción real, no una instrucción o texto repetido.
- Revisa `REVIEW_*`, `FALLBACK` y `ERROR`; `DONE` significa que pasó la validación técnica.
- No edites `config/default.yaml` para ajustes personales: usa `config/local.yaml`.
- Conserva espacio para originales + resultados + temporales; una colección grande puede
  requerir decenas de gigabytes adicionales.
- No ejecutes dos colas sobre el mismo proyecto al mismo tiempo.
- Voz clonada y material del juego pueden estar sujetos a derechos y normas locales. Usa
  el resultado de forma responsable y no suplantes a personas reales.

## Cutscenes y conservación del fondo

Para Speech se procesa cada WAV sin Demucs por defecto. Para cutscenes, el downmix sólo se
usa en ASR/separación: la reconstrucción parte de todos los canales originales, atenúa la
voz inglesa estimada y conserva música, efectos y ambiente. Cada segmento traducido se
coloca en su ventana temporal y se vuelve a distribuir según la energía de los canales.

### Encaje de cada réplica sin deformar la voz

El español suele ocupar más que el inglés, y CosyVoice habla a su propio ritmo. Estirar cada
réplica al frame exacto de la original la dejaba arrastrada cuando sobraba tiempo y acelerada
cuando faltaba. Por eso en cutscenes:

- **La voz nunca se ralentiza.** Si la réplica es más corta que su hueco, lo que sobra es
  silencio.
- **Si es más larga, primero ocupa el silencio que la sigue**, hasta `max_overrun_seconds` y
  sin pisar la réplica siguiente.
- **Si aun así no cabe, se reescribe con números**: Qwen recibe cuánto dura su versión, cuánto
  tiempo tiene el personaje y cuántas palabras debe usar como máximo. Se hacen hasta
  `length_regeneration_attempts` intentos y se queda el que mejor encaja.
- **Sólo como último recurso se acelera**, nunca más de `hard_max_speedup` (1,35x). Si ni así
  cabe, la cola se corta con un fundido y el segmento queda en `REVIEW_DURATION`.
- **Risas, quejidos, huecos minúsculos y frases que quedan igual en español conservan la voz
  del actor original**, igual que una traducción que Qwen repite en bucle dos veces seguidas.
  También se descartan las alucinaciones típicas de Whisper sobre música ("You You You...").

La base de datos guarda en `duration_error_pct` la deformación aplicada a cada réplica de
cutscene (0 = velocidad natural), que es lo que realmente se oye.

La validación exige WAV PCM firmado de 16 bits, legibilidad, sample rate, canales y número
de frames idénticos al archivo original.

### Varias personas en una cutscene

Una cutscene es una escena larga donde varios personajes se turnan. Whisper no corta por
hablante, sino por pausas: un mismo intervalo puede contener la réplica de un personaje y la
respuesta de otro, y a veces parte una frase en dos. Doblar esos bloques tal cual mezcla dos
timbres en una frase o traduce fragmentos sueltos.

Por eso, tras el ASR:

1. Cada intervalo se corta **sólo en finales de frase** (`.`, `?`, `!`), que es donde cambia
   quien habla. Nunca a mitad de frase.
2. Cada frase recibe una huella vocal con
   [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb). Sólo las
   frases de al menos `min_cluster_seconds` fundan personajes; las más cortas dan huellas
   ruidosas y se asignan al personaje más parecido en lugar de inventar uno nuevo.
3. Las frases seguidas del mismo personaje se vuelven a unir, y las que Whisper dejó a medias
   se completan con su continuación.

Cada réplica se clona a partir de su propio audio original. Sólo se fabrica una **referencia
vocal común** —varias intervenciones del mismo personaje concatenadas— cuando el grupo se
parece a sí mismo claramente más que al personaje vecino más cercano
(`profile_min_contrast`). Un umbral absoluto no sirve: la escala de distancias cambia de una
escena a otra. La referencia común también se usa como reserva cuando una réplica es demasiado
corta para clonarse desde su propio audio.

El relleno de las referencias nunca invade a los vecinos, y la traducción recibe el diálogo de
la escena etiquetado por personaje para mantener género gramatical y tratamiento coherentes.

El identificador de voces tiene límites reales con este audio: las voces de personajes
distintos no quedan muy separadas entre sí, y las frases de teléfono o muy cortas se etiquetan
peor. Por eso nada depende de acertar la etiqueta: aunque un personaje salga como dos, cada
réplica suena con la voz de quien la dijo.

Antes se cortaba en una cuadrícula fija de ventanas de 1,5 s (`split_turns: true`). Partía
frases por la mitad e inventaba personajes: 20 de 22 escenas acababan en el tope de 8 voces.
Se conserva como opción, pero no se recomienda.

### Voces simultáneas (SepFormer)

SepFormer divide el stem vocal en dos fuentes y Whisper vuelve a transcribirlas por separado.
Está **desactivado por defecto** (`speaker_separation.enabled: false`): en las escenas del juego
saltaba en 18-21 intervenciones por escena de diálogo normal por turnos. Whisper alucinaba frases
sobre la fuente débil y juntaba media conversación en bloques de hasta 69 s con una sola voz. Sin
él, en un solapamiento real Whisper toma la voz dominante, que es un fallo mucho menor.

La voz inglesa combinada se proyecta sobre los canales físicos, se retira una sola vez al
100% y después se agregan las voces nuevas, lo que evita tanto el inglés residual como la doble
sustracción en los solapamientos.

ECAPA (y SepFormer si se activa) usan CPU por defecto para no competir con Whisper/CosyVoice
en una GPU de 8 GB. En equipos con más VRAM se puede probar:

```yaml
diarization:
  enabled: true
  device: cuda
```

Con `backend: spectral` todo funciona sin descargar ECAPA, usando una huella log-mel en numpy:
separa timbres claramente distintos, pero es menos fiable.

## Desarrollo

Las pruebas unitarias no descargan modelos ni necesitan los audios del juego:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) antes de enviar cambios.

## Privacidad y licencia

La inferencia se ejecuta localmente. Ollama escucha por defecto en `127.0.0.1`; no expongas
ese puerto a redes no confiables. Los modelos pueden descargarse desde sus proveedores,
pero los WAV no se envían a un servicio de doblaje remoto.

El código está publicado bajo la [licencia MIT](LICENSE). Cada dependencia y modelo mantiene
su propia licencia. GTA IV, sus personajes, nombres y recursos pertenecen a sus respectivos
titulares.

---

<div align="center">

Si el proyecto te sirve, deja una ⭐ y acompaña el desarrollo en<br>
## 📺 [youtube.com/@DanielMcDt](https://www.youtube.com/@DanielMcDt)

</div>
