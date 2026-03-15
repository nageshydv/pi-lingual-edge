import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
import os
import wave
import tempfile
import subprocess
import requests
import numpy as np
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000
RECORD_SECONDS    = 6
SILENCE_THRESHOLD = 0.001
OLLAMA_MODEL      = "qwen2.5:1.5b"

BASE_DIR   = Path(__file__).parent
VOICES_DIR = BASE_DIR / "voices"
PIPER_BIN  = BASE_DIR / "piper" / "piper"   # standalone ARM64 binary

# whisper.cpp — pure C++, no PyTorch, native ARM. Built by setup_pi.sh.
WHISPER_BIN        = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL_PATH = Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin"

VOICES = {
    "en": VOICES_DIR / "en_US-lessac-medium.onnx",
    "hi": None,   # Hindi uses gTTS (Piper Hindi voice unavailable upstream)
    "sk": VOICES_DIR / "sk_SK-lili-medium.onnx",
}
LANG_NAMES = {"en": "English", "hi": "Hindi", "sk": "Slovak"}

# USB mic device index. Find yours with:
#   python3 -c "import sounddevice as sd; print(sd.query_devices())"
# Then set the index of your USB mic here.
MIC_DEVICE    = 1      # USB Microphone (hw:3,0)
SPEAKER_CARD  = "hw:2,0"  # bcm2835 Headphones 3.5mm jack


# ── RECORD ───────────────────────────────────────────────────────────
def record_audio(seconds=RECORD_SECONDS, callback=None):
    """Record from USB mic at its native rate, resample to 16kHz for Whisper."""
    import sounddevice as sd
    import time
    frames = []

    device_info = sd.query_devices(MIC_DEVICE, kind="input")
    native_rate = int(device_info["default_samplerate"])  # typically 44100

    def cb(indata, frame_count, time_info, status):
        chunk = indata.copy().flatten()
        frames.append(chunk)
        if callback:
            callback(float(np.sqrt(np.mean(chunk ** 2))))

    with sd.InputStream(
        samplerate=native_rate, channels=1, dtype="float32",
        device=MIC_DEVICE, blocksize=int(native_rate * 0.1), callback=cb,
    ):
        time.sleep(seconds)

    audio = np.concatenate(frames) if frames else np.zeros(native_rate)

    # Resample to 16kHz if mic runs at a different rate (e.g. 44100)
    if native_rate != SAMPLE_RATE:
        import scipy.signal as sig
        samples_16k = int(len(audio) * SAMPLE_RATE / native_rate)
        audio = sig.resample(audio, samples_16k)

    return audio


# ── TRANSCRIBE ───────────────────────────────────────────────────────
def transcribe(audio):
    """Transcribe using whisper.cpp binary — no PyTorch, runs on ARM."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        with wave.open(tmp, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        if not WHISPER_BIN.exists():
            raise RuntimeError(
                f"whisper.cpp binary not found at {WHISPER_BIN}\n"
                "Run:  cd ~/whisper.cpp && cmake -B build && cmake --build build -j4")

        result = subprocess.run(
            [str(WHISPER_BIN),
             "-m", str(WHISPER_MODEL_PATH),
             "-f", tmp,
             "-l", "auto",
             "--no-timestamps",
             "-otxt",
             "-of", tmp],
            capture_output=True, text=True
        )

        txt_file = tmp + ".txt"
        if not os.path.exists(txt_file):
            raise RuntimeError(f"whisper.cpp failed:\n{result.stderr[:300]}")

        text = open(txt_file).read().strip()
        os.unlink(txt_file)

        # Parse detected language from stderr
        # whisper.cpp prints: "auto-detected language: en (p = 0.97)"
        lang = "en"
        for line in result.stderr.splitlines():
            if "auto-detected language" in line:
                lang = line.split(":")[1].strip().split()[0].strip()
                break

        return text, lang

    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ── TRANSLATE ────────────────────────────────────────────────────────
def translate(text, source_lang, target_lang):
    src = LANG_NAMES.get(source_lang, source_lang)
    tgt = LANG_NAMES[target_lang]
    prompt = (
        f"Translate the following {src} text to {tgt}. "
        f"Output ONLY the translation, nothing else.\n\n"
        f"Text: {text}\n\nTranslation:"
    )
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama not running. Run:  ollama serve &")
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama timed out — model may still be loading.")


# ── SPEAK ────────────────────────────────────────────────────────────
def speak(text, lang):
    tmp_wav = tmp_mp3 = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name

        if lang == "hi":
            from gtts import gTTS
            tmp_mp3 = tmp_wav.replace(".wav", ".mp3")
            gTTS(text=text, lang="hi", slow=False).save(tmp_mp3)
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3,
                 "-ar", "22050", "-ac", "1", "-acodec", "pcm_s16le",
                 tmp_wav],
                capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg: {r.stderr.decode()[:100]}")
        else:
            voice = VOICES.get(lang)
            if not voice or not voice.exists():
                raise RuntimeError(f"Voice file missing: {voice}")
            if not PIPER_BIN.exists():
                raise RuntimeError(
                    f"Piper binary not found: {PIPER_BIN}\n"
                    "Run setup_pi.sh to download it.")
            r = subprocess.run(
                [str(PIPER_BIN), "--model", str(voice), "--output_file", tmp_wav],
                input=text.encode(), capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"Piper: {r.stderr.decode()[:100]}")

        if not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) < 44:
            raise RuntimeError(f"Audio file empty for {lang}")

        subprocess.run(["aplay", "-q", "-D", SPEAKER_CARD, tmp_wav], check=True)

    finally:
        for f in [tmp_wav, tmp_mp3]:
            if f and os.path.exists(f):
                os.unlink(f)


# ── HELPERS ──────────────────────────────────────────────────────────
def get_targets(source_lang):
    return [l for l in LANG_NAMES if l != source_lang]

def normalise_lang(detected):
    return {"en":"en","hi":"hi","sk":"sk","cs":"sk"}.get(detected, "en")
