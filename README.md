# π-Lingual Edge

A side project to get 3-way offline voice translation running on a barebones Raspberry Pi 4.  The best part is it runs completely offline with no cloud dependencies or API keys needed (except for Hindi TTS).

This project brings back some memories from my PhD days working with constrained hardware. The challenge is always the same: watch your memory budget, pick the right model sizes and figure out how to work around the limitations of the device.

## Why this is tricky 

Running a full speech-to-text, LLM translation, and text-to-speech pipeline on a $50 (ish, as of Feb 2026)single-board computer with only 4GB of RAM is an interesting challenge for 2026. 


* **whisper.cpp** instead of openai-whisper/PyTorch. It's written in pure C++, compiles natively for ARM, and only takes up about 75MB.
* **qwen2.5:1.5b via Ollama**. This fits comfortably in 4GB of RAM alongside the OS overhead, supports multiple languages and gives surprisingly good translation quality.
* **Piper TTS ARM64 binary**. It comes pre-compiled and completely avoids Python package dependency hell.

Everything runs locally for English and Slovak. Unfortunately, I had to use gTTS for Hindi text-to-speech which does require a network connection, but the actual transcription and translation part stays offline.

## Architecture Overview

Here is the quick summary of how data flows:

```
USB Mic (44100Hz)
  -> sounddevice (ALSA)
  -> scipy resample (44100 to 16000Hz)
  -> whisper.cpp ggml-tiny (C++, ARM native, takes about 3s)
  -> Ollama qwen2.5:1.5b (local, takes about 15-25s)
  -> Piper ARM64 / gTTS
  -> aplay hw:2,0 (3.5mm headphones)
```

There's also a tiny web UI built with Flask that serves on port 5000. It uses Server-Sent Events to stream translation updates to your browser in real time.

## Hardware Requirements

| Component | What you need |
|---|---|
| Board | Raspberry Pi 4 Model B |
| RAM | 4GB is the bare minimum, but 8GB will make your life easier |
| OS | Raspberry Pi OS Bookworm 64-bit |
| Mic | Any standard USB microphone |
| Audio out | 3.5mm headphones connected to the bcm2835 jack |
| Network | WiFi (only needed for the initial setup and Hindi TTS) |


## Quick Setup

**1. Copy the files and run the setup script:**

```bash
# From your computer
scp pilingual-pi.zip nagesh@raspberrypi.local:~

# SSH into the Pi
ssh nagesh@raspberrypi.local
unzip pilingual-pi.zip && cd pilingual-pi
chmod +x setup_pi.sh && ./setup_pi.sh
```

Give it about 15 to 20 minutes to finish. The script needs to compile whisper.cpp from source, download the 1GB Ollama model, and pull the Piper voice model files.

**2. Start the app:**

```bash
source venv/bin/activate
ollama serve &
python3 app.py
```

**3. Open the interface:**

Point your browser to `http://<your-pi-ip>:5000`. You can find your Pi's IP address by running `hostname -I` in the terminal.

## Manual Setup Steps

If you want to install everything by hand instead of using the script:

```bash
# 1. Install system dependencies
sudo apt update && sudo apt install -y \
  python3-pip python3-venv portaudio19-dev ffmpeg \
  alsa-utils cmake git libespeak-ng1

# 2. Set up the Python environment
python3 -m venv venv && source venv/bin/activate
pip install flask sounddevice numpy requests scipy gtts onnxruntime

# 3. Get the Piper TTS binary for ARM64
mkdir -p piper
curl -L https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz \
  | tar -xz -C piper --strip-components=1

# 4. Compile whisper.cpp from source (this gets us around the PyTorch requirement)
cd ~ && git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
cmake -B build && cmake --build build --config Release -j4
bash models/download-ggml-model.sh tiny

# 5. Install Ollama and pull the model
curl -fsSL https://ollama.com/install.sh | sh
ollama serve & && sleep 4
ollama pull qwen2.5:1.5b

# 6. Download the Piper voice models
cd ~/pilingual-pi/voices
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/sk/sk_SK/lili/medium/sk_SK-lili-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/sk/sk_SK/lili/medium/sk_SK-lili-medium.onnx.json
```

## Tweaking the Configuration

You can change the default settings at the top of `translator_pi.py`:

| Variable | Default Value | Notes |
|---|---|---|
| `MIC_DEVICE` | `1` | Your USB mic index. Run `python3 -c "import sounddevice as sd; print(sd.query_devices())"` to find it. |
| `SPEAKER_CARD` | `hw:2,0` | The 3.5mm jack. Run `aplay -l` to find the correct card. |
| `RECORD_SECONDS` | `6` | How long to record for each interaction. |
| `SILENCE_THRESHOLD` | `0.001` | Drop this number if the mic isn't picking up your voice. |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | If you have an 8GB Pi, you can bump this to `qwen2.5:3b` for better translations. |

## What's in the repo

```
pilingual-pi/
  app.py              # Flask server, handles SSE and the web UI
  translator_pi.py    # The main engine for recording, transcribing, translating, and speaking
  setup_pi.sh         # Helper script to compile whisper.cpp and grab models
  requirements_pi.txt # Standard Python dependencies
  architecture.html   # The interactive architecture diagram
  voices/
    en_US-lessac-medium.onnx       # English TTS model
    en_US-lessac-medium.onnx.json
    sk_SK-lili-medium.onnx         # Slovak TTS model
    sk_SK-lili-medium.onnx.json
  piper/
    piper                          # The ARM64 TTS binary the setup script grabs
```

## License

MIT
