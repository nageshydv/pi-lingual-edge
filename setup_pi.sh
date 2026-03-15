#!/bin/bash
set -e

echo ""
echo "===================================================="
echo "  pi-Lingual — Raspberry Pi 4 Setup"
echo "===================================================="
echo ""

# ── 1. System packages ────────────────────────────────────────────────
echo "[1/6] System packages..."
sudo apt update -qq
sudo apt install -y \
  python3-pip python3-venv python3-dev \
  portaudio19-dev \
  ffmpeg \
  alsa-utils \
  curl git \
  libespeak-ng1          # needed by piper binary at runtime

# ── 2. Python version check ──────────────────────────────────────────
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python version: $PYVER"

# ── 3. Piper standalone binary (ARM64) ───────────────────────────────
echo ""
echo "[2/6] Downloading Piper TTS binary (ARM64)..."
mkdir -p piper
PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz"
curl -# -L "$PIPER_URL" -o /tmp/piper.tar.gz
tar -xzf /tmp/piper.tar.gz -C piper --strip-components=1
rm /tmp/piper.tar.gz

if [ ! -f "piper/piper" ]; then
  echo "  ERROR: piper binary not found after extraction."
  echo "  Check the release URL or download manually:"
  echo "  https://github.com/rhasspy/piper/releases"
  exit 1
fi

chmod +x piper/piper
echo "  Piper binary ready: $(piper/piper --version 2>/dev/null || echo 'ok')"

# ── 4. Python venv + packages ────────────────────────────────────────
echo ""
echo "[3/6] Python environment..."
python3 -m venv venv
./venv/bin/pip install --quiet --upgrade pip
./venv/bin/pip install --quiet -r requirements_pi.txt
echo "  Python packages installed."

# ── 5. Ollama + model ────────────────────────────────────────────────
echo ""
echo "[4/6] Ollama..."
if ! command -v ollama &>/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "  Ollama already installed."
fi

pkill ollama 2>/dev/null || true
sleep 2
ollama serve > /tmp/ollama_pi.log 2>&1 &
OLLAMA_PID=$!
echo "  Waiting for Ollama..."
for i in $(seq 1 20); do
  sleep 2
  ollama list &>/dev/null 2>&1 && echo "  Ready." && break
  [ $i -eq 20 ] && echo "  ERROR: Ollama failed. Check /tmp/ollama_pi.log" && exit 1
  echo "  Still waiting... ($i/20)"
done

echo "  Pulling qwen2.5:1.5b (~1GB)..."
ollama pull qwen2.5:1.5b
echo "  Model ready."

# ── 6. Voice models ───────────────────────────────────────────────────
echo ""
echo "[5/6] Piper voice models..."
mkdir -p voices
cd voices

echo "  English voice..."
curl -# -L -o en_US-lessac-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
curl -# -L -o en_US-lessac-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

echo "  Slovak voice..."
curl -# -L -o sk_SK-lili-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/sk/sk_SK/lili/medium/sk_SK-lili-medium.onnx"
curl -# -L -o sk_SK-lili-medium.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/sk/sk_SK/lili/medium/sk_SK-lili-medium.onnx.json"

cd ..

# Validate voice JSON
for f in voices/*.onnx.json; do
  python3 -c "import json; json.load(open('$f'))" 2>/dev/null || {
    echo "  WARNING: $f corrupt — re-downloading..."
    rm -f "$f"
    fname=$(basename "$f")
    if [[ "$fname" == en* ]]; then
      curl -# -L -o "voices/$fname" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/$fname"
    elif [[ "$fname" == sk* ]]; then
      curl -# -L -o "voices/$fname" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/sk/sk_SK/lili/medium/$fname"
    fi
  }
done

# ── 7. Quick smoke test ───────────────────────────────────────────────
echo ""
echo "[6/6] Smoke tests..."

# Test piper binary
echo "  Testing Piper..."
echo "hello" | ./piper/piper \
  --model voices/en_US-lessac-medium.onnx \
  --output_file /tmp/piper_test.wav \
  --quiet 2>/dev/null && \
  [ -f /tmp/piper_test.wav ] && \
  [ $(stat -c%s /tmp/piper_test.wav 2>/dev/null || stat -f%z /tmp/piper_test.wav) -gt 44 ] && \
  echo "  Piper OK" || echo "  WARNING: Piper test failed — check voice files"
rm -f /tmp/piper_test.wav

# Test Ollama
echo "  Testing Ollama..."
RESULT=$(curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:1.5b","prompt":"say hi","stream":false}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK' if d.get('response') else 'FAIL')" 2>/dev/null)
echo "  Ollama: $RESULT"

# Audio devices
echo "  Audio devices:"
arecord -l 2>/dev/null | grep card || echo "  (no capture devices found — check USB mic)"
aplay  -l 2>/dev/null | grep card || echo "  (no playback devices found)"

PI_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "===================================================="
echo "  Setup complete!"
echo ""
echo "  Start:"
echo "    source venv/bin/activate"
echo "    ollama serve &"
echo "    python3 app.py"
echo ""
echo "  Open browser:"
echo "    http://localhost:5000"
echo "    http://${PI_IP}:5000   (from another device)"
echo ""
echo "  If wrong mic: edit MIC_DEVICE in translator_pi.py"
echo "===================================================="
