# Privacy-First Face Presence with Homomorphic Encryption

## System Architecture

```mermaid
flowchart LR
    subgraph enrollment["ENROLLMENT (One-time)"]
        direction TB
        A[Enrollment Camera] --> B[Enrollment Hub]
        B --> C[Generate Files:<br/>• HE Keys<br/>• Projection<br/>• Samples Database<br/>]
    end

    subgraph runtime["RECOGNITION (Runtime)"]
        direction TB
        D[Camera Node<br/>Capture + Detect Face] -->|🔒 Encrypted 128D| E[HE Server<br/>Compare Encrypted]
        E -->|🔒 Encrypted Scores| D
        E --> F[Decrypt + Log:<br/>PRESENT/NOT]

    end

    C -.->|Copy Keys| D
    C -.->|Copy Keys and<br/> Face Embeddings| E

    style A fill:#e1f5ff
    style D fill:#e1f5ff
    style E fill:#ffe1e1
    style C fill:#e1ffe1
    style F fill:#d4edda
```

## Overview

This system demonstrates privacy-preserving face recognition where **raw video never leaves the device**. The camera detects faces locally, computes encrypted embeddings using homomorphic encryption (HE), and sends only ciphertexts to the server. The server performs encrypted similarity matching without ever seeing plaintext data. This project aims to solve the problem of attackers being able to tap into your live camera feeds (assuming that you only use the cameras to detect faces).

### Key Privacy Features

- **No raw video transmission** - Only encrypted 128-D vectors leave the camera
- **Homomorphic computation** - Server compares faces while data remains encrypted
- **End-to-end encryption** - CKKS homomorphic encryption protects data in transit

### Security Considerations

**Protected:**
- Network traffic is encrypted (CKKS ciphertexts)
- Server cannot view raw video or plaintext embeddings
- Minimal attack surface for a live feed hijack like in movies

**Limitations:**
- Secret key (`he_secret_ctx.bin`) must be protected, and also somehow securely passed between systems at setup.
- Hardware - someone could tamper with the camera hardware and tap into the raw live feed

---

## System Components

### 1. Enrollment Hub
Generates HE contexts, captures face samples, and creates the enrollment database.

**Responsibilities:**
- Create CKKS encryption context (one-time setup)
- Generate random projection matrix (512-D → 128-D)
- Capture and average multiple face samples per identity
- Store normalized embeddings in `faceid_db_server.json`

### 2. Camera Node
Captures video, detects faces locally, encrypts embeddings, and sends to server.

**Responsibilities:**
- Local face bounding box detection and embedding computation
- Project embeddings (512-D -> 128-D) and normalize
- Encrypt with CKKS and transmit only ciphertexts
- Display HUD with detection feedback (demo only)

### 3. HE Server
Receives encrypted probes and performs homomorphic similarity matching.

**Responsibilities:**
- Load enrollment database
- Compute homomorphic comparison operations
- Decrypt scores locally and log detections
- Return encrypted scores to camera

---

## Quick Start

### Installation

```bash
# Navigate to project directory
cd ~/classes/321/mastery_extension/client_server

# Activate your virtual environment
source .venv/bin/activate

# Install the package in editable mode
pip install -e .
```

This installs the `faceid` package and creates three command-line tools:
- `faceid-enroll` - Enrollment Hub
- `faceid-camera` - Camera Node
- `faceid-server` - HE Server

### Basic Setup (Local Camera)

**Step 1: Enroll an identity**

```bash
faceid-enroll --enroll-name "Alice" --input camera --source 0 --avg-samples 5
```

- Press `e` to capture samples (5 times from different angles)
- Press `q` to quit when done
- Creates: `he_public_ctx.bin`, `he_secret_ctx.bin`, `he_proj.npy`, `faceid_db_server.json`

**Step 2: Start the server**

```bash
faceid-server
```

- Prints context fingerprint and listens on `127.0.0.1:9009`

**Step 3: Start the camera node**

```bash
faceid-camera --input camera --source 0 --he-send-fps 1
```

- Opens preview window (local only)
- Sends encrypted probes every ~1 second
- Server logs: `Person detected: Alice (score 0.92) -> PRESENT`

---

## Advanced: WSL/Network Mode

If your camera is on Windows and you're running the system in WSL/Linux, stream video over UDP with FFmpeg.

### Windows Side (PowerShell)

```powershell
ffmpeg -f dshow -i video="Integrated Camera" `
  -vf scale=640:-1,fps=15 `
  -vcodec libx264 -preset ultrafast -tune zerolatency `
  -g 30 -x264-params "keyint=30:min-keyint=30:scenecut=0:repeat-headers=1" `
  -b:v 1.2M -maxrate 1.2M -bufsize 2.4M `
  -pix_fmt yuv420p `
  -f mpegts ("udp://{0}:5000?pkt_size=1316" -f (
    (wsl hostname -I) -split '\s+' |
    ? { $_ -match '^\d{1,3}(\.\d{1,3}){3}$' } |
    select -First 1
  ))
```

### WSL/Linux Side

```bash
# Enroll from UDP stream
faceid-enroll --enroll-name "Alice" --input udp --udp-url "udp://@0.0.0.0:5000" --avg-samples 5

# Start server
faceid-server

# Start camera node (reading UDP, sending encrypted probes)
faceid-camera --input udp --udp-url "udp://@0.0.0.0:5000" --he-send-fps 1
```

**Troubleshooting UDP:**
- Verify Windows firewall allows outbound UDP to WSL IP:5000
- Test reception: `ffplay -fflags nobuffer -flags low_delay -i udp://@0.0.0.0:5000`

---

## Enrollment Best Practices

### Multi-Sample Averaging (Recommended)

Capture 3-5 samples per identity from slightly different angles/lighting:

1. Press `e` when face is centered and well-lit
2. Rotate head slightly (small yaw/pitch changes)
3. System auto-finalizes after N samples (default: 5)

**Why average?** Single embeddings vary with micro-expressions and noise. Averaging produces stable templates with better separation between identities.

### Enrollment Controls

- `e` - Capture sample
- `f` - Finalize early (average current samples)
- `r` - Reset sample buffer
- `s` - Save snapshot
- `q` - Quit

### Quality Filters

- Minimum bbox area: 9000 pixels (configurable via `--min-bbox-area`)
- Small/blurry faces are automatically skipped

---

## Tuning Recognition

All configuration is in `config.yaml`. Edit values and restart components to apply changes.

### Threshold Adjustment

```yaml
threshold: 0.9  # Higher = stricter, Lower = more lenient
```

Typical ranges:
- `0.95+` - Very strict, minimal false positives
- `0.85-0.90` - Balanced (recommended)
- `<0.80` - Lenient, may trigger on similar faces

### Projection Dimension

```yaml
proj_dim: 128  # Default: fast HE operations, good accuracy
```

**Note:** Changing `proj_dim` requires re-enrollment (delete `he_proj.npy` and `faceid_db_server.json`).

### Context Consistency

All three components must use the **same HE context**. Verify fingerprints match:

```
[HUB] Using existing HE context. Public FP=a1b2c3d4e5f6g7h8
[SERVER] Loaded cached public context.
[CAM] Loaded HE context. Public FP=a1b2c3d4e5f6g7h8
```

If mismatched, copy `he_public_ctx.bin` and `he_secret_ctx.bin` from Enrollment Hub to all nodes.

---

## File Reference

| File | Description | Security |
|------|-------------|----------|
| `he_public_ctx.bin` | CKKS public context | Safe to share |
| `he_secret_ctx.bin` | CKKS secret key | **Keep secure!** |
| `he_proj.npy` | Projection matrix (512→128) | Safe to share |
| `faceid_db_server.json` | Enrollment database | **Biometric data - protect** |
| `config.yaml` | System configuration | Safe to share |

**Deployment:** Copy `he_public_ctx.bin`, `he_secret_ctx.bin`, `he_proj.npy`, and `faceid_db_server.json` from Enrollment Hub to Server and Camera nodes.

---

## Command Reference

### Enrollment Hub

```bash
faceid-enroll \
  --enroll-name "Name" \
  --input {camera|udp} \
  --source 0 \
  --avg-samples 5 \
  --min-bbox-area 9000 \
  --device {auto|gpu|cpu}
```

### Camera Node

```bash
faceid-camera \
  --input {camera|udp} \
  --source 0 \
  --he-send-fps 1.0 \
  --server-host 127.0.0.1 \
  --server-port 9009 \
  --device {auto|gpu|cpu}
```

### HE Server

```bash
faceid-server
```

Server configuration is in `config.yaml`.

---

## Troubleshooting

**Random huge scores (±1e7):**
- HE context mismatch. Copy `he_secret_ctx.bin` from Enrollment Hub to all nodes and restart.

**ConnectionRefusedError:**
- Start server first
- Verify `--server-host` and `--server-port` match
- Check firewall allows TCP/9009

**No video on UDP:**
- Confirm FFmpeg is running on Windows
- Verify destination IP is correct (WSL IP, not 127.0.0.1)
- Test with: `ffplay -fflags nobuffer -flags low_delay -i udp://@0.0.0.0:5000`

**CUDA not available:**
- Check `onnxruntime-gpu` is installed
- Verify CUDA/CuDNN compatibility
- Use `--device cpu` as fallback

---

## Performance Notes

- **GPU recommended** for face detection/embedding
- **CPU fallback** available but much slower
- **HE operations** are CPU-bound but fast on 128-D vectors
- **Send frequency** of 1 Hz balances responsiveness and compute load (set lower or higher depending on needs and hardware)

---