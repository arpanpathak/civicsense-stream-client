# 🧠 CivicSense Stream Client: YOLOv8 Object Detection in Pure Rust

A Rust client that connects to a Raspberry Pi Zero 2 W MJPEG stream, runs **YOLOv8n object detection on every frame using Candle (pure Rust ML)**, draws bounding boxes, and saves annotated frames to disk. No Python, no ONNX Runtime, no GPU required, just a single native binary.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![ML](https://img.shields.io/badge/ML-Candle%20(YOLOv8n)-D97757)](https://github.com/huggingface/candle)
[![Platform](https://img.shields.io/badge/Platform-macOS%2FLinux-A22846)](https://www.rust-lang.org)
[![CivicSense](https://img.shields.io/badge/CivicSense-Part%20of%20the%20ecosystem-8A2BE2)](https://github.com/arpanpathak/driving-civicsense-vision-model)

> Part of the [CivicSense](https://github.com/arpanpathak/driving-civicsense-vision-model) ecosystem: a privacy-first, edge-native AI vision system for intersection discipline and road civility. This client consumes the [Pi stream server](https://github.com/arpanpathak/civicsense-pi-stream) and detects vehicles on device, so no video ever leaves your hardware.

---

## 📦 What's Inside

| Component | Detail |
|---|---|
| Stream source | MJPEG over HTTP (`multipart/x-mixed-replace`) from a Pi Zero 2 W |
| Detection | YOLOv8n via `candle-transformers`, pure Rust inference |
| Post-processing | Non-maximum suppression, COCO 80-class labels |
| Rendering | Bounding boxes drawn with `imageproc` |
| Output | Annotated JPEG frames saved to `output/` |
| Runtime | Async: `reqwest` + `tokio`, streaming with `futures-util` |

**Why Candle?** It is a fully pure-Rust ML framework from Hugging Face. That means the whole pipeline (network, inference, drawing) is one Rust binary with no external runtime to install, which fits the CivicSense privacy-first philosophy.

---

## ✨ Features

- **Fetches MJPEG** from the Pi over HTTP multipart, frame by frame
- **Parses JPEG frames** from the stream on the fly
- **Runs YOLOv8n** object detection with `candle-transformers`
- **Applies non-maximum suppression** to remove duplicate boxes
- **Draws bounding boxes** with class labels on every frame
- **Saves annotated frames** to the local `output/` directory

---

## 🔧 Requirements

- macOS or Linux (x86_64 or aarch64), also runs on the Pi itself
- Rust (latest stable)
- A YOLOv8n model in **safetensors** format (not ONNX)

---

## 🚀 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/arpanpathak/civicsense-stream-client.git
   cd civicsense-stream-client
   ```

2. **Download a YOLOv8n safetensors model**

   The model is not bundled (keep the repo permissive). Export one from Ultralytics or grab a safetensors checkpoint from the Candle model hub, then place it at:

   ```text
   models/yolov8n.safetensors
   ```

3. **Update the stream URL**

   In `src/main.rs`, point the client at your Pi stream (default: `http://192.168.0.43:8000`).

4. **Build**

   ```bash
   cargo build --release
   ```

5. **Run**

   ```bash
   ./target/release/pistream_client
   ```

   Watch the annotated frames appear in `output/`.

---

## 🖼️ How It Works

```
Pi Zero 2 W (civicsense-pi-stream)          Mac / edge device (this client)
┌────────────────────────────┐              ┌────────────────────────────────┐
│  Arducam IMX335            │   MJPEG      │  reqwest stream                │
│  rpicam-vid                │ ───────────▶ │  JPEG frame parse              │
│  multipart/x-mixed-replace │              │  YOLOv8n (Candle) inference    │
└────────────────────────────┘              │  NMS + COCO labels             │
                                            │  Bounding boxes (imageproc)    │
                                            │  Save annotated frame          │
                                            └────────────────────────────────┘
```

---

## 🤝 Credits

- [Hugging Face Candle](https://github.com/huggingface/candle) for the pure-Rust ML framework
- Ultralytics for YOLOv8 architecture and weights
- [civicsense-pi-stream](https://github.com/arpanpathak/civicsense-pi-stream) for the camera stream

## 📄 License

MIT, do whatever you want with this code. See [LICENSE](LICENSE).

> Note: the YOLOv8 weights are not included in this repository. Download them from Ultralytics and review their license separately.
