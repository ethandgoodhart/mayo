# Alpamayo Live Predictions

Live webcam -> Alpamayo-R1 trajectory predictions, served from a Modal GPU container and rendered locally on your Mac.

## Prerequisites

- A [Modal](https://modal.com) account with the CLI installed and authenticated:
  ```bash
  pip install modal
  modal setup
  ```
- A HuggingFace token saved as a Modal secret named `huggingface` (with access to `nvidia/Alpamayo-R1-10B`).
- Python 3.10+ on your Mac with a working webcam.

## 1. Deploy the server (Modal, H100)

From the repo root:

```bash
modal deploy scripts/live_demo_server.py
```

Modal prints an HTTPS URL like `https://<your-app>.modal.run`. The WebSocket endpoint is that URL with `https` swapped to `wss` and `/ws` appended:

```
wss://<your-app>.modal.run/ws
```

For ephemeral dev runs, use `modal serve scripts/live_demo_server.py` instead.

## 2. Run the client (Mac)

Install client deps:

```bash
pip install opencv-python websockets msgpack numpy matplotlib
```

Then run, passing the WebSocket URL from step 1:

```bash
python3 scripts/live_demo_client.py wss://<your-app>.modal.run/ws
```

A window opens showing your webcam on the left and the predicted top-down trajectory (BEV) on the right. Press `ESC` or `Ctrl-C` to quit.

## Tearing down

```bash
modal app stop alpamayo-live-demo
```
