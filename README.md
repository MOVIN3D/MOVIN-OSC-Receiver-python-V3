# MOVIN OSC Receiver & Viewer v3.3.0 (Python)

View motion and point clouds streamed from MOVIN Studio, with character names,
bone and point counts, received FPS and viewer FPS.

## Requirements

- **MOVIN Studio v3.0.0 or later** for reception, visualization and FPS in the viewer.
- **MOVIN Studio v3.3.0 or later** for connection status and FPS in Studio.
- Recommended setup: **Windows, Python 3.11**, and an OpenGL-capable display.

This is a Python application. Install its dependencies before running it.

## Installation

Download [MOVIN-OSC-Receiver-v3.3.0.zip](https://github.com/MOVIN3D/MOVIN-OSC-Receiver-python-V3/releases/download/v3.3.0/MOVIN-OSC-Receiver-v3.3.0.zip),
extract it, and open a terminal in that folder. Choose one setup method below.

### Conda

```powershell
conda env create -f environment.yml
conda activate movin-osc
```

### Python venv

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

1. With your environment activated, run:

   ```powershell
   python main.py
   ```

2. In Studio, select **OSC** and the Actor or Character to stream.
3. Set the destination to the receiver computer's IPv4 address, or `127.0.0.1` when
   both apps run on the same computer. Use port **11235**.
4. Click **Start Streaming**. Enable **Hand** or **Pointcloud** in Studio if needed.

Allow the receiver's UDP port (**11235** by default) through the firewall. To show
status in Studio on another computer, also allow replies on **UDP 39581** into Studio.

## Viewer Controls

| Control | Action |
|---|---|
| Left mouse drag | Rotate the camera |
| Mouse wheel | Zoom |
| `R` | Reset the camera |
| Up / Down arrows | Move the camera target vertically |
| `Esc` | Close the viewer |

The overlay shows each character's name, bone count and received motion FPS, plus
point count, received point cloud FPS and viewer FPS. Received FPS measures incoming
complete frames; viewer FPS measures how often the window renders.

## Options

For example, to use a different port:

```powershell
python main.py --port 11236
```

| Option | Default | Use |
|---|---|---|
| `--port` | `11235` | Receiving port; match it in Studio |
| `--fps` | `60` | Viewer refresh-rate target |
| `--point-size` | `3.0` | Point cloud display size |
| `--axis-size` | `0.08` | Joint-axis display length |
| `--timeout` | `1.0` | Seconds without data before clearing the display |

## Update

Close the old receiver and extract the new ZIP into a separate folder. Activate your
environment, update dependencies, then run the new `main.py`.

- Conda: `conda env update -f environment.yml --prune`
- venv: `pip install -r requirements.txt`

## Troubleshooting

| Problem | Check |
|---|---|
| Port already in use | Close another receiver using the port, or set a different port in both apps. |
| Waiting for motion | Select OSC in Studio, start streaming, and check the destination IP, port and firewall. |
| No point cloud | Enable Pointcloud in Studio. |
| Studio is connected but nothing moves | Check received FPS and confirm Studio is sending motion. |
| Viewer FPS is lower than received FPS | Reduce display workload or disable point cloud streaming when it is not needed. |
| Dependency or OpenGL startup error | Activate the intended Python environment, install requirements and check the graphics driver. |

## License

Copyright 2025 MOVIN. All Rights Reserved.
