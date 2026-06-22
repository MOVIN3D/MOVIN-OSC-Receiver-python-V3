# MOVIN OSC Receiver & Viewer (Python)

Python project for receiving MOVIN OSC motion and point cloud packets and visualizing them in real time.

## Features

- Receives `/MOVIN/Frame` skeleton chunks
- Reassembles chunked motion frames
- Receives `/MOVIN/PointCloud` point cloud chunks
- Visualizes skeleton and point cloud together in one `pygame` + `PyOpenGL` window
- Supports multiple actors with different skeleton colors
- Shows per-joint local axes in RGB
- Handles non-UTF-8 OSC strings, including common Korean encodings

## OSC Formats

### `/MOVIN/Frame`

Header:

`[timestamp, actorName, frameIdx, numChunks, chunkIdx, totalBoneCount, chunkBoneCount]`

Per-bone payload:

`[boneIndex, parentIndex, boneName, px, py, pz, rqx, rqy, rqz, rqw, qx, qy, qz, qw, sx, sy, sz]`

Where:

- `px, py, pz` are the local position.
- `rqx, rqy, rqz, rqw` are the rest-pose local rotation.
- `qx, qy, qz, qw` are the local rotation.
- `sx, sy, sz` are the local scale.

### `/MOVIN/PointCloud`

Header:

`[frameIdx, totalPoints, chunkIdx, numChunks, chunkPointCount]`

Per-point payload:

`[x, y, z]`

Where:

- `x, y, z` are the world-coordinate position.

## Install With Conda

```powershell
conda env create -f environment.yml
conda activate movin-osc
```

If you prefer to update an existing environment:

```powershell
conda env update -f environment.yml --prune
conda activate movin-osc
```

## Install With venv

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

Optional flags:

```powershell
python main.py --host 0.0.0.0 --port 11235 --point-size 3.0 --fps 60 --axis-size 0.08
```

Default values:

- `--port 11235`
- `--fps 60`
- `--point-size 3.0`
- `--axis-size 0.08`
- `--timeout 1.0`

Viewer controls:

- Left mouse drag: rotate camera
- Mouse wheel: zoom
- `R`: reset camera
- Up/Down arrows: move camera target vertically
- `Esc`: exit

## License

Copyright 2025 MOVIN. All Rights Reserved.
