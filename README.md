# MOVIN OSC Receiver & Viewer v3.3.0 (Python)

Python project for receiving MOVIN OSC motion and point cloud packets and visualizing them in real time.

**Motion and point cloud reception supports MOVIN Studio v3.0.0 and later.**
The viewer's received FPS and viewer FPS also work with older Studio versions.
Displaying connection status and FPS in Studio requires **MOVIN Studio v3.3.0 or later**.

The v3.0.0, v3.1.0 and v3.2.0 sender formats were checked against their source tags;
legacy-format packets were tested over loopback UDP. Those older app binaries were
not run during release validation.

## Download and Requirements

Download and extract [MOVIN-OSC-Receiver-v3.3.0.zip](https://github.com/MOVIN3D/MOVIN-OSC-Receiver-python-V3/releases/download/v3.3.0/MOVIN-OSC-Receiver-v3.3.0.zip),
then open a terminal in the extracted folder. Install dependencies using Conda or
venv below and run `python main.py`. This is a Python distribution, not a standalone executable.

Verified on Windows with Python 3.11 and an OpenGL-capable display. Python 3.11 is
recommended; macOS and Linux were not tested for this release. The ZIP contains the
receiver, dependency files and user documentation; development tests and tools remain
in the repository. Release checksums and `verification.json` are separate downloads.

To update, close the running receiver, extract the new ZIP into a separate folder,
update dependencies if needed, then run the new `main.py`. Select **OSC** in Studio
and send to this computer's IPv4 address on UDP `11235` (or the configured port).

## Features

- Receives `/MOVIN/Frame` skeleton chunks
- Reassembles chunked motion frames
- Receives `/MOVIN/PointCloud` point cloud chunks
- Visualizes skeleton and point cloud together in one `pygame` + `PyOpenGL` window
- Supports up to eight actors from one Studio sender, with different skeleton colors
- Discards duplicate and late frames, bounds incomplete frames, and expires stale motion and point clouds
- Shows per-joint local axes in RGB
- Shows actor/character names, received bone counts, and point count in the upper-left panel
- Shows received motion FPS per actor, received point cloud FPS, and viewer FPS; reports the same measurements to Studio
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

`boneIndex` and `parentIndex` refer to the original rig. They can be sparse when Hand streaming is off; `totalBoneCount` is the number of transmitted bones, not the maximum index. A parent of `-1` denotes a root. Every other parent must be included in the same frame. Bone indices and names must be unique.

The current Studio sender omits finger bones and their descendants when Hand is off. It keeps the wrists and splits OSC motion and point clouds into datagrams of at most 1,200 bytes. Addresses and argument order are unchanged from the earlier sender. Update this receiver together with Studio to support the sparse bone indices.

### `/MOVIN/PointCloud`

Header:

`[frameIdx, totalPoints, chunkIdx, numChunks, chunkPointCount]`

Per-point payload:

`[x, y, z]`

Where:

- `x, y, z` are the world-coordinate position.

Motion transforms and points use Unity coordinates; the viewer converts them to its OpenGL coordinates. A point cloud frame with zero points explicitly clears the display. Motion and point cloud frame numbers are independent.

## Receiving and Recovery

- One fixed receive thread handles OSC packets. Rendering uses the latest complete frames without copying the full point cloud each refresh.
- The first valid sender IP and UDP source port are selected. Other senders cannot contribute chunks to its frames. A different endpoint can take over after one second without packets from the selected sender.
- Frames are only published when all chunks, counts, and the bone hierarchy are valid. Late and duplicate frames are discarded. A lower frame index can start a new stream after one second without a completed advancing frame; this does not require a session ID.
- At most three incomplete frames are retained per actor and for the point cloud. Incomplete frames expire after 0.5 seconds, even if no further packets arrive.
- Both the skeleton and point cloud disappear after `--timeout` seconds without a completed frame (default: one second).
- Limits: 4,096 bones per actor, 100,000 points per cloud frame, 4,096 chunks per frame, and 256 characters per actor/bone name. Invalid counts, types, non-finite transforms, duplicate bones, missing parents, and cyclic hierarchies are rejected. Invalid-packet warnings are limited to one per second.
- Existing larger datagrams are still readable up to the UDP datagram limit. Smaller chunks are recommended to avoid IP fragmentation.
- Studio requests status once per second and receives replies through its shared UDP port (39581). The OSC panel shows connection status, Source, received Motion/Pointcloud FPS, and Viewer FPS. A missing reply never stops streaming.
- Received FPS counts completed, accepted frames, not packets or chunks; duplicate and incomplete frames do not count. Viewer FPS counts rendered frames independently. Rates use the last second and fall to zero when updates stop. The first frame alone reports zero until a frame interval is available.

### Optional status replies

`/MOVIN/OSC/Status/Request [token:string, replyPort:int, actorName:string]`

`/MOVIN/OSC/Status [token:string, version:int, actorName:string, motionFPS:float, pointcloudFPS:float, viewerFPS:float, motionAge:float, pointcloudAge:float, viewerAge:float, sameSource:int]`

Version is `1`; token is a 32-character hexadecimal request ID. Ages are seconds since the latest completed/rendered frame, or `-1` when unavailable. Motion FPS is for the requested actor. `sameSource` is `1` only when the request comes from the selected motion/point cloud sender's IP and UDP source port. Status probes do not select or keep alive a stream source. Replies go to the requester's IP and specified reply port using the existing receive socket.

`connected` means the receiver responds; it does not mean motion is arriving or the viewer is rendering. FPS and frame ages distinguish those states. Other OSC receivers can ignore status requests and continue receiving the unchanged motion/point cloud format.

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

## Tests

From a repository checkout with the dependencies installed, run `python -m unittest -v test_receiver`. Tests include real loopback UDP, sparse indices, reordering, restart recovery, malformed packets, source isolation, and bounded frame storage.

Maintainers can build the user ZIP and SHA-256 manifest with
`python tools/package_release.py`. Update `VERSION`, this README and `CHANGELOG.md`
together. Only publish packages whose manifest records a clean source commit;
verify the extracted receiver and viewer before creating a release tag.

## License

Copyright 2025 MOVIN. All Rights Reserved.
