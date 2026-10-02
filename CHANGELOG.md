# Changelog

## 3.3.0 — 2026-10-02

First versioned receiver release. Motion and point clouds support MOVIN Studio
3.0.0 and later; Studio connection/FPS feedback requires Studio 3.3.0 or later.

- Reassemble sparse bone indices, reordered chunks and legacy larger datagrams.
- Validate frame counts, argument types, finite transforms and bone hierarchies;
  reject malformed, duplicate and stale frames without stopping reception.
- Isolate the active UDP sender and bound partial-frame storage. Recover after
  restarted frame indices and expire both motion and point cloud display.
- Use one receive thread and share the latest complete point cloud with rendering.
- Show actor/character names, bone and point counts, received motion/point cloud
  FPS and viewer FPS in the viewer overlay.
- Reply to optional Studio status requests through the receiving socket. Status
  requests do not select a sender or keep its stream alive.
- Add 24 receiver tests, including real UDP and legacy packet compatibility.
- Distribute a Python ZIP with version, source commit and SHA-256 records.

Verified on Windows/Python 3.11. Older Studio compatibility was checked from
tagged sender sources and representative UDP packets, not by running old binaries.
