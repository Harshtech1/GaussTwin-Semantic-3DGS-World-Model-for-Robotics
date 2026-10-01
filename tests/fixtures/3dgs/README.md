# Tiny 3DGS fixture

`scene.json` is a version-control-safe description of a deterministic COLMAP binary sparse model: one PINHOLE camera, one image pose, and two colored sparse points. The integration test materializes the three binary COLMAP files in a temporary directory because binary bytes are not patch-friendly source assets.

`images/frame.ppm` is an 8×8 RGB source image. The CPU integration test validates the COLMAP model and optimization contract without invoking Pillow or gsplat. Actual image decoding and gsplat rasterization remain covered by a Kaggle GPU run.
