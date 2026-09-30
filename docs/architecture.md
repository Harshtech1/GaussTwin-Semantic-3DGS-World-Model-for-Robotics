# Architecture

GaussTwin is organized around replaceable stage boundaries: reconstruction produces calibrated cameras and geometry; a Gaussian backend consumes that representation; semantic fusion attaches open-vocabulary features; scene graph construction aggregates objects and relations; spatial reasoning serves language and robot queries.

`src/gausstwin/gaussian` will define backend-neutral contracts, with `gsplat` as the first planned GPU backend. Configuration owns paths and device selection. No module should hard-code Kaggle, CUDA devices, or data locations.

Current implementation covers configuration and environment inspection only. All scientific pipeline modules are explicit placeholders.
