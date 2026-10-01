# GaussTwin-002 — adaptive 3DGS baseline

GaussTwin-001 plateaued because its 43,188 COLMAP-initialized Gaussians were fixed: optimization could alter their attributes but could not allocate more representation capacity to persistent high-error regions. GaussTwin-002 preserves the same COLMAP input, local-neighborhood scale initialization, gsplat renderer, single-GPU execution, and RGB loss, but adds conservative structural adaptation.

At scheduled intervals after warm-up, it ranks visible Gaussians by mean screen-space gradient (`meta["means2d"].absgrad`) normalized by observation count. High-gradient, sufficiently observed Gaussians are split or duplicated subject to a count cap. Low-opacity, sufficiently observed Gaussians can be pruned while respecting a minimum retained count. Adam is rebuilt after a structural event; version 1 deliberately does not migrate optimizer moments.

This remains a reconstruction-only baseline. Semantic features, language, scene graphs, robotics, 4DGS, and multi-GPU execution are out of scope.
