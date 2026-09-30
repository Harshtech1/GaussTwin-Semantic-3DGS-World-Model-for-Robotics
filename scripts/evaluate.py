#!/usr/bin/env python3
import _bootstrap  # noqa: F401
from gausstwin.cli import planned_stage

raise SystemExit(planned_stage("evaluation", "configs/base.yaml"))
