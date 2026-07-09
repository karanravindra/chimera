# Working agreements

- **If the user says "sweep," run an actual wandb sweep** (`wandb sweep <file>.yaml` + `wandb agent`) — do not substitute a hand-rolled sequential loop of direct `train.py` invocations, even if it seems more controllable (e.g. for naming). If a sweep's grid parameters can't express something (like per-combo run names), solve it within the sweep (e.g. a wandb name template, or a thin wrapper `program` that derives `--run-name` from the swept args) rather than dropping the sweep abstraction entirely.
