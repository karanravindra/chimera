"""Mirror a run's stdout/stderr to its output log file, live.

Every training script calls :func:`tee_to_logfile` once, right after it knows its
run's output directory, so the run's console output is ALSO written to
``outputs/<run>/<run>.log`` as it happens -- logs and artifacts live together in one
per-run folder, no manual ``> logs/foo.log`` redirect needed, and ``tail -f`` shows it
live. The console stream is left intact, so output still appears wherever the process
was launched.
"""

import sys
from datetime import datetime
from pathlib import Path


class _Tee:
    """A write-through fan-out stream that flushes after every write so the log
    file updates live (``tail -f``-friendly), including tqdm's ``\\r`` redraws."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def tee_to_logfile(log_path, argv: list[str] | None = None) -> Path:
    """Duplicate ``sys.stdout``/``sys.stderr`` into ``log_path`` (created if needed).

    Pass the per-run path, e.g. ``RUN_DIR / f"{run}.log"``, so the log sits alongside
    that run's checkpoints/grids. Appends (a resumed run keeps its history) and writes a
    header line marking the start time and command. Returns the log path.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    f = open(log_path, "a", buffering=1)  # line-buffered text mode
    f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} :: "
            f"{' '.join(argv or sys.argv)} =====\n")
    f.flush()

    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return log_path
