"""Human-readable run log (D45).

Written to the console and to ``<case>.log`` with fixed-width columns,
readable with ``tail -f`` or ``more``.

Layout for steady runs: one line per pseudo-time iteration. (Unsteady runs,
stage 4, write one line per physical time step.)
"""

import sys
import time
from pathlib import Path

import numpy as np


class RunLog:
    """Write messages and iteration lines to the console and a log file.

    Parameters
    ----------
    path : Path or None
        Log file; None writes to the console only.
    console : bool
        Echo to stdout.
    """

    def __init__(self, path=None, console=True):
        self.path = Path(path) if path is not None else None
        self._file = open(self.path, "w") if self.path is not None else None
        self._console = console
        self._t0 = time.perf_counter()
        self._columns = None

    def close(self):
        """Close the log file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        """Support ``with RunLog(...) as log``."""
        return self

    def __exit__(self, *exc):
        """Close the file when leaving the ``with`` block."""
        self.close()

    def write(self, text=""):
        """Write a line."""
        if self._console:
            print(text)
            sys.stdout.flush()
        if self._file is not None:
            self._file.write(text + "\n")
            self._file.flush()

    def section(self, title):
        """Write a section heading."""
        self.write("")
        self.write(title)
        self.write("-" * len(title))

    def table(self, rows, indent=2):
        """Write ``key: value`` rows aligned on the colon."""
        width = max((len(k) for k, _ in rows), default=0)
        for key, value in rows:
            self.write(" " * indent + f"{key:<{width}}  {value}")

    def iteration_header(self, residual_names):
        """Write the column header of the iteration lines."""
        self._columns = list(residual_names)
        cols = "".join(f"{'log(R_' + n + ')':>12}" for n in self._columns)
        self.write("")
        self.write(f"{'iter':>6}{cols}{'CFL':>10}{'time[s]':>10}  note")

    def iteration(self, it, residuals, cfl, accepted=True, note=""):
        """Write one iteration line."""
        with np.errstate(divide="ignore"):
            logs = np.log10(np.asarray(residuals, dtype=float))
        cols = "".join(f"{v:>12.4f}" for v in logs)
        elapsed = time.perf_counter() - self._t0
        flag = note if accepted else ("rejected " + note).strip()
        self.write(f"{it:>6d}{cols}{cfl:>10.2e}{elapsed:>10.1f}  {flag}".rstrip())

    @property
    def elapsed(self):
        """Seconds since the log was opened."""
        return time.perf_counter() - self._t0
