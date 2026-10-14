"""Convergence history as CSV (D45)."""

import csv
from pathlib import Path


class HistoryWriter:
    """Append one CSV row per iteration.

    Parameters
    ----------
    path : Path
    residual_names : sequence of str
        One column ``residual_<name>`` per variable.
    """

    def __init__(self, path, residual_names):
        self.path = Path(path)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "iteration",
                *(f"residual_{n}" for n in residual_names),
                "cfl",
                "accepted",
                "time_s",
            ]
        )

    def row(self, it, residuals, cfl, accepted, elapsed):
        """Write one row."""
        self._writer.writerow(
            [
                it,
                *(f"{r:.6e}" for r in residuals),
                f"{cfl:.6e}",
                int(accepted),
                f"{elapsed:.3f}",
            ]
        )
        self._file.flush()

    def close(self):
        """Close the file."""
        self._file.close()
