"""Lightweight local experiment tracker (ClearML-style scalar logging,
without needing an account or a server).

Every metric logged during a run ends up in two places inside the run
directory:
  * metrics.csv   -- tidy long-format table (step, epoch, phase, metric,
                      value), easy to reload with pandas for later analysis.
  * TensorBoard event files -- live/interactive loss & metric curves, viewed
                      with `tensorboard --logdir models/<run>/tensorboard`.
TensorBoard logging is best-effort: if the `tensorboard` package isn't
installed, it is silently skipped and only the CSV is written.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsLogger:
    def __init__(self, run_dir: str | Path, use_tensorboard: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["step", "epoch", "phase", "metric", "value"])

        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except ImportError:
                logger.warning(
                    "tensorboard package not installed, skipping TensorBoard logging "
                    "(metrics.csv is still written). Install it with `pip install tensorboard`."
                )

    def log(self, metric: str, value: float, step: int, epoch: int | None = None, phase: str = "train") -> None:
        self._csv_writer.writerow([step, epoch, phase, metric, value])
        self._csv_file.flush()
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(f"{phase}/{metric}", value, step)

    def log_many(self, metrics: dict[str, float], step: int, epoch: int | None = None, phase: str = "train") -> None:
        for name, value in metrics.items():
            self.log(name, value, step=step, epoch=epoch, phase=phase)

    def log_group(self, group: str, values: dict[str, float], step: int, epoch: int | None = None) -> None:
        """Logs several related scalars (e.g. {"train": ..., "val": ...})
        that should be overlaid as separate lines on ONE TensorBoard chart.
        A plain `log()` per phase would put them on separate charts instead
        (TensorBoard groups by the tag prefix), so this uses `add_scalars`,
        which is the API meant for train-vs-val comparison plots.
        """
        for name, value in values.items():
            self._csv_writer.writerow([step, epoch, group, name, value])
        self._csv_file.flush()
        if self._tb_writer is not None:
            self._tb_writer.add_scalars(group, values, step)

    def close(self) -> None:
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
