"""Run the MPLNet NWPU experiment for seeds 1, 2, and 3.

This launcher reuses the settings in ``run_pycharm.py`` (especially DATA_DIR)
but forces a 50-epoch run for each seed. Every seed runs in a separate Python
process so CUDA memory and random state are fully reset between experiments.
The complete console output and the final accuracy summary are written to one
timestamped log file under ``logs``.
"""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


SEEDS = (1, 2, 3)
MAX_EPOCH = 50

PROJECT_ROOT = Path(__file__).resolve().parent
PYCHARM_LAUNCHER = PROJECT_ROOT / "run_pycharm.py"
LOG_DIR = PROJECT_ROOT / "logs"

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TOP1_MULTI_RE = re.compile(
    r"Evaluation metrics:.*?\btop1_multi:\s*([0-9]+(?:\.[0-9]+)?)"
)


def _child_code(seed: int) -> str:
    return (
        "import run_pycharm\n"
        f"run_pycharm.SEED = {seed}\n"
        f"run_pycharm.MAX_EPOCH = {MAX_EPOCH}\n"
        "run_pycharm.main()\n"
    )


def main() -> int:
    if not PYCHARM_LAUNCHER.is_file():
        raise FileNotFoundError(
            f"Cannot find {PYCHARM_LAUNCHER}. Put this script in the MPLNet "
            "repository root."
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    combined_log = LOG_DIR / f"three_seeds_{timestamp}.log"

    results: Dict[int, Optional[float]] = {}
    return_codes: Dict[int, int] = {}
    started_at = time.time()

    with combined_log.open("w", encoding="utf-8", newline="\n") as log_file:

        def emit(message: str = "") -> None:
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        emit("MPLNet three-seed experiment")
        emit(f"Project: {PROJECT_ROOT}")
        emit(f"Python: {sys.executable}")
        emit(f"Seeds: {SEEDS}")
        emit(f"Epochs per seed: {MAX_EPOCH}")
        emit(f"Combined log: {combined_log}")

        for index, seed in enumerate(SEEDS, start=1):
            emit()
            emit("=" * 80)
            emit(f"Starting seed {seed} ({index}/{len(SEEDS)})")
            emit("=" * 80)

            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUNBUFFERED"] = "1"

            metric: Optional[float] = None
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", _child_code(seed)],
                cwd=str(PROJECT_ROOT),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
                    log_file.flush()

                    plain_line = ANSI_ESCAPE_RE.sub("", line)
                    match = TOP1_MULTI_RE.search(plain_line)
                    if match:
                        metric = float(match.group(1))
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                emit()
                emit(f"Interrupted while running seed {seed}.")
                emit(f"Partial log saved to: {combined_log}")
                return 130

            return_code = process.wait()
            return_codes[seed] = return_code
            results[seed] = metric

            if return_code == 0 and metric is not None:
                emit(f"Seed {seed} completed: top1_multi = {metric:.3f}")
            elif return_code != 0:
                emit(f"Seed {seed} failed with exit code {return_code}.")
            else:
                emit(
                    f"Seed {seed} finished, but top1_multi could not be "
                    "parsed from its output."
                )

        elapsed = time.time() - started_at
        successful = [
            results[seed]
            for seed in SEEDS
            if return_codes.get(seed) == 0 and results.get(seed) is not None
        ]

        emit()
        emit("#" * 80)
        emit("FINAL SUMMARY")
        emit("#" * 80)
        for seed in SEEDS:
            value = results.get(seed)
            if return_codes.get(seed) == 0 and value is not None:
                emit(f"seed {seed}: top1_multi = {value:.3f}")
            else:
                emit(
                    f"seed {seed}: FAILED "
                    f"(exit code {return_codes.get(seed, 'unknown')})"
                )

        if successful:
            mean_value = statistics.fmean(successful)
            std_value = statistics.pstdev(successful)
            emit(f"mean top1_multi = {mean_value:.3f}")
            emit(f"std  top1_multi = {std_value:.3f}")
        else:
            emit("No successful result was available for aggregation.")

        emit(f"elapsed time = {elapsed / 60:.1f} minutes")
        emit(f"combined log = {combined_log}")

    return 0 if len(successful) == len(SEEDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
