"""Windows/PyCharm launcher for MPLNet.

Edit the values in the CONFIG section, then run this file directly from
PyCharm. The launcher executes ``mplnet/train.py`` in the same Python process,
so breakpoints inside the training code continue to work.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


# ============================== CONFIG ======================================
# This directory must contain subdirectories such as ``nwpu_resisc45`` or
# ``aid``. Example: D:\RemoteSensingDatasets\nwpu_resisc45
DATA_DIR = Path(r"D:\RemoteSensingDatasets")

DATASET_NAME = "nwpu_resisc45"  # nwpu_resisc45, aid, or rsd46
NUM_SHOTS = 1                    # 1, 2, 4, 8, or 16
SEED = 1                         # the paper uses seeds 1, 2, and 3

# Keep this at 1 for the first smoke test. Change it to 50 for the paper setup.
MAX_EPOCH = 1
BATCH_SIZE = 16
INFERENCE_BATCH_SIZE = 128

GPU_ID = "0"
CLIP_NAME = "clip_resnet50"
USE_FP16 = True

# Zero is the safest setting for native Windows/PyCharm. Increase to 2 or 4
# only after the first run succeeds.
NUM_WORKERS = 0

# The original shell scripts enable these wrappers. They are unnecessary on a
# single GPU and can be less stable on native Windows.
PARALLEL_TEXT_ENCODER = False
PARALLEL_VISION_ENCODER = False

N_GLOBAL_PROMPTS = 4
N_REGIONAL_PROMPTS = 4
N_LOCAL_PROMPTS = 4
# ============================================================================


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "mplnet" / "train.py"
SAVE_DIR = PROJECT_ROOT / "results"


def _bool(value: bool) -> str:
    return "True" if value else "False"


def _check_environment() -> None:
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Cannot find {TRAIN_SCRIPT}. Put run_pycharm.py in the MPLNet "
            "repository root."
        )

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"DATA_DIR does not exist: {DATA_DIR}\n"
            "Edit DATA_DIR near the top of run_pycharm.py."
        )

    expected_dataset_dir = DATA_DIR / DATASET_NAME
    if not expected_dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {expected_dataset_dir}\n"
            "The directory name must match DATASET_NAME."
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the selected PyCharm interpreter."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Select the mplnet Conda interpreter and "
            "install a CUDA-enabled PyTorch build."
        )

    print(f"Project: {PROJECT_ROOT}")
    print(f"Dataset: {expected_dataset_dir}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def _override_windows_num_workers() -> None:
    """Override the repository's hard-coded ``num_workers=10`` safely."""
    import mplnet.datasets.tools as dataset_tools

    original_get_train_loader = dataset_tools.get_train_loader

    def get_train_loader_windows(
        dataset,
        batch_size,
        num_workers=10,
        persistent_workers=False,
    ):
        del num_workers
        return original_get_train_loader(
            dataset,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            persistent_workers=persistent_workers and NUM_WORKERS > 0,
        )

    dataset_tools.get_train_loader = get_train_loader_windows


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    _check_environment()
    _override_windows_num_workers()

    experiment_name = f"{DATASET_NAME}_shot_{NUM_SHOTS}_seed_{SEED}"
    sys.argv = [
        str(TRAIN_SCRIPT),
        "--seed", str(SEED),
        "--clip_name", CLIP_NAME,
        "--exp_name", experiment_name,
        "--save_dir", str(SAVE_DIR),
        "--data_dir", str(DATA_DIR),
        "--dataset_name", DATASET_NAME,
        "--max_epoch", str(MAX_EPOCH),
        "--batch_size", str(BATCH_SIZE),
        "--inference_batch_size", str(INFERENCE_BATCH_SIZE),
        "--eval_ood", "False",
        "--eval_domains", "False",
        "--eval_freq", str(MAX_EPOCH),
        "--checkpointing_segments", "0",
        "--use_local_features", "True",
        "--lr_init", "0.002",
        "--warmup_epoch", "1",
        "--num_shots", str(NUM_SHOTS),
        "--use_fp16", _bool(USE_FP16),
        "--parallel_text_encoder", _bool(PARALLEL_TEXT_ENCODER),
        "--parallel_vision_encoder", _bool(PARALLEL_VISION_ENCODER),
        "--learn_global_prompts", "True",
        "--n_global_prompts", str(N_GLOBAL_PROMPTS),
        "--learn_regional_prompts", "True",
        "--n_regional_prompts", str(N_REGIONAL_PROMPTS),
        "--learn_local_prompts", "True",
        "--n_local_prompts", str(N_LOCAL_PROMPTS),
    ]

    print("Starting MPLNet with arguments:")
    print(" ".join(sys.argv))
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
