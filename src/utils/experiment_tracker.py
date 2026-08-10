"""Standardized Reproducible Experiment Tracking Module for SEM Image Restoration.

This module provides ``ExperimentTracker``, which automatically gathers runtime
configuration, system environment, compute context, model parameter counts, training
hyperparameters, evaluation metrics (PSNR, SSIM, LPIPS), and artifact paths into a
standardized, machine-readable YAML experiment record.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.utils.config import Config


def _get_git_commit() -> Optional[str]:
    """Retrieve current Git commit SHA safely.

    Returns:
        Optional[str]: 40-character Git commit hash string, or None if git is unavailable.
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return commit if commit else None
    except Exception:
        return None


def _detect_platform() -> str:
    """Detect current execution platform environment.

    Returns:
        str: 'Kaggle', 'Google Colab', or local OS system name ('Windows', 'Linux', 'Darwin').
    """
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "Kaggle"
    if "google.colab" in sys.modules:
        return "Google Colab"
    return platform.system()


def _normalize_dataset_path(path_str: Optional[Union[str, Path]]) -> str:
    """Normalize dataset path to a portable string without hardcoding private absolute paths."""
    if not path_str:
        return "./datasets"
    try:
        path_obj = Path(path_str)
        cwd = Path.cwd()
        if path_obj.is_absolute():
            try:
                rel = path_obj.relative_to(cwd)
                return f"./{rel.as_posix()}"
            except ValueError:
                return path_obj.as_posix()
        return path_obj.as_posix()
    except Exception:
        return str(path_str)


def _get_compute_environment(
    device_str: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    """Capture current compute, GPU, PyTorch, and Python runtime information.

    Args:
        device_str: Device string ('cuda' or 'cpu').

    Returns:
        Dict[str, Any]: Dictionary containing device, GPU model, CUDA, and library versions.
    """
    dev_str = str(device_str)
    is_cuda_avail = torch.cuda.is_available()
    is_cuda_device = is_cuda_avail and ("cuda" in dev_str.lower())

    gpu_name = torch.cuda.get_device_name(0) if is_cuda_device else None
    cuda_ver = torch.version.cuda if is_cuda_avail else None

    return {
        "platform": _detect_platform(),
        "device": dev_str,
        "gpu": gpu_name,
        "cuda_version": cuda_ver,
        "pytorch_version": str(torch.__version__),
        "python_version": sys.version.split()[0],
    }


def _get_config_value(config: Config, key_paths: list, default: Any = None) -> Any:
    """Extract nested configuration value from Config instance."""
    cfg_dict = config.to_dict()
    for key_path in key_paths:
        parts = key_path.split(".")
        curr: Any = cfg_dict
        found = True
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                found = False
                break
        if found and curr is not None:
            return curr
    return default


class ExperimentTracker:
    """Standardized Machine-Readable Experiment Tracker.

    Tracks metadata, configuration, compute environment, metric histories, and
    artifact paths across training runs, enabling automatic incremental record
    persistence after every validation epoch.

    Args:
        config: Master Config instance.
        model: Optional PyTorch model nn.Module instance.
        optimizer: Optional PyTorch optimizer instance.
        scheduler: Optional PyTorch LR scheduler instance.
        criterion: Optional loss module instance.
        record_dir: Optional output directory where experiment record YAML will be saved.
            Defaults to 'outputs/experiments'.
    """

    def __init__(
        self,
        config: Config,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        record_dir: Optional[Union[str, Path]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config
        self.exp_id = str(
            _get_config_value(
                config,
                ["experiment_id", "experiment.id", "exp_id"],
                default="default_experiment",
            )
        )

        self.record_dir = (
            Path(record_dir).resolve()
            if record_dir is not None
            else Path(
                _get_config_value(
                    config,
                    ["system.experiment_dir", "experiment_dir"],
                    default="outputs/experiments",
                )
            ).resolve()
        )
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.record_file_path = self.record_dir / f"{self.exp_id}_record.yaml"

        # 1. Experiment Metadata
        self.git_commit = _get_git_commit()
        env_dataset_dir = os.environ.get("SEM_DATASET_ROOT") or os.environ.get(
            "SEM_DATASET_DIR"
        )
        raw_dataset_dir = (
            dataset_dir
            or env_dataset_dir
            or _get_config_value(
                config,
                ["data.dataset_dir", "dataset_dir", "data.dataset_path"],
                default="./datasets",
            )
        )
        resolved_dataset_dir = _normalize_dataset_path(raw_dataset_dir)

        dataset_splits = _get_config_value(
            config, ["data.splits", "splits"], default=["train", "val"]
        )

        # 2. Model Metadata
        arch_name = (
            type(model).__name__
            if model is not None
            else str(_get_config_value(config, ["model.name", "model.architecture"], default="NAFNet"))
        )
        total_params = (
            sum(p.numel() for p in model.parameters()) if model is not None else 0
        )
        trainable_params = (
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            if model is not None
            else 0
        )
        model_cfg = _get_config_value(config, ["model"], default={})

        # 3. Training Metadata
        opt_name = (
            type(optimizer).__name__
            if optimizer is not None
            else "AdamW"
        )
        lr_val = float(
            _get_config_value(
                config,
                ["train.learning_rate", "train.lr", "learning_rate"],
                default=1e-3,
            )
        )
        sched_name = (
            type(scheduler).__name__
            if scheduler is not None
            else str(_get_config_value(config, ["train.scheduler"], default="CosineAnnealingLR"))
        )
        loss_name = (
            type(criterion).__name__
            if criterion is not None
            else str(_get_config_value(config, ["loss.name"], default="CharbonnierLoss"))
        )
        batch_size = int(
            _get_config_value(
                config, ["train.batch_size", "batch_size", "data.train_batch_size"], default=4
            )
        )
        epochs = int(
            _get_config_value(config, ["train.epochs", "epochs"], default=100)
        )
        seed = int(
            _get_config_value(config, ["system.seed", "seed", "train.seed"], default=42)
        )

        # 4. Compute Context
        device_setting = str(
            _get_config_value(config, ["system.device", "train.device", "device"], default="auto")
        )
        device_str = (
            "cuda"
            if device_setting == "cuda" or (device_setting == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        compute_info = _get_compute_environment(device_str=device_str)

        # 5. Metrics Initialization
        self.metrics_record = {
            "psnr": {"best": None, "epoch": None},
            "ssim": {"best": None, "epoch": None},
            "lpips": {"best": None, "epoch": None},
        }

        # 6. Artifact Paths Initialization
        checkpoint_dir = str(
            _get_config_value(
                config, ["system.checkpoint_dir", "checkpoint_dir"], default="outputs/checkpoints"
            )
        )
        tensorboard_dir = str(
            _get_config_value(
                config, ["system.tensorboard_dir", "tensorboard_dir"], default="outputs/tensorboard"
            )
        )
        log_dir = str(
            _get_config_value(config, ["system.log_dir", "log_dir"], default="logs")
        )

        self.artifact_paths = {
            "best_checkpoint": f"{checkpoint_dir}/best_model.pth",
            "latest_checkpoint": f"{checkpoint_dir}/checkpoint_latest.pth",
            "tensorboard": tensorboard_dir,
            "logs": log_dir,
            "predictions": "outputs/predictions",
            "report": f"experiments/{self.exp_id}_report.md",
        }

        self.record_dict: Dict[str, Any] = {
            "experiment": {
                "id": self.exp_id,
                "git_commit": self.git_commit,
                "dataset": {
                    "path": resolved_dataset_dir,
                    "splits": dataset_splits,
                },
            },
            "model": {
                "architecture": arch_name,
                "parameters": {
                    "total": total_params,
                    "trainable": trainable_params,
                },
                "config": model_cfg if isinstance(model_cfg, dict) else {},
            },
            "training": {
                "optimizer": opt_name,
                "learning_rate": lr_val,
                "scheduler": sched_name,
                "loss": loss_name,
                "batch_size": batch_size,
                "epochs": epochs,
                "seed": seed,
            },
            "compute": compute_info,
            "metrics": self.metrics_record,
            "artifacts": self.artifact_paths,
        }

        # Save initial record template
        self.save()

    def update_validation(
        self,
        epoch: int,
        val_metrics: Dict[str, Optional[float]],
    ) -> None:
        """Update metrics record after a validation epoch and save record incrementally.

        Args:
            epoch: Current validation epoch integer.
            val_metrics: Dict with metric values (e.g. 'val_psnr', 'val_ssim', 'val_lpips').
        """
        # PSNR (higher is better)
        psnr_val = val_metrics.get("val_psnr")
        if psnr_val is not None:
            curr_best = self.metrics_record["psnr"]["best"]
            if curr_best is None or psnr_val > curr_best:
                self.metrics_record["psnr"]["best"] = round(psnr_val, 4)
                self.metrics_record["psnr"]["epoch"] = epoch

        # SSIM (higher is better)
        ssim_val = val_metrics.get("val_ssim")
        if ssim_val is not None:
            curr_best = self.metrics_record["ssim"]["best"]
            if curr_best is None or ssim_val > curr_best:
                self.metrics_record["ssim"]["best"] = round(ssim_val, 4)
                self.metrics_record["ssim"]["epoch"] = epoch

        # LPIPS (lower is better)
        lpips_val = val_metrics.get("val_lpips")
        if lpips_val is not None:
            curr_best = self.metrics_record["lpips"]["best"]
            if curr_best is None or lpips_val < curr_best:
                self.metrics_record["lpips"]["best"] = round(lpips_val, 4)
                self.metrics_record["lpips"]["epoch"] = epoch

        self.record_dict["metrics"] = self.metrics_record

        # Incrementally persist record after every validation update
        self.save()

    def set_artifact_paths(self, artifacts_dict: Dict[str, str]) -> None:
        """Update artifact paths and persist record.

        Args:
            artifacts_dict: Dictionary mapping artifact names to path strings.
        """
        self.artifact_paths.update(artifacts_dict)
        self.record_dict["artifacts"] = self.artifact_paths
        self.save()

    def to_dict(self) -> Dict[str, Any]:
        """Return full experiment record as a native Python dictionary."""
        return self.record_dict

    def save(self, record_path: Optional[Union[str, Path]] = None) -> Path:
        """Save standardized YAML experiment record to disk atomically.

        Args:
            record_path: Optional custom destination file path.

        Returns:
            Path: Path to saved YAML record file.
        """
        target_path = (
            Path(record_path).resolve()
            if record_path is not None
            else self.record_file_path
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        return target_path
