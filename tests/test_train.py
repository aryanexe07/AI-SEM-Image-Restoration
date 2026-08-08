"""Unit tests for train.py CLI entry point and training orchestration pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.engine.checkpoint import CheckpointManager
from src.utils.config import Config
from train import _get_config_val, main, parse_args, resolve_device


class DummyDataset(Dataset):
    """Simple mock dataset for training CLI unit tests."""

    def __init__(self, length: int = 4) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return {
            "input": torch.zeros((1, 32, 32), dtype=torch.float32),
            "target": torch.ones((1, 64, 64), dtype=torch.float32),
            "filename": f"sample_{index}",
        }


def test_parse_args_defaults() -> None:
    """Test parse_args returns default argument values."""
    args = parse_args([])
    assert args.config == "configs/train.yaml"
    assert args.experiment is None
    assert args.resume is None


def test_parse_args_custom() -> None:
    """Test parse_args correctly parses custom --config, --experiment, and --resume."""
    cli_input = [
        "--config",
        "custom_config.yaml",
        "--experiment",
        "custom_exp.yaml",
        "--resume",
        "checkpoint.pth",
    ]
    args = parse_args(cli_input)
    assert args.config == "custom_config.yaml"
    assert args.experiment == "custom_exp.yaml"
    assert args.resume == "checkpoint.pth"


def test_resolve_device() -> None:
    """Test resolve_device maps 'auto' to 'cuda' or 'cpu' and passes specific strings through."""
    resolved_auto = resolve_device("auto")
    assert resolved_auto in ("cuda", "cpu")
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_get_config_val() -> None:
    """Test _get_config_val resolves nested dot-delimited key paths and fallbacks."""
    cfg = Config({"train": {"lr": 0.005}, "system": {"seed": 123}})
    assert (
        _get_config_val(cfg, ["train.learning_rate", "train.lr"], default=1e-3) == 0.005
    )
    assert _get_config_val(cfg, ["system.seed"], default=42) == 123
    assert _get_config_val(cfg, ["nonexistent.key"], default="fallback") == "fallback"


def test_invalid_config_path_raises() -> None:
    """Test main() raises FileNotFoundError for non-existent config path."""
    args = parse_args(["--config", "non_existent_config_file_12345.yaml"])
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        main(args)


def test_invalid_experiment_path_raises(tmp_path: Path) -> None:
    """Test main() raises FileNotFoundError for non-existent experiment path."""
    valid_cfg = tmp_path / "valid.yaml"
    valid_cfg.write_text("train: {}\n")
    args = parse_args(
        [
            "--config",
            str(valid_cfg),
            "--experiment",
            "non_existent_exp_12345.yaml",
        ]
    )
    with pytest.raises(
        FileNotFoundError, match="Experiment configuration override file not found"
    ):
        main(args)


def test_invalid_resume_path_raises(tmp_path: Path) -> None:
    """Test main() raises FileNotFoundError for non-existent resume path."""
    valid_cfg = tmp_path / "valid.yaml"
    valid_cfg.write_text("train: {}\n")
    args = parse_args(
        [
            "--config",
            str(valid_cfg),
            "--resume",
            "non_existent_resume_12345.pth",
        ]
    )
    with pytest.raises(FileNotFoundError, match="Resume checkpoint file not found"):
        main(args)


@patch("train.build_dataloaders")
@patch("train.build_model")
@patch("train.build_loss")
@patch("train.Trainer")
def test_main_pipeline_fresh_start(
    mock_trainer_cls: MagicMock,
    mock_build_loss: MagicMock,
    mock_build_model: MagicMock,
    mock_build_dataloaders: MagicMock,
    tmp_path: Path,
) -> None:
    """Test main() orchestration for fresh training execution."""
    cfg_file = tmp_path / "train.yaml"
    cfg_file.write_text("train: {epochs: 2}\n")

    loader = DataLoader(DummyDataset(), batch_size=2)
    mock_build_dataloaders.return_value = {"train": loader}

    mock_model = nn.Linear(1, 1)
    mock_build_model.return_value = mock_model

    mock_loss = nn.MSELoss()
    mock_build_loss.return_value = mock_loss

    mock_trainer_instance = MagicMock()
    mock_trainer_instance.fit.return_value = {
        "epochs_completed": 2,
        "final_train_loss": 0.1,
        "best_val_psnr": 30.0,
        "history": [],
    }
    mock_trainer_cls.return_value = mock_trainer_instance

    args = parse_args(["--config", str(cfg_file)])
    summary = main(args)

    assert mock_build_dataloaders.called
    assert mock_build_model.called
    assert mock_build_loss.called
    assert mock_trainer_cls.called

    trainer_kwargs = mock_trainer_cls.call_args[1]
    assert trainer_kwargs["model"] == mock_model
    assert trainer_kwargs["train_loader"] == loader
    assert trainer_kwargs["criterion"] == mock_loss
    assert isinstance(trainer_kwargs["optimizer"], optim.AdamW)
    assert trainer_kwargs["epochs"] == 2
    assert trainer_kwargs["device"] in ("cuda", "cpu")

    mock_trainer_instance.fit.assert_called_once_with(start_epoch=1)
    assert summary["epochs_completed"] == 2


@patch("train.build_dataloaders")
@patch("train.build_model")
@patch("train.build_loss")
@patch("train.Trainer")
def test_main_pipeline_resume(
    mock_trainer_cls: MagicMock,
    mock_build_loss: MagicMock,
    mock_build_model: MagicMock,
    mock_build_dataloaders: MagicMock,
    tmp_path: Path,
) -> None:
    """Test main() orchestration when resuming from a checkpoint."""
    cfg_file = tmp_path / "train.yaml"
    cfg_file.write_text("train: {epochs: 5}\n")

    # Create dummy checkpoint
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_mgr = CheckpointManager(ckpt_dir)
    mock_m = nn.Linear(1, 1)
    mock_opt = optim.AdamW(mock_m.parameters(), lr=1e-3)
    saved_ckpt = ckpt_mgr.save(epoch=3, model=mock_m, optimizer=mock_opt, metric=25.0)

    loader = DataLoader(DummyDataset(), batch_size=2)
    mock_build_dataloaders.return_value = {"train": loader}
    mock_build_model.return_value = mock_m
    mock_build_loss.return_value = nn.MSELoss()

    mock_trainer_instance = MagicMock()
    mock_trainer_instance.fit.return_value = {"epochs_completed": 2}
    mock_trainer_cls.return_value = mock_trainer_instance

    args = parse_args(["--config", str(cfg_file), "--resume", str(saved_ckpt)])
    main(args)

    # Resume must start at epoch = checkpoint["epoch"] + 1 -> 3 + 1 = 4
    mock_trainer_instance.fit.assert_called_once_with(start_epoch=4)
