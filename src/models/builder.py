"""Model builder factory for NAFNet architecture construction.

This module provides the ``build_model`` factory function that reads a
configuration object and instantiates a validated NAFNet model instance.

Evidence:
    Builder pattern verified against:
    - docs/software_architecture.md Sec 8: ``build_model`` factory function
    - NAFNet_Architecture_Reverse_Engineering.md Sec 12: Configuration system
    - NAFNet_Implementation_Specification.md Sec 3.3: Constructor parameters
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.nafnet import NAFNet


def build_model(cfg: Any) -> nn.Module:
    """Construct a NAFNet model instance from a configuration object.

    Reads model hyperparameters from the configuration, validates all
    parameter types and constraints, and returns an instantiated NAFNet.

    The configuration object may be a dictionary or an attribute-access
    object (e.g., ``Config`` from ``src.utils.config``). The function
    supports both access styles.

    Expected configuration keys under ``model`` namespace:
        - ``img_channel`` (int, optional): Input/output image channels. Default 1.
        - ``width`` (int, optional): Base feature channel count. Default 32.
        - ``middle_blk_num`` (int, optional): Middle bottleneck blocks. Default 1.
        - ``enc_blk_nums`` (list[int], optional): Encoder block counts. Default [1, 1, 1].
        - ``dec_blk_nums`` (list[int], optional): Decoder block counts. Default [1, 1, 1].
        - ``upscale`` (int, optional): SR upscaling factor. Default 2.
        - ``drop_out_rate`` (float, optional): Dropout probability. Default 0.0.

    Args:
        cfg: Configuration object or dictionary. If it has a ``model``
            attribute/key, parameters are read from that sub-namespace.
            Otherwise, parameters are read directly from ``cfg``.

    Returns:
        Instantiated ``NAFNet`` model (``nn.Module``).

    Raises:
        ValueError: If any parameter fails type or constraint validation.
        TypeError: If configuration object is not a supported type.

    Example:
        >>> cfg = {"model": {"width": 32, "enc_blk_nums": [2, 2, 4],
        ...                  "middle_blk_num": 12, "dec_blk_nums": [2, 2, 2]}}
        >>> model = build_model(cfg)
        >>> type(model).__name__
        'NAFNet'
    """
    # --- Extract model configuration sub-namespace ---
    model_cfg = _extract_model_config(cfg)

    # --- Read parameters with defaults ---
    img_channel = _get_param(model_cfg, "img_channel", default=1)
    width = _get_param(model_cfg, "width", default=32)
    middle_blk_num = _get_param(model_cfg, "middle_blk_num", default=1)
    enc_blk_nums = _get_param(model_cfg, "enc_blk_nums", default=[1, 1, 1])
    dec_blk_nums = _get_param(model_cfg, "dec_blk_nums", default=[1, 1, 1])
    upscale = _get_param(model_cfg, "upscale", default=2)
    drop_out_rate = _get_param(model_cfg, "drop_out_rate", default=0.0)

    # --- Type Validation ---
    _validate_positive_int("img_channel", img_channel)
    _validate_positive_int("width", width)
    _validate_positive_int("middle_blk_num", middle_blk_num)
    _validate_positive_int("upscale", upscale)

    if not isinstance(enc_blk_nums, (list, tuple)):
        raise ValueError(
            f"enc_blk_nums must be a list or tuple, got {type(enc_blk_nums).__name__}"
        )
    if not isinstance(dec_blk_nums, (list, tuple)):
        raise ValueError(
            f"dec_blk_nums must be a list or tuple, got {type(dec_blk_nums).__name__}"
        )

    if not isinstance(drop_out_rate, (int, float)):
        raise ValueError(
            f"drop_out_rate must be a number, got {type(drop_out_rate).__name__}"
        )

    # --- Construct Model ---
    model = NAFNet(
        img_channel=img_channel,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=list(enc_blk_nums),
        dec_blk_nums=list(dec_blk_nums),
        upscale=upscale,
        drop_out_rate=float(drop_out_rate),
    )

    return model


def _extract_model_config(cfg: Any) -> Any:
    """Extract model sub-configuration from a configuration object.

    Args:
        cfg: Full configuration object or dictionary.

    Returns:
        Model sub-configuration namespace.

    Raises:
        TypeError: If cfg type is not supported.
    """
    # Dictionary access
    if isinstance(cfg, dict):
        return cfg.get("model", cfg)

    # Attribute access (e.g., Config object)
    if hasattr(cfg, "model"):
        return cfg.model

    # Direct access (cfg itself contains model params)
    return cfg


def _get_param(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a parameter from a configuration object with fallback default.

    Args:
        cfg: Configuration object or dictionary.
        key: Parameter name to read.
        default: Default value if parameter is not found.

    Returns:
        Parameter value or default.
    """
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _validate_positive_int(name: str, value: Any) -> None:
    """Validate that a value is a positive integer.

    Args:
        name: Parameter name for error messages.
        value: Value to validate.

    Raises:
        ValueError: If value is not a positive integer.
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
