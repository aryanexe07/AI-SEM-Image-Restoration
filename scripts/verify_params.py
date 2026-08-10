import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from src.utils.config import load_config
from src.models.builder import build_model

def print_model_params(config_path):
    print(f"\n--- Verifying {config_path} ---")
    cfg = load_config("configs/train.yaml", config_path)
    model = build_model(cfg)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model config: width={cfg.model.width}, upscale={cfg.model.upscale}")

if __name__ == "__main__":
    print_model_params("configs/experiments/exp001.yaml")
    print_model_params("configs/experiments/exp002_nafnet_width48.yaml")
    print_model_params("configs/experiments/exp003_nafnet_width64.yaml")
