import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from src.utils.config import load_config
from src.models.builder import build_model
from src.engine.trainer import build_loss

def smoke_test(config_path):
    print(f"\n=== Smoke Test: {config_path} ===")
    cfg = load_config("configs/train.yaml", config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = build_model(cfg).to(device)
    model.train()
    
    # 2. Setup inputs (B, C, H, W) -> (4, 1, 128, 128)
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 128).to(device)
    target = torch.randn(batch_size, 1, 256, 256).to(device)
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {target.shape}")
    
    # 3. Forward Pass
    try:
        y = model(x)
        print(f"Output shape: {y.shape}")
        assert y.shape == target.shape, f"Shape mismatch: {y.shape} != {target.shape}"
        print("Forward pass: SUCCESS")
    except Exception as e:
        print(f"Forward pass: FAILED - {e}")
        return
        
    # 4. Loss Calculation
    try:
        criterion = build_loss(cfg.loss).to(device)
        loss = criterion(y, target)
        print(f"Loss value: {loss.item()}")
        print("Loss calculation: SUCCESS")
    except Exception as e:
        print(f"Loss calculation: FAILED - {e}")
        return
        
    # 5. Backward Pass
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Backward pass: SUCCESS")
    except Exception as e:
        print(f"Backward pass: FAILED - {e}")
        return
        
    print(f"--- Smoke Test {config_path} COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    smoke_test("configs/experiments/exp002_nafnet_width48.yaml")
    smoke_test("configs/experiments/exp003_nafnet_width64.yaml")
