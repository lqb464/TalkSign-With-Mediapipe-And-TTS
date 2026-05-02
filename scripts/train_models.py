import argparse
import json
import random
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Import các thành phần chúng ta đã cấu trúc lại
from src.models.model_factory import get_model_from_meta

try:
    import wandb
except ImportError:
    wandb = None

class ASLSequenceDataset(Dataset):
    """Dataset xử lý chuỗi tọa độ landmarks."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = int(self.y[idx])

        # Tính toán độ dài thực tế (không tính padding)
        nonzero_frames = np.any(x != 0.0, axis=1)
        length = int(nonzero_frames.sum())
        length = max(length, 1)

        return (
            torch.from_numpy(x).float(),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )

def load_dataset(path: Path):
    """Tải dữ liệu từ file .npz."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset tại: {path}")
    
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]

def evaluate(model, data_loader, criterion, device):
    """Hàm đánh giá mô hình độc lập."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y, _ in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / max(len(data_loader), 1)
    accuracy = correct / max(total, 1)
    return accuracy, avg_loss

def train():
    # 1. Tải cấu hình
    with open("configs/train.yaml", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)["train"]
    with open("configs/model.yaml", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # Thiết lập seed
    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 2. Chuẩn bị dữ liệu
    data_path = Path(train_cfg["dataset"])
    meta_path = Path(train_cfg.get("meta_file", "data/processed/train_meta.json"))
    
    X, y = load_dataset(data_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    
    dataset = ASLSequenceDataset(X, y)
    val_split = train_cfg.get("val_split", 0.2)
    val_size = int(len(dataset) * val_split)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    # 3. Khởi tạo Model & Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_from_meta(model_cfg, str(meta_path)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=train_cfg["lr"], 
        weight_decay=train_cfg.get("weight_decay", 0.0001)
    )

    # 4. WandB Logging
    if wandb and train_cfg.get("use_wandb", False):
        wandb.init(project="talksign-asl", config={**train_cfg, **model_cfg})
        wandb.watch(model)

    # 5. Vòng lặp huấn luyện
    best_acc = 0.0
    out_dir = Path(train_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Bắt đầu huấn luyện trên {device}...")
    for epoch in range(train_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y, _ in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Đánh giá
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        avg_train_loss = train_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{train_cfg['epochs']}] - Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.4f}")

        if wandb and train_cfg.get("use_wandb", False):
            wandb.log({"train_loss": avg_train_loss, "val_acc": val_acc, "val_loss": val_loss})

        # Lưu model tốt nhất
        if val_acc > best_acc:
            best_acc = val_acc
            model.save(
                out_dir / "best_model.pth",
                model_config=model_cfg,
                label_map=meta_data["label_map"],
                extra={"val_acc": val_acc, "epoch": epoch}
            )

    print(f"Huấn luyện hoàn tất! Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train()