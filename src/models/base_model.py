import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

class SignLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path: Path, model_config: Dict[str, Any], label_map: Dict[str, int], extra: Dict | None = None):
        """Lưu model kèm theo label_map để hậu kiểm dễ dàng."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_config": model_config,
            "label_map": label_map,
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)