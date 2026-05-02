import torch
import json
from src.models.rnn_model import SequenceRNNClassifier
from src.models.transformer_model import TransformerClassifier

def get_model_from_meta(config_dict: dict, meta_path: str):
    """Khởi tạo model tự động dựa trên file train_meta.json."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    input_dim = meta["feature_dim"]
    num_classes = len(meta["label_map"])
    
    m_cfg = config_dict["model"]
    m_type = m_cfg["type"].lower()
    params = m_cfg.get("params", {})

    if m_type in ["gru", "lstm"]:
        return SequenceRNNClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            model_type=m_type,
            **params
        )
    elif m_type == "transformer":
        return TransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            **params
        )
    raise ValueError(f"Loại mô hình {m_type} không hỗ trợ.")