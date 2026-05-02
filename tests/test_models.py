import torch
import json
from pathlib import Path
from src.models.model_factory import get_model_from_meta
from src.models.rnn_model import SequenceRNNClassifier
from src.models.transformer_model import TransformerClassifier

# Giả lập file meta để test
def create_mock_meta(tmp_path):
    meta_data = {
        "label_map": {"APPLE": 0, "CHAMP": 1, "SAIL1": 2},
        "feature_dim": 549,
        "seq_len": 48
    }
    meta_path = tmp_path / "train_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f)
    return meta_path

def test_factory_and_rnn_shape(tmp_path):
    """Kiểm tra khởi tạo GRU qua factory và tính toán đầu ra."""
    meta_path = create_mock_meta(tmp_path)
    config = {
        "model": {
            "type": "gru",
            "params": {
                "hidden_dim": 64,
                "num_layers": 2,
                "bidirectional": True
            }
        }
    }
    
    # Khởi tạo model từ meta
    model = get_model_from_meta(config, str(meta_path))
    assert isinstance(model, SequenceRNNClassifier)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 48, 549) # [batch, seq_len, feature_dim]
    output = model(x)
    
    assert output.shape == (batch_size, 3) # 3 classes từ label_map
    print("RNN Factory & Shape test passed!")

def test_transformer_shape(tmp_path):
    """Kiểm tra khởi tạo Transformer và tính toán đầu ra."""
    meta_path = create_mock_meta(tmp_path)
    config = {
        "model": {
            "type": "transformer",
            "params": {
                "d_model": 128,
                "nhead": 4,
                "num_layers": 2
            }
        }
    }
    
    model = get_model_from_meta(config, str(meta_path))
    assert isinstance(model, TransformerClassifier)
    
    x = torch.randn(2, 48, 549)
    output = model(x)
    
    assert output.shape == (2, 3)
    print("Transformer Shape test passed!")

def test_save_load_logic(tmp_path):
    """Kiểm tra logic lưu và tải model kế thừa từ base."""
    meta_path = create_mock_meta(tmp_path)
    config = {
        "model": {"type": "gru", "params": {"hidden_dim": 32}}
    }
    model = get_model_from_meta(config, str(meta_path))
    
    # Lưu model
    save_path = tmp_path / "model.pth"
    label_map = {"APPLE": 0, "CHAMP": 1, "SAIL1": 2}
    model.save(save_path, config, label_map)
    
    # Tải lại payload (kiểm tra cấu trúc file lưu)
    payload = torch.load(save_path)
    assert "model_config" in payload
    assert "label_map" in payload
    assert payload["label_map"]["CHAMP"] == 1
    
    print("Save/Load logic test passed!")

if __name__ == "__main__":
    # Sử dụng thư mục tạm của hệ thống để chạy test nhanh
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            test_factory_and_rnn_shape(tmp_path)
            test_transformer_shape(tmp_path)
            test_save_load_logic(tmp_path)
            print("\n[SUCCESS] Tất cả các thành phần Model đã sẵn sàng!")
        except Exception as e:
            print(f"\n[FAILED] Lỗi test: {e}")