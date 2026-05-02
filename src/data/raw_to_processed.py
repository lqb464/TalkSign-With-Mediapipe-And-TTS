import argparse
import json
import numpy as np
from pathlib import Path
import yaml

# --- TẢI CẤU HÌNH ---
with open("configs/data.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
PROC_CFG = cfg["processing"]

# 1. ĐỊNH NGHĨA CÁC ĐIỂM THIẾT YẾU (Giống hệt lúc vẽ)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# Tập hợp index mặt duy nhất và sắp xếp
FACE_INDICES = sorted(list(set(FACE_OVAL + LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + LIPS)))

# Kích thước đặc trưng mới
HAND_DIM = 21 * 3
POSE_DIM = 33 * 3 
FACE_DIM = len(FACE_INDICES) * 3
TOTAL_DIM = (HAND_DIM * 2) + POSE_DIM + FACE_DIM

INPUT_DIR = Path(DATA_CFG["raw_data_dir"])
OUTPUT_NAME = Path(DATA_CFG["processed_data_dir"]) / PROC_CFG["output_name"]

def extract_features(frame_data):
    """Trích xuất dữ liệu dựa trên quy tắc: Chỉ lấy những điểm được vẽ."""
    
    # --- 1. Xử lý Hands (21 điểm mỗi tay) ---
    left_hand = [0.0] * HAND_DIM
    right_hand = [0.0] * HAND_DIM
    
    for hand in frame_data.get("hands", []):
        label = str(hand.get("label", "")).lower()
        flat = []
        for p in hand["landmarks"]:
            # Điểm tay thường được vẽ đầy đủ nếu detect được
            flat.extend([float(p[0]), float(p[1]), 0.0])
        
        if "left" in label: left_hand = flat
        else: right_hand = flat

    # --- 2. Xử lý Body (Pose - 33 điểm) ---
    pose = [0.0] * POSE_DIM
    body_list = frame_data.get("body", [])
    if body_list:
        temp_pose = []
        # landmarks ở đây đã được Detector lọc [0,0] nếu visibility < 0.5
        for p in body_list[0]["landmarks"]:
            temp_pose.extend([float(p[0]), float(p[1]), 0.0])
        pose = temp_pose

    # --- 3. Xử lý Face (Chỉ lấy Index thiết yếu) ---
    face = [0.0] * FACE_DIM
    face_list = frame_data.get("face", [])
    if face_list:
        all_pts = face_list[0]["landmarks"]
        temp_face = []
        for idx in FACE_INDICES:
            if idx < len(all_pts):
                p = all_pts[idx]
                temp_face.extend([float(p[0]), float(p[1]), 0.0])
            else:
                temp_face.extend([0.0, 0.0, 0.0])
        face = temp_face

    return left_hand + right_hand + pose + face

def process_sample(raw_sample):
    """Chuyển đổi các frame trong sample thành mảng numpy."""
    sequence = [extract_features(frame) for frame in raw_sample["data"]]
    return np.array(sequence, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_NAME)
    parser.add_argument("--seq_len", type=int, default=int(PROC_CFG["sequence_length"]))
    args = parser.parse_args()

    input_files = sorted(args.input.glob("*.jsonl"))
    if not input_files: 
        print("[!] Không tìm thấy file dữ liệu thô.")
        return

    all_X, all_y, sample_ids = [], [], []
    print(f"[*] Đang xử lý {len(input_files)} file... Vector DIM: {TOTAL_DIM}")

    for line in f:
        sample = json.loads(line)
        feat_seq = np.array(process_sample(sample)) # Chuyển về numpy
        
        num_frames = len(feat_seq)
        
        if num_frames > args.seq_len:
            # Lấy mẫu đều trên toàn bộ chiều dài video thay vì truncate
            indices = np.linspace(0, num_frames - 1, args.seq_len).astype(int)
            final_feat = feat_seq[indices]
        else:
            # Nếu ngắn hơn thì vẫn dùng padding như cũ
            final_feat = np.zeros((args.seq_len, TOTAL_DIM), dtype=np.float32)
            final_feat[:num_frames] = feat_seq
        
        all_X.append(final_feat)
        all_y.append(sample["label"])

    # Encode Labels
    unique_labels = sorted(list(set(all_y)))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y_encoded = np.array([label_map[l] for l in all_y], dtype=np.int64)

    # Lưu dữ liệu đã xử lý
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=np.array(all_X),
        y=y_encoded,
        sample_ids=np.array(sample_ids),
        label_map=json.dumps(label_map)
    )

    # Lưu file meta để dùng cho Inference sau này
    with open(args.output.with_name("train_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "label_map": label_map, 
            "feature_dim": TOTAL_DIM, 
            "seq_len": args.seq_len,
            "face_indices": FACE_INDICES 
        }, f, ensure_ascii=False, indent=4)

    print(f"[✓] Hoàn tất! File lưu tại: {args.output}")