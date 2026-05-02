import argparse
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple
import cv2
import yaml
import os
import contextlib

# Import các công cụ từ project
from src.utils.hand_detector import HandDetector
from src.utils.face_detector import FaceDetector
from src.utils.body_detector import BodyDetector

# Tắt log hệ thống
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- LOAD CẤU HÌNH ---
with open("configs/data.yaml", encoding="utf-8") as f:
    cfg_data = yaml.safe_load(f)

with open("configs/utils.yaml", encoding="utf-8") as f:
    cfg_utils = yaml.safe_load(f)

DATA_CFG = cfg_data["data"]
RAW_DIR = Path(DATA_CFG["raw_data_dir"])
RECORD_FPS = cfg_utils["webcam"]["fps"]

@contextlib.contextmanager
def suppress_native_stderr():
    """Chặn các log rác từ C++ của MediaPipe."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        os.close(devnull_fd)

# --- LOGIC XỬ LÝ VIDEO ---

# Sửa hàm này để nhận và trả về timestamp hiện tại
def process_video(video_path: Path, detectors: Dict, record_fps: float, start_timestamp_ms: int) -> Tuple[List[Dict], int]:
    """Trích xuất trọn bộ landmarks (Hand, Face, Body) từ video."""
    cap = cv2.VideoCapture(str(video_path))
    sequence_data = []
    frame_duration_ms = 1000.0 / record_fps
    current_ts = start_timestamp_ms # Bắt đầu từ mốc thời gian của video trước

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Cộng dồn timestamp để đảm bảo tính tăng tiến tuyệt đối
        current_ts += int(frame_duration_ms)
        
        # Chạy đồng thời 3 bộ detector với timestamp liên tục
        h_res = detectors['hand'].detect(frame, current_ts)
        f_res = detectors['face'].detect(frame, current_ts)
        b_res = detectors['body'].detect(frame, current_ts)

        # Đóng gói dữ liệu chuẩn ASL
        frame_data = {
            "hands": detectors['hand'].get_hands_data(h_res, frame.shape),
            "face": detectors['face'].get_faces_data(f_res, frame.shape),
            "body": detectors['body'].get_bodies_data(b_res, frame.shape)
        }
        
        sequence_data.append(frame_data)

    cap.release()
    return sequence_data, current_ts # Trả về timestamp cuối cùng để video sau dùng tiếp

# --- CÁC HÀM PHỤ TRỢ ---

def load_video_labels(splits_dir: Path) -> Dict[str, str]:
    """Lấy nhãn từ các file CSV metadata."""
    video_to_label = {}
    for csv_path in splits_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                video_to_label[row['Video file']] = str(row['Gloss']).upper()
        except Exception as e:
            print(f"  [!] Lỗi đọc {csv_path.name}: {e}")
    return video_to_label

def get_processed_videos(raw_dir: Path) -> Set[str]:
    """Kiểm tra xem video nào đã được xử lý trước đó."""
    processed = set()
    for jsonl in raw_dir.glob("session_*.jsonl"):
        with open(jsonl, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                    parts = sample["sample_id"].split("_")
                    if len(parts) >= 3:
                        processed.add("_".join(parts[2:]))
                except json.JSONDecodeError:
                    print(f"  [!] Bỏ qua dòng {line_num} bị lỗi trong {jsonl.name}")
                    continue
    return processed

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Import external videos and extract landmarks")
    parser.add_argument("--test", action="store_true", help="Chạy thử nghiệm với thư mục test")
    args = parser.parse_args()

    if args.test:
        print("[!] Đang chạy ở chế độ TEST...")
        base_path = Path("data/external/test")
        videos_dir = base_path
        splits_dir = base_path
    else:
        print("[*] Đang chạy ở chế độ THẬT...")
        base_path = Path(DATA_CFG["external_data_dir"])
        videos_dir = base_path / "videos"
        splits_dir = base_path / "splits"

    # 1. Khởi tạo trọn bộ Detectors (Chỉ khởi tạo 1 lần)
    print("[1/3] Khởi tạo AI Models...")
    with suppress_native_stderr():
        detectors = {
            'hand': HandDetector(),
            'face': FaceDetector(),
            'body': BodyDetector()
        }

    # 2. Chuẩn bị danh sách xử lý
    print(f"[2/3] Kiểm tra các video cần xử lý...")
    video_labels = load_video_labels(splits_dir)
    processed_stems = get_processed_videos(RAW_DIR)
    
    all_videos = sorted([v for v in videos_dir.glob("*") if v.suffix.lower() in [".mp4", ".mov", ".avi"]])
    to_process = [v for v in all_videos if v.stem not in processed_stems]

    print(f"    > Trạng thái: Tổng {len(all_videos)} | Đã xong {len(processed_stems)} | Còn lại {len(to_process)}")

    if not to_process: 
        print("[3/3] Hoàn thành import external videos.")
        return

    # 3. Vòng lặp xử lý chính
    print("[3/3] Bắt đầu trích xuất landmarks...")
    session_id = int(time.time() * 1000)
    if DATA_CFG.get("run_kaggle", False):
        SAVE_RAW_DIR = Path("data/raw")
        SAVE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    else:
        SAVE_RAW_DIR = RAW_DIR
        SAVE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SAVE_RAW_DIR / f"session_ext_{session_id}.jsonl"
    fps = float(RECORD_FPS)
    
    # Biến lưu trữ timestamp xuyên suốt tất cả video
    global_ts_accumulator = 0

    with open(output_path, "a", encoding="utf-8") as f_out:
        for i, vp in enumerate(to_process, 1):
            label = video_labels.get(vp.name) or (vp.stem.split("-")[-1].upper() if "-" in vp.stem else None)
            
            if not label:
                print(f"    [?] Bỏ qua {vp.name}: Không có nhãn")
                continue

            try:
                # Truyền global_ts_accumulator vào và nhận lại giá trị mới
                sequence, global_ts_accumulator = process_video(vp, detectors, fps, global_ts_accumulator)
                
                sample = {
                    "sample_id": f"ext_{int(time.time()*1000)}_{vp.stem}",
                    "label": label,
                    "data": sequence
                }
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                
                if i % 100 == 0:
                    print(f"    > Tiến độ: {i}/{len(to_process)} video...")
            except Exception as e:
                print(f"    [!] Lỗi video {vp.name}: {e}")

    # Giải phóng
    for d in detectors.values(): d.close()
    print(f"\n[HOÀN THÀNH] Dữ liệu lưu tại: {output_path}")

if __name__ == "__main__":
    main()