import math
import time
import cv2
import yaml

from src.data.label_data import ask_label, close_labeler, init_labeler, save_session_to_jsonl
from src.utils.hand_detector import HandDetector
from src.utils.face_detector import FaceDetector
from src.utils.body_detector import BodyDetector
from src.utils.webcam import Webcam

# Tải cấu hình
with open("configs/data.yaml", encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)
REC_CFG = data_cfg["record"]

with open("configs/utils.yaml", encoding="utf-8") as f:
    utils_cfg = yaml.safe_load(f)
CAM_CFG = utils_cfg["webcam"]

WINDOW_NAME = "ASL Data Collection"
RECORD_FPS = int(CAM_CFG["fps"])
RECORD_SECONDS = float(REC_CFG["record_seconds"])
COUNTDOWN_SECONDS = float(REC_CFG["countdown_seconds"])
TOTAL_TARGET_FRAMES = int(RECORD_SECONDS * RECORD_FPS)

def main():
    # Khởi tạo thiết bị và các bộ lọc
    cam = Webcam(camera_index=int(CAM_CFG["camera_index"]), 
                 width=int(CAM_CFG["width"]), 
                 height=int(CAM_CFG["height"]), 
                 fps=int(CAM_CFG["fps"]))
    
    hand_detector = HandDetector()
    face_detector = FaceDetector()
    body_detector = BodyDetector()
    
    init_labeler()

    all_collected_samples = []
    start_session_time = time.time()
    sequence = []
    countdown_active = True
    recording = False
    countdown_start = time.time()
    last_frame_time = 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None: break

            now = time.time()
            timestamp_ms = int((now - start_session_time) * 1000)

            # 1. Chạy Detection (Toàn bộ các thành phần ASL)
            hand_res = hand_detector.detect(frame, timestamp_ms)
            face_res = face_detector.detect(frame, timestamp_ms)
            body_res = body_detector.detect(frame, timestamp_ms)

            # 2. Vẽ lên màn hình để người dùng căn chỉnh
            frame = body_detector.draw_bodies(frame, body_res)
            frame = face_detector.draw_faces(frame, face_res)
            frame = hand_detector.draw_hands(frame, hand_res)

            # 3. Lấy dữ liệu Landmarks
            hands = hand_detector.get_hands_data(hand_res, frame.shape)
            faces = face_detector.get_faces_data(face_res, frame.shape)
            bodies = body_detector.get_bodies_data(body_res, frame.shape)

            status = "IDLE"

            # logic Đếm ngược
            if countdown_active and not recording:
                elapsed = now - countdown_start
                remaining = max(0.0, COUNTDOWN_SECONDS - elapsed)
                status = f"WAITING: {math.ceil(remaining)}s"
                if elapsed >= COUNTDOWN_SECONDS:
                    countdown_active, recording = False, True
                    sequence = []
                    last_frame_time = 0.0

            # Logic Ghi hình
            elif recording:
                status = f"RECORDING... | {len(sequence)}/{TOTAL_TARGET_FRAMES}"
                
                # Capture theo nhịp RECORD_FPS
                if now - last_frame_time >= (1.0 / RECORD_FPS):
                    # Lưu trữ đồng thời cả 3 nguồn dữ liệu
                    sequence.append({
                        "hands": hands,
                        "face": faces,
                        "body": bodies
                    })
                    last_frame_time = now

                if len(sequence) >= TOTAL_TARGET_FRAMES:
                    recording = False
                    label_input = ask_label(frame.copy(), len(sequence), WINDOW_NAME)
                    
                    if label_input == 27: # ESC
                        break 
                    elif str(label_input).lower() != "skip":
                        all_collected_samples.append({
                            "sample_id": f"asl_{int(time.time()*1000)}",
                            "label": label_input,
                            "data": sequence
                        })
                    
                    countdown_active, countdown_start = True, time.time()

            # Hiển thị UI điều khiển
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {len(all_collected_samples)}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    finally:
        if all_collected_samples:
            path = save_session_to_jsonl(all_collected_samples)
            print(f"\n[SUCCESS] Đã lưu {len(all_collected_samples)} mẫu vào: {path}")
        
        hand_detector.close()
        face_detector.close()
        body_detector.close()
        cam.release()
        cv2.destroyAllWindows()
        close_labeler()