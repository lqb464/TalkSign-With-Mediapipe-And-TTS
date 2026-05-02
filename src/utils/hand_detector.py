import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import yaml

with open("configs/utils.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

HD_CFG = cfg["hand_detector"]

class HandDetector:
    def __init__(
        self,
        model_path: str | None = None,
        num_hands: int | None = None,
        min_hand_detection_confidence: float | None = None,
        min_hand_presence_confidence: float | None = None,
        min_tracking_confidence: float | None = None,
    ):

        model_path = model_path or HD_CFG["model_path"]
        num_hands = num_hands or HD_CFG["num_hands"]
        min_hand_detection_confidence = (
            min_hand_detection_confidence or HD_CFG["min_hand_detection_confidence"]
        )
        min_hand_presence_confidence = (
            min_hand_presence_confidence or HD_CFG["min_hand_presence_confidence"]
        )
        min_tracking_confidence = (
            min_tracking_confidence or HD_CFG["min_tracking_confidence"]
        )

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Missing model: {model_file}")

        base_options = python.BaseOptions(model_asset_path=str(model_file))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_result = None

    def detect(self, frame, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        self.last_result = result
        return result

    def get_hands_data(self, result, frame_shape):
        """
        Lấy đủ 3 trục x, y, z từ MediaPipe.
        Trả về list các dict chứa tọa độ chuẩn hóa (float 0.0 - 1.0).
        """
        hands_data = []
        if result is None or not result.hand_landmarks:
            return hands_data

        for i, hand_landmarks in enumerate(result.hand_landmarks):
            # CẬP NHẬT: Lấy cả 3 tọa độ x, y, z
            points = [[float(lm.x), float(lm.y), float(lm.z)] for lm in hand_landmarks]

            label = None
            score = None
            if result.handedness and i < len(result.handedness):
                label = result.handedness[i][0].category_name
                score = result.handedness[i][0].score

            hands_data.append(
                {
                    "landmarks": points,
                    "label": label,
                    "score": score,
                }
            )

        # Sắp xếp để ưu tiên thứ tự hiển thị
        label_order = {"Left": 0, "Right": 1, None: 2}
        hands_data.sort(key=lambda hand: label_order.get(hand["label"], 2))

        return hands_data

    def draw_hands(self, frame, result):
        """
        Vẽ landmarks lên khung hình 2D. 
        Tọa độ z sẽ tự động bị bỏ qua trong quá trình vẽ.
        """
        if result is None or not result.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        # Định nghĩa các nhóm đường nối ngón tay
        THUMB = [0, 1, 2, 3, 4]
        INDEX = [0, 5, 6, 7, 8]
        MIDDLE = [9, 10, 11, 12]
        RING = [13, 14, 15, 16]
        PINKY = [0, 17, 18, 19, 20]
        PALM = [5, 9, 13, 17] 

        # Lấy dữ liệu 3D nhưng chỉ dùng x, y để vẽ
        hands_data = self.get_hands_data(result, frame.shape)

        for hand_data in hands_data:
            points = hand_data["landmarks"]
            # Chuyển đổi tọa độ chuẩn hóa (0-1) về tọa độ pixel để vẽ
            pts = [(int(p[0] * w), int(p[1] * h)) for p in points]
            
            # Vẽ đường nối
            for finger in [THUMB, INDEX, MIDDLE, RING, PINKY]:
                for j in range(len(finger) - 1):
                    cv2.line(frame, pts[finger[j]], pts[finger[j+1]], (0, 255, 0), 2)

            for j in range(len(PALM) - 1):
                cv2.line(frame, pts[PALM[j]], pts[PALM[j+1]], (0, 255, 0), 2)

            # Vẽ các điểm landmarks
            for x, y in pts:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        return frame

    def close(self):
        self.landmarker.close()