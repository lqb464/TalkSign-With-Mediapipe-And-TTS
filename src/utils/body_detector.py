import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import yaml

with open("configs/utils.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

BD_CFG = cfg["body_detector"]

class BodyDetector:
    def __init__(
        self,
        model_path: str | None = None,
        num_poses: int | None = None,
        min_pose_detection_confidence: float | None = None,
        min_pose_presence_confidence: float | None = None,
        min_tracking_confidence: float | None = None,
        output_segmentation_masks: bool | None = None,
    ):

        model_path = model_path or BD_CFG["model_path"]
        num_poses = num_poses or BD_CFG["num_poses"]
        min_pose_detection_confidence = (
            min_pose_detection_confidence or BD_CFG["min_pose_detection_confidence"]
        )
        min_pose_presence_confidence = (
            min_pose_presence_confidence or BD_CFG["min_pose_presence_confidence"]
        )
        min_tracking_confidence = (
            min_tracking_confidence or BD_CFG["min_tracking_confidence"]
        )
        output_segmentation_masks = (
            output_segmentation_masks
            if output_segmentation_masks is not None
            else BD_CFG.get("output_segmentation_masks", False)
        )

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Missing model: {model_file}")

        base_options = python.BaseOptions(model_asset_path=str(model_file))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=output_segmentation_masks,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.last_result = None

    def detect(self, frame, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        self.last_result = result
        return result

    def get_bodies_data(self, result, frame_shape):
        """
        Lấy tọa độ 3D (x, y, z) cho toàn bộ 33 điểm landmark của cơ thể.
        """
        bodies_data = []
        if result is None or not result.pose_landmarks:
            return bodies_data

        VISIBILITY_THRESHOLD = 0.5 

        for pose_landmarks in result.pose_landmarks:
            # CẬP NHẬT: Lấy đủ x, y, z dưới dạng float chuẩn hóa.
            # Nếu visibility thấp, ta vẫn giữ tọa độ nhưng mô hình AI sẽ dựa vào mảng visibility để xử lý.
            points = [
                [float(lm.x), float(lm.y), float(lm.z)] 
                for lm in pose_landmarks
            ]
            visibility = [float(lm.visibility) for lm in pose_landmarks]

            bodies_data.append({
                "landmarks": points,
                "visibility": visibility,
            })
        return bodies_data

    def draw_bodies(self, frame, result):
        """
        Vẽ bộ khung xương lên frame 2D, bỏ qua trục z khi hiển thị.
        """
        if result is None or not result.pose_landmarks:
            return frame

        h, w, _ = frame.shape
        
        CONNECTIONS = [
            (11, 12), (11, 23), (12, 24), (23, 24), # Thân
            (11, 13), (12, 14), # Cánh tay trên
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), # Chân trái
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32), # Chân phải
        ]

        VISIBILITY_THRESHOLD = 0.5
        bodies_data = self.get_bodies_data(result, frame.shape)

        for body_data in bodies_data:
            points = body_data["landmarks"]
            visibility = body_data["visibility"]
            
            # Chuyển đổi về pixel để vẽ
            pts = [(int(p[0] * w), int(p[1] * h)) for p in points]

            # Vẽ các đường nối
            for start_idx, end_idx in CONNECTIONS:
                if visibility[start_idx] >= VISIBILITY_THRESHOLD and visibility[end_idx] >= VISIBILITY_THRESHOLD:
                    cv2.line(frame, pts[start_idx], pts[end_idx], (0, 255, 0), 2)

            # Vẽ các điểm khớp
            for i, (x, y) in enumerate(pts):
                # Bỏ qua mặt và các điểm ngón tay khi vẽ để bớt rối hình
                if i >= 11 and i not in range(15, 23) and visibility[i] >= VISIBILITY_THRESHOLD:
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        return frame

    def close(self):
        self.landmarker.close()