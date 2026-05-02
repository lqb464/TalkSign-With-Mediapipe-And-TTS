import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import yaml

with open("configs/utils.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

FD_CFG = cfg["face_detector"]

class FaceDetector:
    def __init__(
        self,
        model_path: str | None = None,
        num_faces: int | None = None,
        min_face_detection_confidence: float | None = None,
        min_face_presence_confidence: float | None = None,
        min_tracking_confidence: float | None = None,
        output_face_blendshapes: bool | None = None,
    ):

        model_path = model_path or FD_CFG["model_path"]
        num_faces = num_faces or FD_CFG["num_faces"]
        min_face_detection_confidence = (
            min_face_detection_confidence or FD_CFG["min_face_detection_confidence"]
        )
        min_face_presence_confidence = (
            min_face_presence_confidence or FD_CFG["min_face_presence_confidence"]
        )
        min_tracking_confidence = (
            min_tracking_confidence or FD_CFG["min_tracking_confidence"]
        )
        output_face_blendshapes = (
            output_face_blendshapes
            if output_face_blendshapes is not None
            else FD_CFG.get("output_face_blendshapes", False)
        )

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Missing model: {model_file}")

        base_options = python.BaseOptions(model_asset_path=str(model_file))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=output_face_blendshapes,
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.last_result = None

    def detect(self, frame, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        self.last_result = result
        return result

    def get_faces_data(self, result, frame_shape):
        """
        Lấy tọa độ 3D (x, y, z) cho các điểm đặc trưng trên khuôn mặt.
        """
        faces_data = []
        if result is None or not result.face_landmarks:
            return faces_data

        # Danh sách các điểm quan trọng để vẽ/nhận diện biểu cảm
        ESSENTIAL_INDICES = {10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185}

        for face_landmarks in result.face_landmarks:
            # CẬP NHẬT: Lấy đủ x, y, z cho các điểm thiết yếu
            points = [
                [float(lm.x), float(lm.y), float(lm.z)] if idx in ESSENTIAL_INDICES else [0.0, 0.0, 0.0]
                for idx, lm in enumerate(face_landmarks)
            ]

            faces_data.append({"landmarks": points, "blendshapes": None})
        return faces_data

    def draw_faces(self, frame, result):
        """
        Vẽ lưới khuôn mặt 2D từ dữ liệu 3D.
        """
        if result is None or not result.face_landmarks:
            return frame

        h, w, _ = frame.shape

        FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
        LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33]
        RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]
        LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]

        ESSENTIAL_INDICES = set(FACE_OVAL + LEFT_EYE + RIGHT_EYE + LIPS + LEFT_EYEBROW + RIGHT_EYEBROW)

        faces_data = self.get_faces_data(result, frame.shape)

        for face_data in faces_data:
            points = face_data["landmarks"]
            # Chuyển đổi tọa độ float chuẩn hóa về pixel
            pts = [(int(p[0] * w), int(p[1] * h)) for p in points]

            regions = [FACE_OVAL, LEFT_EYE, RIGHT_EYE, LIPS, LEFT_EYEBROW, RIGHT_EYEBROW]
            for region in regions:
                for j in range(len(region) - 1):
                    cv2.line(frame, pts[region[j]], pts[region[j+1]], (0, 255, 0), 1)

            for idx in ESSENTIAL_INDICES:
                cv2.circle(frame, pts[idx], 1, (0, 0, 255), -1)

        return frame

    def close(self):
        self.landmarker.close()