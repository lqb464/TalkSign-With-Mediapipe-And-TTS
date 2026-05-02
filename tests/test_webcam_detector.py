import time
import cv2
from src.utils.body_detector import BodyDetector
from src.utils.face_detector import FaceDetector
from src.utils.hand_detector import HandDetector
from src.utils.webcam import Webcam

def main():
    print("Khởi động webcam...")
    cam = Webcam()

    print("Khởi động các bộ Detectors...")
    hand_detector = HandDetector()
    face_detector = FaceDetector()
    body_detector = BodyDetector()

    print("Bắt đầu. Nhấn 'Esc' để thoát.\n")

    fps = 0.0
    prev_time = time.time()

    while True:
        frame = cam.read()
        if frame is None:
            print("Không đọc được frame.")
            break

        h, w, _ = frame.shape
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now
        timestamp_ms = int(now * 1000)

        # 1. Chạy Detection
        hand_result = hand_detector.detect(frame, timestamp_ms)
        face_result = face_detector.detect(frame, timestamp_ms)
        body_result = body_detector.detect(frame, timestamp_ms)

        # 2. Lấy dữ liệu (Dạng float 0.0 - 1.0)
        hands_data  = hand_detector.get_hands_data(hand_result, frame.shape)
        faces_data  = face_detector.get_faces_data(face_result, frame.shape)
        bodies_data = body_detector.get_bodies_data(body_result, frame.shape)

        # 3. Vẽ landmarks lên frame bằng các hàm nội bộ của Detector (đã xử lý tọa độ bên trong)
        frame = body_detector.draw_bodies(frame, body_result)
        
        # --- LOGIC NỐI CÙI TRỎ VỚI CỔ TAY (ĐÃ CẬP NHẬT TỌA ĐỘ) ---
        for body in bodies_data:
            b_pts = body["landmarks"]
            b_vis = body["visibility"] 
            
            for hand in hands_data:
                h_pts = hand["landmarks"]
                
                # Chuyển đổi tọa độ float sang pixel (int) trước khi tính toán và vẽ
                wrist_x, wrist_y = int(h_pts[0][0] * w), int(h_pts[0][1] * h)
                
                elbow_l_x, elbow_l_y = int(b_pts[13][0] * w), int(b_pts[13][1] * h)
                elbow_r_x, elbow_r_y = int(b_pts[14][0] * w), int(b_pts[14][1] * h)
                
                # Tính khoảng cách dựa trên pixel thực tế
                dist_l = ((elbow_l_x - wrist_x)**2 + (elbow_l_y - wrist_y)**2)**0.5
                dist_r = ((elbow_r_x - wrist_x)**2 + (elbow_r_y - wrist_y)**2)**0.5
                
                # Nối với bên gần hơn và đảm bảo điểm đó có độ tin cậy tốt
                if dist_l < dist_r:
                    if b_vis[13] > 0.5:
                        cv2.line(frame, (elbow_l_x, elbow_l_y), (wrist_x, wrist_y), (0, 255, 0), 2)
                else:
                    if b_vis[14] > 0.5:
                        cv2.line(frame, (elbow_r_x, elbow_r_y), (wrist_x, wrist_y), (0, 255, 0), 2)
        # -------------------------------------------------------

        frame = face_detector.draw_faces(frame, face_result)
        frame = hand_detector.draw_hands(frame, hand_result)

        # 4. Hiển thị bảng thông số
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        line_type = cv2.LINE_AA

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), font, font_scale, (200, 200, 200), thickness, line_type)
        cv2.putText(frame, f"Hands: {len(hands_data)}", (20, 65), font, font_scale, (255, 255, 0), thickness, line_type)
        cv2.putText(frame, f"Faces: {len(faces_data)}", (20, 95), font, font_scale, (0, 200, 255), thickness, line_type)
        cv2.putText(frame, f"Bodies: {len(bodies_data)}", (20, 125), font, font_scale, (0, 165, 255), thickness, line_type)

        cv2.imshow("TalkSign - Integrated Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    hand_detector.close()
    face_detector.close()
    body_detector.close()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()