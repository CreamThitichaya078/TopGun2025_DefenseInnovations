import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import joblib
from pathlib import Path

# เป็นโครงหลักของระบบที่รวมทุกโมดูลเข้าด้วยกัน
class DroneTrackingSystem:
    """
    ระบบติดตามและทำนายพิกัดโดรนแบบเรียลไทม์
    รวมการตรวจจับ (YOLO), การติดตาม (BYTETracker), และการทำนายพิกัด (XGBoost)
    พร้อมการแสดงผลแบบเรียลไทม์บนหน้าจอ
    """
    # โหลดโมเดล YOLO (detect + track) และ XGBoost (predict coordinate)
    def __init__(self, yolo_model_path, xgb_model_path):
        """
        เริ่มต้นระบบโดยโหลดโมเดลทั้งสอง

        Parameters:
        - yolo_model_path: path ไปยังไฟล์ .pt ของ YOLOv8n ที่ train แล้ว
        - xgb_model_path: path ไปยังไฟล์ .joblib ของ XGBoost
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.xgb_model = joblib.load(xgb_model_path)
        self.track_colors = {}
        self.tracks_path = {}

        print("✓ โหลดโมเดลทั้งหมดเรียบร้อยแล้ว")
    # สุ่มสีเฉพาะสำหรับแต่ละ track_id
    def get_track_color(self, track_id):
        """
        สร้างหรือดึงสีที่กำหนดให้กับ track_id นั้นๆ
        ทำให้โดรนแต่ละลำมีสีที่แตกต่างกันและคงที่ตลอดวิดีโอ
        """
        if track_id not in self.track_colors:
            np.random.seed(int(track_id))
            color = tuple(map(int, np.random.randint(50, 255, 3)))
            self.track_colors[track_id] = color
        return self.track_colors[track_id]
    # ทำนายพิกัดจาก bounding box ที่ YOLO ตรวจพบ
    def predict_coordinates(self, detections_df):
        """
        ทำนายพิกัด (lat, lon, alt) จากข้อมูลการตรวจจับ

        Parameters:
        - detections_df: DataFrame ที่มี columns: center_x, center_y, width, height

        Returns:
        - DataFrame ที่มี columns: Latitude, Longitude, Altitude
        """
        if len(detections_df) == 0:
            return pd.DataFrame(columns=['Latitude', 'Longitude', 'Altitude'])

        features = detections_df[['center_x', 'center_y', 'width', 'height']]
        predictions = self.xgb_model.predict(features)
        coords_df = pd.DataFrame(predictions, columns=['Latitude', 'Longitude', 'Altitude'])

        return coords_df
    # แสดง แผงข้อมูลด้านขวาบน ของวิดีโอ เช่น
    # Frame ปัจจุบัน
    # จำนวนโดรนที่ตรวจพบ
    # FPS (frame per second)
    # สถานะ paused
    def draw_info_panel(self, frame, frame_count, total_frames, num_drones, fps_display, paused=False):
        """
        วาดแผงข้อมูลที่มุมบนขวาของหน้าจอ เพื่อแสดงสถานะต่างๆ ของระบบ

        Parameters:
        - frame: เฟรมที่จะวาดข้อมูลลงไป
        - frame_count: หมายเลขเฟรมปัจจุบัน
        - total_frames: จำนวนเฟรมทั้งหมด
        - num_drones: จำนวนโดรนที่ตรวจพบในเฟรมนี้
        - fps_display: ค่า FPS ในการประมวลผลจริง
        - paused: สถานะการหยุดชั่วคราว
        """
        height, width = frame.shape[:2]

        # สร้างพื้นหลังสำหรับแผงข้อมูล
        panel_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (width - 300, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # เตรียมข้อความที่จะแสดง
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        y_offset = 25

        # แสดงหมายเลขเฟรม
        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        frame_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
        cv2.putText(frame, frame_text, (width - 290, y_offset),
                    font, font_scale, (255, 255, 255), thickness)

        # แสดงจำนวนโดรนที่ตรวจพบ
        y_offset += 30
        drone_text = f"Drones: {num_drones}"
        color = (0, 255, 0) if num_drones > 0 else (128, 128, 128)
        cv2.putText(frame, drone_text, (width - 290, y_offset),
                    font, font_scale, color, thickness)

        # แสดง FPS ที่ประมวลผลได้จริง
        y_offset += 30
        fps_text = f"FPS: {fps_display:.1f}"
        cv2.putText(frame, fps_text, (width - 290, y_offset),
                    font, font_scale, (255, 255, 0), thickness)

        # แสดงสถานะการหยุดชั่วคราว
        if paused:
            y_offset += 30
            cv2.putText(frame, "PAUSED", (width - 290, y_offset),
                        font, font_scale, (0, 0, 255), thickness)

        return frame

    # คำแนะนำการควบคุม ที่มุมล่างซ้าย
    def draw_instructions(self, frame):
        """
        วาดคำแนะนำการใช้งานที่มุมล่างซ้ายของหน้าจอ
        เพื่อให้ผู้ใช้ทราบว่าสามารถกดปุ่มอะไรได้บ้าง
        """
        instructions = [
            "Controls:",
            "SPACE - Pause/Resume",
            "Q/ESC - Quit",
            "S - Save current frame"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        y_start = frame.shape[0] - 100

        # วาดพื้นหลังโปร่งแสง
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y_start - 25), (250, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # วาดข้อความทีละบรรทัด
        for i, instruction in enumerate(instructions):
            y_position = y_start + (i * 20)
            cv2.putText(frame, instruction, (15, y_position),
                        font, font_scale, (255, 255, 255), thickness)

        return frame
    # เป็น “หัวใจ” ของโปรแกรม — ทำงานกับวิดีโอทีละเฟรม
    def process_video(self, video_path, output_path, conf_threshold=0.01, show_display=True,
                      display_scale=1.0, save_frames=False):
        """
        ประมวลผลวิดีโอทีละเฟรม พร้อมแสดงผลการติดตามและพิกัดแบบเรียลไทม์

        Parameters:
        - video_path: path ของวิดีโออินพุต
        - output_path: path สำหรับบันทึกวิดีโอผลลัพธ์
        - conf_threshold: ค่าความมั่นใจขั้นต่ำสำหรับการตรวจจับ (0-1)
        - show_display: แสดงหน้าต่างวิดีโอระหว่างประมวลผลหรือไม่
        - display_scale: ขนาดของหน้าต่างแสดงผล (1.0 = ขนาดเต็ม, 0.5 = ครึ่งหนึ่ง)
        - save_frames: บันทึกเฟรมแต่ละเฟรมเป็นไฟล์ภาพหรือไม่
        """
        # เปิดวิดีโอ
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"ไม่สามารถเปิดไฟล์วิดีโอ: {video_path}")

        # ดึงข้อมูลวิดีโอ
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n📹 ข้อมูลวิดีโอ:")
        print(f"   - ความละเอียด: {width}x{height}")
        print(f"   - FPS: {fps}")
        print(f"   - จำนวนเฟรม: {total_frames}")
        print(f"   - ความยาว: {total_frames / fps:.2f} วินาที")

        # สร้างวิดีโออัพพุต
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # ตัวแปรสำหรับควบคุมการทำงาน
        frame_count = 0
        paused = False

        # ตัวแปรสำหรับคำนวณ FPS จริง
        import time
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_display = 0

        # สร้างหน้าต่างแสดงผลถ้าต้องการ
        if show_display:
            window_name = "Drone Tracking System - Real-time View"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # ปรับขนาดหน้าต่างตาม scale ที่กำหนด
            display_width = int(width * display_scale)
            display_height = int(height * display_scale)
            cv2.resizeWindow(window_name, display_width, display_height)

        print(f"\n🚁 เริ่มประมวลผลวิดีโอ...")
        if show_display:
            print(f"   💡 กด SPACE เพื่อหยุดชั่วคราว, Q หรือ ESC เพื่อออก, S เพื่อบันทึกเฟรมปัจจุบัน")

        # สร้างโฟลเดอร์สำหรับบันทึกเฟรม (ถ้าต้องการ)
        if save_frames:
            frames_dir = Path("saved_frames")
            frames_dir.mkdir(exist_ok=True)

        while True:
            # ถ้าไม่ได้หยุดชั่วคราว ให้อ่านเฟรมใหม่
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                fps_frame_count += 1

                # คำนวณ FPS จริงทุกๆ 30 เฟรม
                if fps_frame_count >= 30:
                    fps_end_time = time.time()
                    fps_display = fps_frame_count / (fps_end_time - fps_start_time)
                    fps_start_time = time.time()
                    fps_frame_count = 0

                # ใช้ YOLO ตรวจจับและติดตามโดรนด้วย BYTETracker
                results = self.yolo_model.track(
                    frame,
                    persist=True,
                    conf=conf_threshold,
                    iou=0.25,
                    tracker="bytetrack.yaml",
                    verbose=False  # ปิดการแสดงผลข้อความจาก YOLO
                )

                # ดึงข้อมูลการตรวจจับ
                detections = []

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()

                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        center_x, center_y, w, h = box

                        # ปรับค่าให้เป็น normalized coordinates
                        center_x_norm = center_x / width
                        center_y_norm = center_y / height
                        w_norm = w / width
                        h_norm = h / height

                        detections.append({
                            'track_id': track_id,
                            'center_x': center_x_norm,
                            'center_y': center_y_norm,
                            'width': w_norm,
                            'height': h_norm,
                            'center_x_pixel': center_x,
                            'center_y_pixel': center_y,
                            'w_pixel': w,
                            'h_pixel': h,
                            'confidence': conf
                        })



                # สร้างเฟรมสำหรับแสดงผล
                display_frame = frame.copy()

                # --- วาดเส้น tracking ---
                for det in detections:
                    track_id = det['track_id']
                    cx, cy = det['center_x_pixel'], det['center_y_pixel']

                    if track_id not in self.tracks_path:
                        self.tracks_path[track_id] = []
                    self.tracks_path[track_id].append((cx, cy))
                    if len(self.tracks_path[track_id]) > 50:  # เก็บ 50 จุดล่าสุด
                        self.tracks_path[track_id].pop(0)

                for tid, points in self.tracks_path.items():
                    color = self.get_track_color(tid)
                    for i in range(1, len(points)):
                        if points[i - 1] is None or points[i] is None:
                            continue
                        cv2.line(display_frame, points[i - 1], points[i], color, 2)

                # ถ้ามีโดรนที่ตรวจพบ ให้ทำนายพิกัดและวาดผลลัพธ์
                if len(detections) > 0:
                    detections_df = pd.DataFrame(detections)
                    coords_df = self.predict_coordinates(
                        detections_df[['center_x', 'center_y', 'width', 'height']]
                    )

                    # วาดผลลัพธ์บนเฟรม
                    for idx, det in enumerate(detections):
                        track_id = det['track_id']

                        # คำนวณมุมของ bounding box
                        x1 = int(det['center_x_pixel'] - det['w_pixel'] / 2)
                        y1 = int(det['center_y_pixel'] - det['h_pixel'] / 2)
                        x2 = int(det['center_x_pixel'] + det['w_pixel'] / 2)
                        y2 = int(det['center_y_pixel'] + det['h_pixel'] / 2)

                        # เลือกสีตาม track_id
                        color = self.get_track_color(track_id)

                        # วาดกรอบสี่เหลี่ยม (หนาขึ้นเล็กน้อยเพื่อให้เห็นชัดเจน)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

                        # ดึงพิกัดที่ทำนายได้
                        lat = coords_df.iloc[idx]['Latitude']
                        lon = coords_df.iloc[idx]['Longitude']
                        alt = coords_df.iloc[idx]['Altitude']
                        conf = det['confidence']

                        # จัดรูปแบบข้อความ
                        label_id = f"ID: {track_id} ({conf:.2f})"
                        label_coords = f"Lat: {lat:.6f}"
                        label_lon = f"Lon: {lon:.6f}"
                        label_alt = f"Alt: {alt:.2f}m"

                        # ตั้งค่าฟอนต์
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2

                        # คำนวณขนาดข้อความเพื่อสร้างพื้นหลัง
                        labels = [label_id, label_coords, label_lon, label_alt]
                        max_width = 0
                        line_height = 0

                        for label in labels:
                            (w_text, h_text), _ = cv2.getTextSize(label, font, font_scale, thickness)
                            max_width = max(max_width, w_text)
                            line_height = max(line_height, h_text)

                        # วาดพื้นหลังสำหรับข้อความ
                        padding = 5
                        text_y_start = y1 - (len(labels) * (line_height + padding)) - padding

                        # ตรวจสอบว่าข้อความไม่ล้นออกนอกเฟรม
                        if text_y_start < 0:
                            text_y_start = y2 + padding

                        overlay = display_frame.copy()
                        cv2.rectangle(overlay,
                                      (x1, text_y_start),
                                      (x1 + max_width + (padding * 2),
                                       text_y_start + (len(labels) * (line_height + padding)) + padding),
                                      (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

                        # วาดข้อความทีละบรรทัด
                        current_y = text_y_start + line_height + padding
                        for i, label in enumerate(labels):
                            text_color = color if i == 0 else (255, 255, 255)
                            cv2.putText(display_frame, label, (x1 + padding, current_y),
                                        font, font_scale, text_color, thickness)
                            current_y += line_height + padding

                # วาดแผงข้อมูลและคำแนะนำ
                display_frame = self.draw_info_panel(display_frame, frame_count, total_frames,
                                                     len(detections), fps_display, paused)
                if show_display:
                    display_frame = self.draw_instructions(display_frame)

                # เขียนเฟรมลงในไฟล์วิดีโอ
                out.write(display_frame)

            # แสดงผลบนหน้าจอ
            if show_display:
                cv2.imshow(window_name, display_frame)

                # รอรับคำสั่งจากแป้นพิมพ์
                key = cv2.waitKey(1 if not paused else 0) & 0xFF

                # กด SPACE เพื่อหยุดชั่วคราวหรือเล่นต่อ
                if key == ord(' '):
                    paused = not paused
                    if paused:
                        print(f"\n⏸️  หยุดชั่วคราวที่เฟรม {frame_count}")
                    else:
                        print(f"▶️  เล่นต่อจากเฟรม {frame_count}")
                        fps_start_time = time.time()  # รีเซ็ตตัวนับ FPS
                        fps_frame_count = 0

                # กด Q หรือ ESC เพื่อออกจากโปรแกรม
                elif key == ord('q') or key == 27:
                    print(f"\n⏹️  หยุดการประมวลผลที่เฟรม {frame_count}")
                    break

                # กด S เพื่อบันทึกเฟรมปัจจุบัน
                elif key == ord('s') and save_frames:
                    save_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(save_path), display_frame)
                    print(f"💾 บันทึกเฟรมไว้ที่: {save_path}")

            # แสดง progress ทุกๆ 30 เฟรม
            if not paused and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   ความคืบหน้า: {progress:.1f}% ({frame_count}/{total_frames} เฟรม) | FPS: {fps_display:.1f}")

        # ปิดทุกอย่าง
        cap.release()
        out.release()

        if show_display:
            cv2.destroyAllWindows()

        print(f"\n✅ เสร็จสิ้น! บันทึกวิดีโอไว้ที่: {output_path}")
        print(f"   ประมวลผลทั้งหมด {frame_count} เฟรม")
        print(f"   เวลารวม: {(frame_count / fps):.2f} วินาที")


# ==================== การใช้งาน ====================

if __name__ == "__main__":
    # กำหนด paths ของโมเดลและวิดีโอ
    YOLO_MODEL_PATH = "best (Punch).pt"
    XGB_MODEL_PATH = "XGB_model.joblib"
    INPUT_VIDEO = "P3_VIDEO.mp4"
    OUTPUT_VIDEO = "output_tracked_video.mp4"

    try:
        # สร้างระบบติดตามโดรน
        system = DroneTrackingSystem(YOLO_MODEL_PATH, XGB_MODEL_PATH)

        # ประมวลผลวิดีโอพร้อมแสดงผลแบบเรียลไทม์
        system.process_video(
            video_path=INPUT_VIDEO,
            output_path=OUTPUT_VIDEO,
            conf_threshold=0.01,
            show_display=True,  # เปิดการแสดงผลแบบเรียลไทม์
            display_scale=0.8,  # ปรับขนาดหน้าต่างเป็น 80% ของขนาดจริง
            save_frames=True  # เปิดการบันทึกเฟรมเมื่อกด S
        )

    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {str(e)}")
        import traceback

        traceback.print_exc()