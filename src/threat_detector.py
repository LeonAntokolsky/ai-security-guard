import cv2
import numpy as np
from ultralytics import YOLO
import os
import requests
import json
import datetime
import boto3
import tensorflow as tf
from collections import deque
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ThreatDetector:
    def __init__(self, sequence_length=10):
        print("[INFO] Loading YOLOv8 Pose model...")
        self.pose_model = YOLO('yolov8n-pose.pt')

        print("[INFO] Loading trained TensorFlow LSTM Temporal model...")
        self.ml_model = tf.keras.models.load_model('data/threat_model_lstm.keras')

        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        self.alert_sent = False

        # Temporal memory sliding window
        self.sequence_length = sequence_length
        self.pose_memory = deque(maxlen=self.sequence_length)

        # Debounce filter to manage pre-fight stances
        self.consecutive_threats = 0
        self.alert_threshold = 70

        # Initialize AWS S3 client via boto3 SDK
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_DEFAULT_REGION')
            )
            print("[INFO] AWS S3 Pipeline initialized successfully.")
        except Exception as e:
            print(f"[WARNING] Could not connect to real AWS hardware. Using mock fallback mode: {e}")
            self.s3_client = None

    def _upload_to_s3(self, local_path, s3_key):
        """Uploads critical file data assets straight to cloud S3 bucket storage."""
        if not self.s3_client or not self.bucket_name:
            print(
                f"[MOCK AWS] Simulating successful cloud upload of '{local_path}' -> S3://{self.bucket_name}/{s3_key}")
            return f"https://s3.mock-simulation.amazonaws.com/{self.bucket_name}/{s3_key}"

        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            print(f"[AWS SUCCESS] Resource uploaded smoothly to cloud bucket: {s3_key}")
            return f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
        except Exception as e:
            print(f"[AWS ERROR] Cloud transfer pipeline failed: {e}")
            return None

    def _process_incident_pipeline(self, raw_frame, annotated_frame, confidence):
        """Executes the full enterprise incident response: Telegram, local log, and AWS upload."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"\n[INCIDENT PIPELINE INITIATED] Timestamp: {timestamp}")

        # 1. Generate local and cloud image assets
        combined_frame = cv2.hconcat([raw_frame, annotated_frame])
        height, width = combined_frame.shape[:2]
        high_res_frame = cv2.resize(combined_frame, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

        local_img_path = f"data/incident_{timestamp}.jpg"
        cv2.imwrite(local_img_path, high_res_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # 2. Upload image to S3 cloud storage
        s3_image_key = f"images/incident_{timestamp}.jpg"
        s3_image_url = self._upload_to_s3(local_img_path, s3_image_key)

        # 3. Create structural JSON log payload
        incident_data = {
            "incident_id": f"INC-{timestamp}",
            "timestamp": timestamp,
            "threat_type": "Physical Altercation / Fight",
            "detection_model": "TensorFlow LSTM Temporal Network",
            "confidence_score": f"{confidence:.2f}%",
            "camera_id": "CAM-04-NORTH-CORRIDOR",
            "evidence_image_s3_url": s3_image_url,
            "status": "UNRESOLVED - CRITICAL"
        }

        local_json_path = f"data/incident_{timestamp}.json"
        with open(local_json_path, 'w') as json_file:
            json.dump(incident_data, json_file, indent=4)
        print(f"[INFO] Technical operational telemetry saved locally: {local_json_path}")

        # 4. Upload JSON metadata telemetry pack to S3
        s3_json_key = f"logs/incident_{timestamp}.json"
        self._upload_to_s3(local_json_path, s3_json_key)

        # 5. Dispatch legacy emergency alert directly to Telegram chat
        if self.bot_token and self.chat_id:
            _, buffer = cv2.imencode('.jpg', high_res_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            payload = {
                'chat_id': self.chat_id,
                'caption': f"🚨 CRITICAL ALERT: Fight Confirmed ({confidence:.1f}%)!\nTelemetry dispatched to AWS S3 storage node."
            }
            files = {'photo': buffer.tobytes()}
            try:
                requests.post(url, data=payload, files=files)
                print("[SUCCESS] Operational Telegram dispatch finished.")
            except Exception as e:
                print(f"[ERROR] Chat dispatch network failure: {e}")

    def analyze_video(self, input_path, output_path):
        """Analyzes video streams using Pose Estimation combined with an LSTM Neural Network."""
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.alert_sent = False
        self.pose_memory.clear()
        self.consecutive_threats = 0

        print(f"[INFO] Processing video for LSTM threat detection: {input_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame = frame.copy()
            results = self.pose_model(frame, verbose=False)
            annotated_frame = results[0].plot()

            if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
                person_keypoints = results[0].keypoints.xyn[0].cpu().numpy()
                coords = []
                for kp in person_keypoints:
                    coords.extend([kp[0], kp[1]])

                if len(coords) == 34:
                    self.pose_memory.append(coords)

                    if len(self.pose_memory) == self.sequence_length:
                        input_data = np.array([list(self.pose_memory)], dtype=np.float32)
                        prediction_prob = self.ml_model.predict(input_data, verbose=0)[0][0]

                        if prediction_prob < 0.15:
                            self.consecutive_threats += 1
                            confidence = (1.0 - prediction_prob) * 100

                            cv2.putText(annotated_frame, f"TRACKING STANCE... ({confidence:.1f}%)",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

                            if self.consecutive_threats >= self.alert_threshold:
                                cv2.putText(annotated_frame, "CRITICAL: ACTIVE FIGHT!",
                                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                                if not self.alert_sent:
                                    self._process_incident_pipeline(raw_frame, annotated_frame, confidence)
                                    self.alert_sent = True
                        else:
                            self.consecutive_threats = max(0, self.consecutive_threats - 2)

            out.write(annotated_frame)

        cap.release()
        out.release()
        print(f"[SUCCESS] LSTM Video analysis complete. Output saved to: {output_path}")