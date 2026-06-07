import cv2
import boto3
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class VideoProcessor:
    def __init__(self):
        # AWS Credentials
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region = os.getenv('AWS_REGION')
        self.bucket_name = os.getenv('AWS_BUCKET_NAME')

        # Initialize AWS Clients
        self.rekognition = boto3.client(
            'rekognition',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region
        )
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region
        )

        # This list will hold the chronological metadata log
        self.metadata_timeline = []

    def process_video(self, video_path, frame_interval=30):
        """
        Processes video frames, extracts visual metadata using AWS Rekognition,
        and saves the accumulated timeline data to AWS S3.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Fallback to 30 if metadata fails
        frame_count = 0

        print(f"[INFO] Starting data pipeline processing for: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Calculate relative timestamp in seconds
                timestamp_sec = round(frame_count / fps, 2)
                print(f"[PROCESSING] Extracting metadata from frame #{frame_count} ({timestamp_sec}s)...")

                _, buffer = cv2.imencode('.jpg', frame)
                image_bytes = buffer.tobytes()

                try:
                    response = self.rekognition.detect_labels(
                        Image={'Bytes': image_bytes},
                        MaxLabels=10,
                        MinConfidence=75
                    )

                    # Log to terminal console for transparency
                    self._print_labels(response, frame_count)

                    # Store data into memory structure for pipeline export
                    self._accumulate_metadata(response, frame_count, timestamp_sec)

                except Exception as e:
                    print(f"[ERROR] AWS Rekognition failed at frame {frame_count}: {e}")

            frame_count += 1

        cap.release()
        print("[INFO] Frame analysis complete. Exporting pipeline logs to AWS S3...")
        self._upload_logs_to_s3()

    def _accumulate_metadata(self, response, frame_idx, timestamp_sec):
        """Appends structured frame data to the log timeline."""
        detected_objects = []
        for label in response['Labels']:
            detected_objects.append({
                "object": label['Name'],
                "confidence": round(label['Confidence'], 2)
            })

        frame_data = {
            "frame_id": frame_idx,
            "timestamp_seconds": timestamp_sec,
            "detections": detected_objects
        }
        self.metadata_timeline.append(frame_data)

    def _upload_logs_to_s3(self):
        """Compiles timeline data to JSON format and uploads it directly to Amazon S3."""
        if not self.metadata_timeline:
            print("[WARNING] Metadata timeline is empty. Aborting upload.")
            return

        payload = {
            "video_metadata_export": {
                "exported_at": datetime.utcnow().isoformat() + "Z",
                "total_frames_analyzed": len(self.metadata_timeline),
                "timeline": self.metadata_timeline
            }
        }

        # Convert dictionary payload to serialized JSON string string
        json_data = json.dumps(payload, indent=4)

        # Set up a clean file designation name in cloud
        log_filename = f"activity_logs/security_log_{int(datetime.utcnow().timestamp())}.json"

        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=log_filename,
                Body=json_data,
                ContentType='application/json'
            )
            print(f"[SUCCESS] Logs successfully persisted in S3 bucket '{self.bucket_name}'!")
            print(f"[S3 PATH] https://s3.console.aws.amazon.com/s3/buckets/{self.bucket_name}?prefix={log_filename}")
        except Exception as e:
            print(f"[ERROR] Failed to upload JSON file structure to S3 storage: {e}")

    def _print_labels(self, response, frame_idx):
        print(f"\n--- Live Frame Metadata #{frame_idx} ---")
        if not response['Labels']:
            print("No significant environment markers recognized.")
        for label in response['Labels']:
            print(f" -> {label['Name']} ({label['Confidence']:.2f}%)")
        print("-" * 40)