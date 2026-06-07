import cv2
import csv
import os
import numpy as np
from collections import deque
from ultralytics import YOLO


class PoseDataCollector:
    def __init__(self, sequence_length=10):
        print("[INFO] Loading YOLOv8 Pose model...")
        self.model = YOLO('yolov8n-pose.pt')
        self.sequence_length = sequence_length

    def process_video_file(self, video_path, csv_writer, label_name):
        """
        Processes a video file, extracts pose keypoints, and groups them
        into temporal sequences (sliding window) for LSTM training.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[CODEC ERROR] OpenCV cannot open or decode video file: {os.path.basename(video_path)}")
            return 0

        # Deque automatically drops the oldest frame when maxlen is reached,
        # creating a perfect sliding window effect.
        sequence = deque(maxlen=self.sequence_length)
        rows_written = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)

            # Check if any person is detected
            if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
                # For simplicity in this action recognition MVP, we track the primary person
                # (the first bounding box detected in the frame)
                person_keypoints = results[0].keypoints.xyn[0].cpu().numpy()

                coords = []
                for kp in person_keypoints:
                    coords.extend([kp[0], kp[1]])

                if len(coords) == 34:
                    sequence.append(coords)

                    # Only write to CSV if our temporal window is full (10 frames)
                    if len(sequence) == self.sequence_length:
                        row = [label_name]
                        for frame_coords in sequence:
                            row.extend(frame_coords)
                        csv_writer.writerow(row)
                        rows_written += 1

        cap.release()
        return rows_written

    def collect_from_directory(self, directory_path, output_csv, label_name):
        if not os.path.exists(directory_path):
            print(f"[ERROR] Directory not found: {directory_path}")
            return

        print(f"[INFO] Scanning directory '{directory_path}' for '{label_name}' samples...")
        file_exists = os.path.isfile(output_csv)

        with open(output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                # Dynamically generate headers for N frames (e.g., 10 frames * 34 coords)
                header = ['label']
                for frame_idx in range(self.sequence_length):
                    for i in range(17):
                        header.extend([f'f{frame_idx}_kp{i}_x', f'f{frame_idx}_kp{i}_y'])
                writer.writerow(header)

            valid_extensions = ('.avi', '.mpg', '.mp4')
            video_files = [f for f in os.listdir(directory_path) if f.lower().endswith(valid_extensions)]

            print(f"[INFO] Found {len(video_files)} valid video files.")
            total_rows = 0

            for index, filename in enumerate(video_files, 1):
                full_video_path = os.path.join(directory_path, filename)
                print(f"[PROCESSING] [{index}/{len(video_files)}] Extracting sequences from: {filename}")
                try:
                    added_rows = self.process_video_file(full_video_path, writer, label_name)
                    total_rows += added_rows
                except Exception as e:
                    print(f"[WARNING] Failed to process {filename}: {e}")

        print(f"[SUCCESS] Extracted {total_rows} temporal sequences for class '{label_name}'.")