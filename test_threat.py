import os
from src.threat_detector import ThreatDetector

if __name__ == "__main__":
    detector = ThreatDetector()

    # Define input and output paths
    input_video = "data/fight_video.mp4"
    output_video = "data/analyzed_threat.mp4"

    if os.path.exists(input_video):
        detector.analyze_video(input_video, output_video)
    else:
        print(f"[ERROR] Please place '{input_video}' in the data folder.")