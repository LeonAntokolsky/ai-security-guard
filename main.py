import os
from src.vision_module import VideoProcessor
from src.agent_module import SecurityAIAgent

if __name__ == "__main__":
    # Initialize both core modular layers of our tech stack
    processor = VideoProcessor()
    agent = SecurityAIAgent()

    video_file = "data/test_video.mp4"

    # Optional shortcut: Skip processing if you already uploaded your target log file to S3
    run_video_analysis = input("Do you want to process the video stream? (y/n): ").strip().lower()

    if run_video_analysis == 'y':
        if os.path.exists(video_file):
            processor.process_video(video_file, frame_interval=30)
        else:
            print(f"[ERROR] File '{video_file}' was not found in local system space.")
            exit(1)

    print("\n" + "=" * 50)
    print("🤖 AI SECURITY AGENT INTERFACE ONLINE")
    print("Type 'exit' or 'quit' to terminate the session.")
    print("=" * 50)

    # Enter perpetual conversation loop for demo presentation testing
    while True:
        query = input("\nYou: ")
        if query.strip().lower() in ['exit', 'quit']:
            print("Shutting down core AI subsystems. Goodbye.")
            break

        if not query.strip():
            continue

        print("\n[AGENT REASONING LOOP START]")
        try:
            response = agent.ask(query)
            print("\n" + "-" * 50)
            print(f"Agent Response: {response}")
            print("-" * 50)
        except Exception as e:
            print(f"[CRITICAL ERROR] Execution failed: {e}")