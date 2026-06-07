# SmartGuard AI: Autonomous Security Incident Responder

SmartGuard AI is an autonomous, end-to-end video analytics and incident response system. It leverages real-time Edge Computer Vision to monitor security feeds, uses a Deep Learning sequential network to accurately classify complex human behavior over time, and orchestrates cloud infrastructure alongside LLM agents to automatically generate structured enterprise-grade security reports.

## Tech Stack & Architecture
- **Language:** Python 3.10+
- **Edge Computer Vision:** YOLOv8-Pose (Extracts continuous 2D spatial coordinates of human skeleton joints)
- **Deep Learning / Temporal Layers:** TensorFlow / Keras (Custom LSTM Network for sequence and motion analysis)
- **Cloud Infrastructure:** AWS S3 (Secure, immutable storage for high-resolution snapshots and machine logs)
- **LLM Orchestration:** LangChain & Gemini 2.5 Flash (Parses mathematical telemetry into formal corporate incident reports)
- **Emergency Notifications:** Telegram Bot API (Instant multimedia dispatch loops)

## Key Production Capabilities

### 1. Biomechanical Temporal Dynamics (LSTM)
Instead of relying on single, static image classifications—which trigger mass false alarms on benign interactions like handshakes or quick movements—this system captures spatial dynamics over a sliding time-window. The custom **LSTM layer** processes historical joint trajectories mathematically, understanding patterns of physical velocity, stance acceleration, and aggressive proximity:

$$\text{Equation applied to internal recurrent units: } h_t = \sigma(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$

### 2. High-Accuracy Debounce System
To safeguard production pipelines from transient spikes, a **15-frame validation debounce filter** was developed. The threat detector requires a sustained aggressive

### 3. Deployment & Installation
Clone Repository:

Bash
git clone git@github.com:LeonAntokolsky/ai-security-guard.git
cd ai-security-guard
Configure Local Environment:
Initialize your virtual environment and install the verified hardware and software libraries:

Bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Secure Environment Variables (.env):
Create a root-level .env file to contain cloud and notification secrets (Note: This file is blocked by .gitignore and must never be committed):

Plaintext
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET_NAME=ai-concierge-logs-security-project
GEMINI_API_KEY=your_gemini_api_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
Execute System Inference:

Bash
python3 test_threat.py
