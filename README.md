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
