# MediVoice AI

**Description:** A highly configurable, real-time voice AI assistant designed for medical practices to handle inbound patient calls, book appointments, request prescription refills, and intelligently route urgent queries. Built exclusively for **Rahees Ahmed**.

**Stack:** FastAPI + Pipecat + LangGraph + Twilio + Deepgram (STT) + Cartesia (TTS) + Anthropic Claude 3.5 Haiku

---

## Architecture

```
Caller dials Twilio number
  → POST /inbound  (Twilio webhook)
    → TwiML returns WebSocket URL
      → Twilio streams audio to /ws/{call_sid}
        → Pipecat pipeline starts
          → Deepgram STT  (audio → text, ~200ms)
            → LangGraph Agent  (text → intent + tool calls)
              → Cartesia TTS  (text → audio, ~80ms)
                → Back to caller
```

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/RaheesAhmed/MediVoice-AI.git
cd MediVoice-AI

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Environment
```bash
cp .env.example .env
# Fill in all API keys in .env
```

### 3. Configure Your Client
```bash
# Copy and edit the sample config
cp config/dr_smith_cardiology.yaml config/YOUR_CLIENT.yaml
# Edit it, then set CLIENT_ID=YOUR_CLIENT in .env
```

### 4. Run Server
```bash
python main.py
```

### 5. Expose to Internet (dev)
```bash
ngrok http 8000
# Copy the https URL e.g. https://abc123.ngrok.io
```

### 6. Configure Twilio
1. Go to Twilio Console → Phone Numbers → Your Number
2. Set webhook URL: `https://abc123.ngrok.io/inbound`
3. Method: HTTP POST
4. Call your number — agent answers!

---

## Adding a New Client

Zero code changes. Just:
```bash
cp config/dr_smith_cardiology.yaml config/new_client.yaml
# Edit the YAML with new client details
# Set CLIENT_ID=new_client in .env
# Restart server
```

---

## Project Structure

```
voice-ai-agent/
├── main.py                          # FastAPI server + Twilio webhooks
├── core/
│   ├── bot.py                       # Pipecat pipeline (audio processing)
│   ├── agent.py                     # LangGraph agent + tools
│   └── config.py                    # Config loader + system prompt builder
├── config/
│   └── dr_smith_cardiology.yaml     # Per-client config
├── requirements.txt
└── .env.example
```

---

## Tools Available to Agent

| Tool | What it does |
|------|-------------|
| `check_appointment_availability` | Check open slots |
| `book_appointment` | Book via webhook |
| `request_prescription_refill` | Submit refill request |
| `take_message` | Record message for callback |
| `transfer_to_human` | Immediately transfer call |

---

## Deployment (Production)

```bash
# Railway / Render / VPS
# No ngrok needed in prod — use your real domain

# Set env vars on your host
# Update Twilio webhook to your production URL
# Run: python main.py
```

---

## Key API Keys Needed

| Service | Purpose | Free tier? |
|---------|---------|-----------|
| Anthropic | LLM brain | No (cheap) |
| Deepgram | STT | $200 free credits |
| Cartesia | TTS | Free tier |
| Twilio | Phone calls | Trial available |
