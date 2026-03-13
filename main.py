"""
main.py
FastAPI server — entry point for all Twilio webhooks and WebSocket connections.

Flow:
1. Twilio calls POST /inbound  → returns TwiML with WebSocket URL
2. Twilio opens WebSocket to /ws/{call_sid}
3. We spin up Pipecat pipeline for that call
4. Agent handles the conversation in real-time
"""
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse
from loguru import logger
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

from core.config import load_client_config, build_system_prompt
from core.agent import build_agent
from core.bot import run_voice_pipeline

load_dotenv()

# ── Logger setup ─────────────────────────────
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

# ── App State ─────────────────────────────────
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load client config + build agent once on startup."""
    logger.info("Starting Voice AI Agent server...")

    config = load_client_config()
    system_prompt = build_system_prompt(config)
    agent = build_agent(system_prompt, config)

    app_state["config"] = config
    app_state["agent"] = agent

    logger.info(f"Agent ready for: {config['practice_name']}")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Voice AI Agent", lifespan=lifespan)


# ─────────────────────────────────────────────
# INBOUND CALL WEBHOOK
# Twilio hits this when someone calls your number
# ─────────────────────────────────────────────

@app.post("/inbound")
async def inbound_call(request: Request):
    """
    Twilio webhook for inbound calls.
    Returns TwiML that connects the call to our WebSocket pipeline.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    from_number = form.get("From", "unknown")

    logger.info(f"Inbound call from {from_number} | CallSid: {call_sid}")

    # Build WebSocket URL — Twilio will stream audio here
    host = request.headers.get("host")
    ws_url = f"wss://{host}/ws/{call_sid}"

    # TwiML response — tells Twilio to stream audio to our WebSocket
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=ws_url)
    connect.append(stream)
    response.append(connect)

    return PlainTextResponse(
        content=str(response),
        media_type="text/xml"
    )


# ─────────────────────────────────────────────
# WEBSOCKET — Real-time audio stream per call
# ─────────────────────────────────────────────

@app.websocket("/ws/{call_sid}")
async def websocket_endpoint(websocket: WebSocket, call_sid: str):
    """
    One WebSocket connection per active call.
    Pipecat pipeline runs here — STT → LangGraph → TTS
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for call: {call_sid}")

    config = app_state["config"]
    agent = app_state["agent"]

    try:
        await run_voice_pipeline(
            websocket=websocket,
            agent=agent,
            config=config,
            call_sid=call_sid,
        )
    except Exception as e:
        logger.error(f"Pipeline error for {call_sid}: {e}")
    finally:
        logger.info(f"WebSocket closed for call: {call_sid}")


# ─────────────────────────────────────────────
# OUTBOUND CALL (bonus — for appointment reminders etc)
# ─────────────────────────────────────────────

@app.post("/outbound/initiate")
async def initiate_outbound(request: Request):
    """
    Initiate an outbound call — for appointment reminders, follow-ups etc.
    POST body: { "to": "+1234567890", "reason": "appointment_reminder" }
    """
    from twilio.rest import Client

    body = await request.json()
    to_number = body.get("to")
    reason = body.get("reason", "follow_up")

    if not to_number:
        return {"error": "missing 'to' number"}

    client = Client(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN")
    )

    host = request.headers.get("host")
    call = client.calls.create(
        to=to_number,
        from_=os.getenv("TWILIO_PHONE_NUMBER"),
        url=f"https://{host}/inbound",  # reuse same pipeline
    )

    logger.info(f"Outbound call initiated to {to_number}: {call.sid}")
    return {"call_sid": call.sid, "status": call.status}


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    config = app_state.get("config", {})
    return {
        "status": "ok",
        "client": config.get("practice_name", "unknown"),
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info",
    )
