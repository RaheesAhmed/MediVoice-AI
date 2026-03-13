"""
core/bot.py
Pipecat voice pipeline — the real-time audio processing core.
Connects Twilio audio ↔ Deepgram STT ↔ LangGraph Agent ↔ Cartesia TTS
"""
import os
import asyncio
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import LLMMessagesFrame, EndFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.vad.silero import SileroVADAnalyzer

from langchain_core.messages import HumanMessage, AIMessage


# ─────────────────────────────────────────────
# LANGGRAPH FRAME PROCESSOR
# Bridges Pipecat's frame world → LangGraph agent
# ─────────────────────────────────────────────

class LangGraphProcessor(FrameProcessor):
    """
    Wraps a compiled LangGraph graph as a Pipecat FrameProcessor.
    Receives text from STT, runs it through the agent, returns response text to TTS.
    """

    def __init__(self, agent, config: dict, call_sid: str):
        super().__init__()
        self.agent = agent
        self.config = config
        self.call_sid = call_sid
        self.thread_id = call_sid  # unique memory per call
        self.transfer_number = config.get("transfer_number")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMMessagesFrame):
            # Extract the latest user message from context
            messages = frame.messages
            user_text = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_text = msg.get("content", "")
                    break

            if user_text:
                response_text = await self._run_agent(user_text)
                # Push response downstream to TTS
                await self.push_frame(TextFrame(response_text))
        else:
            await self.push_frame(frame, direction)

    async def _run_agent(self, user_input: str) -> str:
        """Run LangGraph agent and return response text."""
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            state = {"messages": [HumanMessage(content=user_input)]}

            result = await asyncio.to_thread(
                self.agent.invoke, state, config
            )

            # Extract last AI message
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content

                    # Check if agent requested a human transfer
                    if "TRANSFER_NOW" in content:
                        logger.info(f"Transfer triggered for call {self.call_sid}")
                        return "Of course, let me transfer you to our staff right away. Please hold."

                    return content

            return "I apologize, I didn't catch that. Could you repeat please?"

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return "I'm having a brief technical issue. Let me transfer you to our staff."


# ─────────────────────────────────────────────
# MAIN PIPELINE RUNNER
# Called once per inbound call
# ─────────────────────────────────────────────

async def run_voice_pipeline(websocket, agent, config: dict, call_sid: str):
    """
    Spin up the full Pipecat pipeline for one call.
    websocket: FastAPI WebSocket from Twilio Media Stream
    """
    greeting = config.get("agent", {}).get(
        "greeting",
        f"Thank you for calling {config.get('practice_name')}. How can I help you?"
    )
    voice_id = config.get("agent", {}).get("voice_id", "79a125e8-cd45-4c13-8a67-188112f4dd22")

    # ── Transport (Twilio WebSocket) ──────────────
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=TwilioFrameSerializer(stream_sid=call_sid),
        ),
    )

    # ── STT — Deepgram ────────────────────────────
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        audio_passthrough=True,
    )

    # ── LangGraph processor ───────────────────────
    lg_processor = LangGraphProcessor(
        agent=agent,
        config=config,
        call_sid=call_sid
    )

    # ── TTS — Cartesia ────────────────────────────
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id=voice_id,
    )

    # ── Context (handles conversation turns) ──────
    context = OpenAILLMContext(
        messages=[{"role": "user", "content": greeting}]
    )
    context_aggregator = context.create_processor()

    # ── Build Pipeline ────────────────────────────
    pipeline = Pipeline([
        transport.input(),       # Twilio audio in
        stt,                     # Audio → text
        context_aggregator.user(), # Aggregate user turns
        lg_processor,            # LangGraph agent
        tts,                     # Text → audio
        transport.output(),      # Audio → Twilio out
        context_aggregator.assistant(),  # Track assistant turns
    ])

    task = PipelineTask(
        pipeline,
        PipelineParams(allow_interruptions=True),
    )

    # Kick off with greeting
    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        await task.queue_frames([
            LLMMessagesFrame([{"role": "user", "content": "start"}])
        ])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info(f"Call ended: {call_sid}")
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner()
    await runner.run(task)
