"""
core/agent.py
LangGraph agent — the brain of the voice agent.
Handles intent, tool calling, and action execution.
"""
import os
import aiohttp
from loguru import logger
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_name: str | None
    patient_dob: str | None
    callback_number: str | None
    transfer_requested: bool


# ─────────────────────────────────────────────
# TOOLS
# Each tool maps to a real action in the clinic
# ─────────────────────────────────────────────

def build_tools(config: dict):
    """Build tools dynamically from client config."""
    webhooks = config.get("webhooks", {})

    @tool
    async def check_appointment_availability(date: str, appointment_type: str) -> str:
        """Check available appointment slots for a given date and type.
        Use this before booking to show the patient their options."""
        url = webhooks.get("check_availability")
        if not url:
            return "Appointment system unavailable. I'll have staff call you back to schedule."
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"date": date, "type": appointment_type}) as r:
                    data = await r.json()
                    slots = data.get("slots", [])
                    if not slots:
                        return f"No availability on {date}. Next available slots are: {data.get('next_available', 'unknown')}."
                    return f"Available times on {date}: {', '.join(slots)}"
        except Exception as e:
            logger.error(f"check_availability error: {e}")
            return "I'm having trouble accessing the schedule right now. A staff member will call to confirm."

    @tool
    async def book_appointment(
        patient_name: str,
        patient_dob: str,
        callback_number: str,
        appointment_date: str,
        appointment_time: str,
        reason: str
    ) -> str:
        """Book an appointment for a patient.
        Always collect name, DOB, callback number, preferred date/time, and reason before calling this."""
        url = webhooks.get("book_appointment")
        if not url:
            return "I've noted your request. A staff member will call you back to confirm the appointment."

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "patient_name": patient_name,
                    "patient_dob": patient_dob,
                    "callback_number": callback_number,
                    "date": appointment_date,
                    "time": appointment_time,
                    "reason": reason,
                }
                async with session.post(url, json=payload) as r:
                    data = await r.json()
                    conf_number = data.get("confirmation_number", "pending")
                    return f"Appointment booked for {appointment_date} at {appointment_time}. Confirmation number: {conf_number}. You'll receive a reminder call 24 hours before."
        except Exception as e:
            logger.error(f"book_appointment error: {e}")
            return "I've recorded your appointment request. Staff will call to confirm within 1 business day."

    @tool
    async def request_prescription_refill(
        patient_name: str,
        patient_dob: str,
        callback_number: str,
        medication_name: str,
        pharmacy_name: str | None = None
    ) -> str:
        """Submit a prescription refill request.
        Collect patient details and medication name before calling this."""
        url = webhooks.get("refill_request")
        if not url:
            return "I've noted your refill request. Please allow 24-48 business hours for processing."

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "patient_name": patient_name,
                    "patient_dob": patient_dob,
                    "callback_number": callback_number,
                    "medication": medication_name,
                    "pharmacy": pharmacy_name,
                }
                async with session.post(url, json=payload) as r:
                    data = await r.json()
                    return f"Refill request submitted for {medication_name}. Please allow 24-48 business hours. The pharmacy will be notified directly."
        except Exception as e:
            logger.error(f"refill_request error: {e}")
            return "Refill request recorded. Allow 24-48 hours for processing."

    @tool
    async def take_message(
        patient_name: str,
        callback_number: str,
        message: str,
        urgency: str = "routine"
    ) -> str:
        """Take a message for the doctor or staff to follow up.
        Use when the patient's need doesn't fit other categories or it's after hours."""
        logger.info(f"Message taken for {patient_name}: {message} | urgency={urgency}")
        # In production: save to DB, send to staff via SMS/email
        return f"Message recorded for {patient_name}. Staff will call {callback_number} back within 1 business day. For urgent matters, please call 911 or go to the nearest emergency room."

    @tool
    async def transfer_to_human(reason: str) -> str:
        """Transfer the call to a human staff member immediately.
        ALWAYS use this for any urgent symptoms, emergencies, or when patient explicitly asks for a human."""
        # Pipecat handles the actual Twilio transfer — this just signals intent
        logger.info(f"Transfer requested: {reason}")
        return "TRANSFER_NOW"  # bot.py watches for this signal

    return [
        check_appointment_availability,
        book_appointment,
        request_prescription_refill,
        take_message,
        transfer_to_human,
    ]


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────

def build_agent(system_prompt: str, config: dict):
    """
    Build and compile the LangGraph agent.
    Returns a compiled graph ready to stream responses.
    """
    tools = build_tools(config)
    llm = ChatAnthropic(
        model="claude-3-5-haiku-20241022",  # fast + cheap for real-time voice
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=300,  # keep responses short — it's a phone call
    ).bind_tools(tools)

    tool_node = ToolNode(tools)

    def agent_node(state: AgentState):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    memory = MemorySaver()  # per-call memory
    return graph.compile(checkpointer=memory)
