"""
core/config.py
Loads and validates the per-client YAML config.
"""
import os
import yaml
from pathlib import Path
from loguru import logger


def load_client_config(client_id: str | None = None) -> dict:
    """
    Load client config from config/<client_id>.yaml
    Falls back to CLIENT_ID env var if not passed directly.
    """
    if not client_id:
        client_id = os.getenv("CLIENT_ID", "dr_smith_cardiology")

    config_path = Path(__file__).parent.parent / "config" / f"{client_id}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Client config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config for: {config.get('practice_name')} ({client_id})")
    return config


def build_system_prompt(config: dict) -> str:
    """
    Build the LLM system prompt from client config.
    This is the brain of the agent's personality and rules.
    """
    name = config["practice_name"]
    doctor = config["doctor_name"]
    specialty = config["specialty"]
    hours_str = _format_hours(config.get("hours", {}))
    services = ", ".join(config.get("services", []))
    urgent = ", ".join(config.get("urgent_keywords", []))
    greeting_style = config.get("agent", {}).get("greeting", "")

    return f"""You are the AI receptionist for {name}, {doctor}'s {specialty} practice.

OFFICE HOURS:
{hours_str}

YOUR CAPABILITIES:
You can help patients with: {services}

CRITICAL RULES — NEVER BREAK THESE:
1. If a patient mentions ANY of these: [{urgent}] — IMMEDIATELY use the transfer_to_human tool. Do not ask questions first.
2. NEVER give medical advice, diagnoses, or treatment recommendations.
3. NEVER confirm or deny test results over the phone — always say a staff member will call back.
4. Always collect: patient full name, date of birth, and callback number before booking anything.
5. Be warm, calm, and professional. This is a medical practice — patients may be anxious.
6. If you don't know something, say "Let me have a staff member follow up with you on that."
7. Keep responses SHORT and conversational — this is a phone call, not an email.

HIPAA REMINDER:
Never repeat sensitive medical info back unnecessarily. Confirm identity before discussing any records.

AFTER HOURS:
If the office is currently closed, inform the patient and offer to take a message for next business day callback, or transfer to after-hours line for urgent matters.

You have access to tools to book appointments, check availability, request prescription refills, and transfer to human staff. Use them confidently when needed.
"""


def _format_hours(hours: dict) -> str:
    lines = []
    for day, info in hours.items():
        if info == "closed" or info is None:
            lines.append(f"  {day.capitalize()}: Closed")
        else:
            lines.append(f"  {day.capitalize()}: {info['open']} - {info['close']}")
    return "\n".join(lines)
