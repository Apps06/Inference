"""
Configuration — loads API keys and model settings from environment.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── Model Config ──────────────────────────────────────────
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "grok-4.20-reasoning")

XAI_BASE_URL = "https://api.x.ai/v1"

GROK_MODELS = [
    "grok-4.20-reasoning",
    "grok-4.20-non-reasoning",
    "grok-4.1-fast-reasoning",
    "grok-4.1-fast-non-reasoning",
]

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
]

# ── Server Config ─────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ── Debate Config ─────────────────────────────────────────
MAX_DEBATE_ROUNDS = 2
MAX_AGENTS = 6
