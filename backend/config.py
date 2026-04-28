"""
Configuration — loads API keys and model settings from environment.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Per-Agent Gemini Keys (Primary) ───────────────────────
GEMINI_KEYS = [
    os.getenv("GEMINI_KEY_1", ""),
    os.getenv("GEMINI_KEY_2", ""),
    os.getenv("GEMINI_KEY_3", ""),
    os.getenv("GEMINI_KEY_4", ""),
    os.getenv("GEMINI_KEY_5", ""),
]

# ── Per-Agent OpenAI Keys (Fallback) ─────────────────────
OPENAI_KEYS = [
    os.getenv("OPENAI_KEY_1", ""),
    os.getenv("OPENAI_KEY_2", ""),
    os.getenv("OPENAI_KEY_3", ""),
    os.getenv("OPENAI_KEY_4", ""),
    os.getenv("OPENAI_KEY_5", ""),
]

# ── Model Config ──────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Models available in the UI dropdown
AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

DEFAULT_MODEL = GEMINI_MODEL

# ── Server Config ─────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ── Debate Config ─────────────────────────────────────────
MAX_DEBATE_ROUNDS = 2
MAX_AGENTS = 6
