"""LLM settings. Change the model in .env, not in code."""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()  # reads .env in the project root (if it exists)

MODEL = os.getenv("LLM_MODEL", "gpt-oss:20b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
NUM_CTX = int(os.getenv("LLM_NUM_CTX", "16384"))      # Ollama's default is too small for tool results
REASONING = os.getenv("LLM_REASONING", "medium")      # gpt-oss: low | medium | high


def get_llm(reasoning: str | bool | None = REASONING) -> ChatOllama:
    """Return the chat model used by every node.

    reasoning: how hard the model "thinks" before answering.
      gpt-oss accepts "low" / "medium" / "high"; other models (qwen3) take True / False.
    """
    return ChatOllama(
        model=MODEL,
        base_url=OLLAMA_URL,
        temperature=0.1,      # low = consistent, factual answers
        num_ctx=NUM_CTX,
        reasoning=reasoning,
        keep_alive="10m",     # keep the model loaded in memory between requests
    )
