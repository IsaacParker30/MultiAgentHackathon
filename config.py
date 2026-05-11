import os
from dotenv import load_dotenv
from autogen import LLMConfig

load_dotenv()


def get_llm_config() -> LLMConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and fill in your key.")
    config = {
        "model": os.getenv("MODEL_NAME", "gpt-4o-mini"),
        "api_key": api_key,
    }
    base_url = os.getenv("OPENAI_API_BASE")
    if base_url:
        config["base_url"] = base_url
    return LLMConfig(config, temperature=0.3)
