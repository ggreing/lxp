"""Helper to let LLM decide whether to automatically end a session."""

import google.generativeai as genai
from .config import API_KEY, MODEL_NAME

genai.configure(api_key=API_KEY)

__all__ = ["should_terminate"]


def should_terminate(conversation: str):
    """Return (bool terminate, str label) from LLM judgment."""
    prompt = f"""대화 로그를 보고, (1) 클로징 성공, (2) 클로징 실패, (3) 주제이탈, (4) 진행중 중 하나를 선택해 주세요. 반드시 label 만 출력: success / fail / off-topic / continue

### 대화 ###
{conversation}
"""
    try:
        label = genai.GenerativeModel(model_name=MODEL_NAME).generate_content(prompt).text.strip().lower()
    except Exception:
        return False, "continue"
    return label in {"success", "fail", "off-topic"}, label