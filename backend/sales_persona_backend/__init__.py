"""Sales Persona AI – Streamlit‑less backend package.

**2025‑07‑14 Update**
• API 키와 기타 민감 정보는 더 이상 하드코딩하지 않고 `.env` 파일에서 로드합니다. 
• 기존 기능·프롬프트는 그대로이며, import 경로/인터페이스도 불변입니다.
"""

from importlib import metadata
__version__ = "2.4.0"  # bumped because of env migration