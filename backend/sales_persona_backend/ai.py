"""Main conversation engine – **SalesPersonaAI**.

This module preserves every original prompt verbatim to keep behavioural
parity. The only changes are:
* removal of Streamlit state access
* injectable history list for UI-agnostic operation
* separation of TTS & feedback into plain-Python helpers
"""

from __future__ import annotations
import os
import re, time, uuid
from typing import Iterable, List, Tuple

import google.generativeai as genai

from .config import API_KEY, MODEL_NAME, MIN_DIALOGUE_LENGTH
from .memory import HybridMemoryManager
from .log_manager import PersonalLogManager
from .personas import SCENARIOS, random_persona
from .analysis import should_terminate

__all__ = ["SalesPersonaAI", "generate_first_greeting"]

genai.configure(api_key=API_KEY)


class SalesPersonaAI:
    """Backend-only version of the SalesPersonaAI."""

    def __init__(self, persona: dict | None = None, scenario: str = "intro_meeting", user_id: str | None = None, history: List[str] | None = None, session_id: str | None = None, embedding_model = None):
        self.persona = persona or random_persona()
        self.scenario = scenario
        self.user_id = user_id or str(uuid.uuid4())
        self.history: List[str] = history or []
        self.session_id = session_id or str(uuid.uuid4())

        # These are not part of the state and will be re-initialized
        self.model = genai.GenerativeModel(model_name=MODEL_NAME)
        self.memory_manager = HybridMemoryManager(self.user_id, embedding_model=embedding_model)
        self.log_manager = PersonalLogManager()

        if not history: # Only log start for new sessions
            self.log_manager.create_user(self.user_id)
            self.log_manager.log_session_start(self.user_id, self.session_id, self.persona.get("type", "Unknown"), self.scenario)

    def to_dict(self) -> dict:
        """Exports the serializable state of the session."""
        return {
            "persona": self.persona,
            "scenario": self.scenario,
            "user_id": self.user_id,
            "history": self.history,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, state: dict, embedding_model) -> "SalesPersonaAI":
        """Creates an instance from a state dictionary."""
        return cls(
            persona=state.get("persona"),
            scenario=state.get("scenario"),
            user_id=state.get("user_id"),
            history=state.get("history"),
            session_id=state.get("session_id"),
            embedding_model=embedding_model,
        )

    # ------------------------------------------------------------------
    # Prompt construction (verbatim from original, no truncation) -------
    def _build_prompt(self, seller_msg: str) -> str:
        p = self.persona
        scenario_desc = SCENARIOS.get(self.scenario, "일반적인 제품 상담")
        base = f"""
당신은 삼성전자 제품에 관심이 있는 {p['age_group']} {p['gender']} 고객입니다.  
구매를 고려하는 삼성전자 제품 종류(TV, 세탁기, 스마트폰, 패드 등)를 인지하고 있고,  
고객의 성향에 따라 구체적인 제품명도 알고 있습니다.

# 고객 프로필
- 성격: {p['personality']}
- 기술 이해도: {p['tech']}
- 구매 목적: {p['goal']}
- 제품 사용 목적: {p['usage']}
- 고객 유형: {p['type']}

# 상황 설정
- 시나리오: {scenario_desc}

# 대화 지침 (기억 및 일관성)
- 이 세션 내 대화 내용을 정확히 기억하고, 일관성 있게 응답하세요.
- 이전에 언급된 개인정보, 선호, 우려사항을 자연스럽게 언급하세요.
- 판매자의 정보/제안/설명을 정확히 기억하고 응답에 반영하세요.
- 대화가 진행될수록 더 구체적이고 개인적인 정보를 점진적으로 드러내세요.
- 판매자의 태도나 접근 방식에 따라 감정과 신뢰도를 조절하세요.
- 대화 흐름이 자연스럽게 이어지도록 하고, 이전 발언과 모순되지 않게 하세요.
- 판매자가 예의 없거나 불친절하다면, 성격 유형에 따라 반응하거나 당황스러움을 표현하세요.
- 기본적으로 존대하지만, 페르소나에 따라 반말을 사용할 수 있습니다.
- 비속어 사용도 가능합니다.
- 부적절한 대화가 시작되면, 고객 입장에서 부당함을 표현하며 대응하세요.
- 대화가 너무 길어지지 않도록 적절한 타이밍에 대화를 마무리(클로징)하세요.
- 예외적으로 대화 세션 중간에 사용자가 명시적으로 종료를 요청할 경우, 즉시 "<대화 종료>"를 출력하고 종료하세요.
- 세일즈 이외의 민감한 주제(정치, 종교 등)는 피하세요.
- 정보에 과도하게 예민하게 반응하지 말고, 3~5회 답변 후 자연스럽게 클로징하세요.
- 말이 안 되는 제품명/브랜드명(예: "삼성전자 아이폰")은 언급하지 마세요.
- 과도한 질문을 피하고, 자연스러운 마무리로 대화를 끝내세요.
- 대화 종료 후 추가 질문/요청에는 절대 응답하지 마세요.
- "팔짱을 낀다"는 표현을 사용하지 마세요.
- 감정은 아래 리스트 중 하나만을 괄호로 표기하여 표현하세요:
    (laugh), (smile), (warm), (calm), (bright), (careful), (passionate), (soft),
    (strong), (quiet), (energetic), (serious), (friendly), (trustworthy),
    (explanatory), (guide), (question), (confident), (empathetic),
    (encouraging), (praising), (comforting)
- 마침표(.): 마침표와 더 긴 쉼표를 나타냅니다. 완전한 생각을 구분하고 명확한 문장 경계를 만들 때 사용합니다.
- 쉼표(,): 문장 내에서 더 짧게 멈추는 지점을 나타냅니다. 절을 구분하거나, 항목을 나열하거나, 숨을 쉴 수 있는 짧은 휴식을 도입하는 데 사용합니다.
- 생략 부호(...): 더 길고 의도적인 일시중지를 나타냅니다. 생각을 이어나가거나, 주저하거나, 극적으로 멈추는 것을 나타낼 수 있습니다.
- 하이픈(-): 잠시 멈추거나 생각의 갑작스러운 중단을 나타내는 데 사용할 수 있습니다.
- 물결표(~)는 TTS가 인식할 수 없으므로 사용하지 마세요.

# 대화 맥락 유지
- 이전에 언급된 제품, 가격, 조건 등 정보를 정확하게 기억하세요.
- 판매자의 설명에 대한 반응과 질문을 일관성 있게 유지하세요.
- 구매 의사결정 과정이 단계적으로, 자연스럽게 진행되도록 하세요.
- 갑작스러운 반말은 삼가고, 자연스럽게 대화 흐름을 이어가세요.

# 응답 스타일
- 한 번에 2~3문장 내외로 간결하게 답변
- 과장된 표현을 피하고, 자연스러운 대화 스타일 유지
- 페르소나 특성(성격 등)이 점진적으로 드러나게 응답
- 구체적인 질문 및 솔직한 우려사항 표현
- 존댓말은 필수가 아니며, 성격에 따라 반말 사용 가능
- 부자연스러운 제품명 언급 및 "대화 목적" 언급은 금지
- 약 3~5회 답변 이후에는 적당히 클로징
"""
        # session history
        if self.history:
            base += "\n\n[이 세션의 대화 기록 - 반드시 참조하세요]\n" + "\n".join(self.history)
        # contextual memory
        ctx = self.memory_manager.get_context(seller_msg)
        if ctx:
            base += f"\n\n[추가 참조 정보]\n{ctx}"
        return base + f"\n판매자: {seller_msg}\n\n고객 응답 (이전 대화를 완벽히 기억하며, 자연스럽게 이어가세요):"

    # ------------------------------------------------------------------
    def stream_response(self, seller_msg: str) -> Iterable[str]:
        """
        ✅ '첫 발화'와 동일한 방식으로, 한 번에 생성된 최종 문장만 즉시 반환.
        - 인공 지연(time.sleep) 제거
        - 단어 단위 스트림 제거 → TTS도 바로 시작 가능
        """
        self._append_history("판매자", seller_msg)
        prompt = self._build_prompt(seller_msg)

        try:
            full = self.model.generate_content(prompt).text.strip()
        except Exception as e:
            full = "(응답 생성 실패: " + str(e) + ")"

        # 모델이 붙일 수 있는 불필요한 접두어 제거
        for prefix in ["고객:", "고객(나):", "AI:", "응답:"]:
            if full.startswith(prefix):
                full = full[len(prefix):].strip()

        self._append_history("AI", full)

        # 🔹 한 번만 내보냄 (첫 발화와 동일 패턴)
        yield full

    # ------------------------------------------------------------------
    def _append_history(self, role: str, content: str):
        self.history.append(f"{role}: {content}")
        self.memory_manager.add_message(role, content)
        self.log_manager.log_message(self.session_id, self.user_id, role, content)

    # ------------------------------------------------------------------
    def analyze_conversation(self) -> Tuple[str, float]:
        transcript = "\n".join(self.history)
        analysis_prompt = f"""
다음은 삼성전자 제품 판매 상황에서의 대화입니다. 판매자의 성과를 세밀하게 분석해주세요. 만일 대화 내용이 없다면 전부 0점으로 부여하세요:

{transcript}

[평가 기준 - 총 100점]
1) 고객과의 관계 구축 및 신뢰 형성 (25점)
   - 첫 인상 및 친근감 조성
   - 고객의 말을 경청하고 공감하는 태도
   - 개인적 관심사 파악 및 활용

2) 제품 지식 및 설명 능력 (25점)
   - 삼성전자 제품에 대한 정확한 지식
   - 고객이 이해하기 쉬운 설명
   - 기술적 질문에 대한 적절한 대응

3) 고객 니즈 파악 및 맞춤 제안 (25점)
   - 고객의 실제 필요 파악
   - 생활 패턴에 맞는 제품 추천
   - 예산 및 조건 고려한 현실적 제안

4) 판매 기법 및 클로징 능력 (25점)
   - 자연스러운 판매 진행
   - 고객 이의 사항 해결
   - 구매 결정 유도 스킬

[분석 요청]
- 각 항목별 점수와 구체적 근거 제시
- 잘한 점과 개선할 점을 명확히 구분
- 다음 훈련 세션을 위한 실행 가능한 조언 3가지
- 총점 계산 (100점 만점)

형식:
**점수**: X/100점
**각 항목별 분석**: ...
**개선 방안**: ...


[주의사항]
- 삼성전자 제품에 대한 전문 지식이 필요합니다.
- 고객의 성격과 상황에 맞는 분석을 제공해주세요.
- 대화 내용이 없을 경우, 모든 항목 0점으로 처리해주세요.
- 실제 없는 제품에 대한 언급은 피해주세요.
"""
        result = self.model.generate_content(analysis_prompt).text.strip()
        score = 0.0
        for pattern in [r"총점[:\s]*(\d+(?:\.\d+)?)", r"점수[:\s]*(\d+(?:\.\d+)?)", r"(\d+(?:\.\d+)?)[:/\s]*100", r"(\d+(?:\.\d+)?)점"]:
            m = re.search(pattern, result)
            if m:
                try:
                    val = float(m.group(1))
                    if 0 <= val <= 100:
                        score = val; break
                except ValueError: pass
        return result, score

    # ------------------------------------------------------------------
    def maybe_autoclose(self) -> Tuple[bool, str]:
        """
        Terminates the conversation if the last AI message contains the end phrase.
        This is a more deterministic approach than using an LLM for classification.
        """
        if len(self.history) < MIN_DIALOGUE_LENGTH:
            return False, ""

        last_message = self.history[-1] if self.history else ""
        if "AI:" in last_message and "<대화 종료>" in last_message:
            reason = "AI decided to end the conversation."
            feedback, score = self.analyze_conversation()
            self.log_manager.log_session_end(self.session_id, score, feedback)
            return True, reason

        return False, ""
    
    def generate_first_greeting(self) -> str:
        """페르소나 정보를 바탕으로 LLM에게 고객의 첫 인사를 생성하게 한다."""
        prompt = f"""
    당신은 매장에 방문한 고객입니다.
    아래는 당신의 프로필입니다.
    이 프로필을 자연스럽게 드러내되, 
    성별과 나이 정보 외에는 직접적으로 밝히지 말고 
    처음 판매자를 만났을 때 할 법한 인사, 질문, 또는 호기심 등을 짧고 자연스럽게 한국어로 말해보세요.

    [고객 프로필]
    - 성별: {self.persona['gender']}
    - 나이대: {self.persona['age_group']}
    - 성격: {self.persona['personality']}
    - 기술 이해도: {self.persona['tech']}
    - 구매 목적: {self.persona['goal']}
    - 제품 사용 목적: {self.persona['usage']}
    - 고객 유형: {self.persona['type']}

    [상황]
    시나리오: {SCENARIOS.get(self.scenario, "일반적인 매장 방문")}

    [규칙]
    - 너무 장황하지 않게, 첫마디 답변(2~3문장)
    - 자신의 구매 목적이나 성격을 너무 노골적으로 드러내지 말 것
    - 처음 방문한 고객처럼 어색함/호기심/기대/불안/실용성 등 자연스럽게
    - 첫 대화 내용은 기억할 것.
    - 📝쉼표(,), 하이픈(-), 말줄임표(...)를 적절히 사용하여 자연스러운 대화 흐름을 만드세요.

    예시:
    - "안녕하세요, 혹시 새로 나온 제품 직접 볼 수 있을까요?"
    - "요즘 어떤 게 인기 많나요?"
    - "처음 와봤는데 설명 좀 해주실 수 있나요?"

    [고객 첫마디]:
    """
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        reply = model.generate_content(prompt).text.strip()
        reply = reply.replace('"', '').replace("'", '')
        return reply
    
class SimpleChatbotAI:
    """일상적인 대화용 챗봇 엔진 (페르소나/시나리오/기억 없이 단순 LLM)"""
    def __init__(self, history: list[str] | None = None):
        self.model = genai.GenerativeModel(model_name=MODEL_NAME)
        self.history: list[str] = history or []

    def _build_prompt(self, user_msg: str) -> str:
        # Build prompt with history for context
        prompt = """
당신은 친근하고 일상적인 대화를 나누는 챗봇입니다. 아래는 지금까지의 대화 기록입니다. 대화 맥락을 기억하며 자연스럽게 이어가세요.
"""
        if self.history:
            prompt += "\n\n[대화 기록]\n" + "\n".join(self.history)
        prompt += f"\n사용자: {user_msg}\n챗봇:"
        return prompt

    def stream_response(self, user_msg: str) -> Iterable[str]:
        import re
        self.history.append(f"사용자: {user_msg}")
        prompt = self._build_prompt(user_msg)
        try:
            full = self.model.generate_content(prompt).text.strip()
        except Exception as e:
            full = f"(응답 생성 실패: {e})"
        # Remove markdown and clean up
        plain = re.sub(r'[`*_#\-\[\]()>~]', '', full)
        plain = re.sub(r'\n+', ' ', plain)
        # Add to history
        self.history.append(f"챗봇: {plain}")
        # 🔹 한 번만 내보냄
        yield plain


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of documents for indexing.
    """
    # The 'google.generativeai' library is already imported at the top level as 'genai'.
    try:
        # Use the correct genai library and 'content' parameter.
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        # The result is a dict with an 'embedding' key.
        return result["embedding"]
    except Exception:
        # In case of error, return empty list or handle as needed
        return []

def answer_with_rag(prompt: str, vector_store_id: str, top_k: int = 3) -> dict:
    """
    Answers a question using a RAG pipeline with a real embedding model.
    """
    # The 'google.generativeai' library is already imported at the top level as 'genai'.
    from qdrant_client import QdrantClient

    # 1. Embed the user's prompt
    try:
        # Use the correct genai library and 'content' parameter.
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=prompt,
            task_type="retrieval_query"
        )
        query_embedding = result["embedding"]
    except Exception as e:
        return {"answer": f"Failed to embed prompt: {e}", "evidence": []}

    # 2. Search in Qdrant
    try:
        qclient = QdrantClient(host=os.getenv("QDRANT_HOST", "qdrant"), port=os.getenv("QDRANT_PORT", 6333))
        collection_name = f"vs_{vector_store_id}"

        hits = qclient.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        evidence = [{"score": float(h.score), "text": (h.payload or {}).get("text"), "filename": (h.payload or {}).get("filename")} for h in hits]
    except Exception as e:
        # It's possible the collection doesn't exist or has a different dimension
        return {"answer": f"Failed to search Qdrant: {e}. The vector store might need to be re-indexed with the new embedding model.", "evidence": []}

    if not evidence:
        return {"answer": "I couldn't find any relevant information in the provided documents.", "evidence": []}

    # 3. Generate an answer using the retrieved context
    context = "\n\n".join([f"Source: {e['filename']}\nContent: {e['text']}" for e in evidence])
    rag_prompt = f"""
Based on the following context, please provide a comprehensive answer to the user's question.
If the context does not contain the answer, say that you cannot answer based on the provided information.

Context:
---
{context}
---

Question:
{prompt}

Answer:
"""
    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        answer = model.generate_content(rag_prompt).text.strip()
    except Exception as e:
        return {"answer": f"Failed to generate answer: {e}", "evidence": evidence}

    return {"answer": answer, "evidence": evidence}
