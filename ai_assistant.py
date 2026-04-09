import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:  # pragma: no cover - handled at runtime
    Groq = None


load_dotenv()


LOCAL_BREED_KNOWLEDGE: Dict[str, Dict[str, str]] = {
    "default": {
        "description": "This is a domesticated bovine breed used in livestock farming. Verify important management details with local veterinary or agricultural experts.",
        "purpose": "Breeds may be selected for milk, meat, draught work, or mixed farming purposes depending on regional practice.",
        "care": "Provide clean water, balanced feed, shade, vaccination, parasite control, and regular veterinary monitoring.",
    },
    "gir": {
        "description": "Gir is a well-known Indian zebu cattle breed recognized for its domed forehead, long ears, and strong heat tolerance.",
        "purpose": "Mainly valued for dairy production, though it is also important in breeding programs because of adaptability.",
        "care": "Do well with good-quality green fodder, mineral mixture, heat stress management, and attention to udder hygiene.",
    },
    "sahiwal": {
        "description": "Sahiwal is a high-yield indigenous dairy breed known for calm temperament and good performance in hot climates.",
        "purpose": "Primarily a milk breed with strong importance in dairy farming.",
        "care": "Needs energy-rich feed during lactation, clean housing, mineral supplementation, and routine mastitis prevention.",
    },
    "murrah": {
        "description": "Murrah is a major buffalo breed known for high milk yield, compact body, and strong value in commercial dairying.",
        "purpose": "Primarily used for milk production, especially high-fat buffalo milk.",
        "care": "Benefits from cooling in hot weather, quality roughage plus concentrate feed, and careful reproductive health management.",
    },
}


class BreedAIAssistant:
    """Handles Groq-based breed explanations and chat using detected breed context."""

    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = self._build_client()

    def _build_client(self):
        if not self.api_key or Groq is None:
            return None
        return Groq(api_key=self.api_key)

    def _breed_facts(self, breed_name: str) -> Dict[str, str]:
        return LOCAL_BREED_KNOWLEDGE.get(breed_name.strip().lower(), LOCAL_BREED_KNOWLEDGE["default"])

    def _system_prompt(self, breed_name: str, confidence: Optional[float] = None) -> str:
        confidence_text = (
            f"The current predicted breed is {breed_name} with confidence {confidence:.2%}. "
            if confidence is not None
            else f"The current predicted breed is {breed_name}. "
        )
        return (
            "You are a helpful livestock assistant for a cattle and buffalo breed classifier app. "
            + confidence_text
            + "Use the detected breed as the main context in every answer. "
            "Give practical, beginner-friendly guidance about breed description, purpose, feeding, care, and health. "
            "Avoid claiming certainty for medical diagnosis and advise consulting a veterinarian for urgent or serious symptoms. "
            "Keep answers concise, structured, and easy to understand."
        )

    def _call_llm(self, user_prompt: str, breed_name: str, confidence: Optional[float] = None) -> str:
        if self.client is None:
            raise RuntimeError("Groq client is not configured.")

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": self._system_prompt(breed_name, confidence)},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def generate_breed_report(self, breed_name: str, confidence: float) -> Dict[str, str]:
        facts = self._breed_facts(breed_name)

        if self.client is None:
            return {
                "description": facts["description"],
                "purpose": facts["purpose"],
                "care": facts["care"],
                "source_note": (
                    "Using built-in breed guidance because `GROQ_API_KEY` is not configured "
                    "or the `groq` package is not installed."
                ),
            }

        prompt = (
            f"Breed: {breed_name}\n"
            f"Confidence: {confidence:.2%}\n\n"
            "Return valid JSON with exactly these keys:\n"
            '- "description"\n'
            '- "purpose"\n'
            '- "care"\n\n'
            "Each value should be a short paragraph focused on the detected breed."
        )

        try:
            content = self._call_llm(user_prompt=prompt, breed_name=breed_name, confidence=confidence)
            parsed = json.loads(content)
            return {
                "description": parsed.get("description", facts["description"]),
                "purpose": parsed.get("purpose", facts["purpose"]),
                "care": parsed.get("care", facts["care"]),
                "source_note": f"Generated by {self.model}.",
            }
        except Exception as exc:
            return {
                "description": facts["description"],
                "purpose": facts["purpose"],
                "care": facts["care"],
                "source_note": f"Groq generation failed ({exc}). Showing built-in breed guidance instead.",
            }

    def _history_to_text(self, history: Optional[List[Dict[str, str]]]) -> str:
        if not history:
            return "No previous conversation."

        lines: List[str] = []
        for item in history:
            role = item.get("role")
            content = self._extract_text(item.get("content"))
            if not content or role not in {"user", "assistant"}:
                continue
            speaker = "User" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines) if lines else "No previous conversation."

    def _extract_text(self, content) -> str:
        """Converts Gradio chat content objects into plain text."""
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            if "text" in content:
                return str(content.get("text", ""))
            if "content" in content:
                return self._extract_text(content.get("content"))
            return json.dumps(content, ensure_ascii=True)

        if isinstance(content, list):
            parts = [self._extract_text(part) for part in content]
            return "\n".join(part for part in parts if part)

        return str(content)

    def chat(self, user_message: str, breed_name: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        facts = self._breed_facts(breed_name)

        if not breed_name:
            return "Please upload an image first so I can answer using the detected breed."

        if self.client is None:
            return (
                f"Current breed context: {breed_name}.\n\n"
                f"Description: {facts['description']}\n"
                f"Purpose: {facts['purpose']}\n"
                f"Care: {facts['care']}\n\n"
                "For richer answers, set `GROQ_API_KEY` and install the `groq` package."
            )

        prompt = (
            f"Current detected breed: {breed_name}\n\n"
            "Conversation so far:\n"
            f"{self._history_to_text(history)}\n\n"
            "Answer the user's latest question using the breed context above.\n"
            f"Latest user question: {user_message}"
        )

        try:
            return self._call_llm(user_prompt=prompt, breed_name=breed_name)
        except Exception as exc:
            return (
                f"I could not reach Groq right now ({exc}), but I can still help with the current breed context: {breed_name}.\n\n"
                f"{facts['care']}"
            )
