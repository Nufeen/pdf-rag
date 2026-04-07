from dataclasses import dataclass

from ollama import Client

from .config import FAST_MODEL, OLLAMA_BASE_URL
from .llm import load_prompt


@dataclass
class SessionContext:
    summary: str = ""
    turn_count: int = 0

    def update(
        self,
        question: str,
        answer: str,
        client: Client | None = None,
        base_url: str = OLLAMA_BASE_URL,
        model: str = FAST_MODEL,
    ) -> str:
        if client is None:
            client = Client(host=base_url)

        prompt = load_prompt(
            "update_session_summary",
            prev_summary=self.summary or "(none)",
            question=question,
            answer=answer,
        )
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        self.summary = response["message"]["content"].strip()
        self.turn_count += 1
        return self.summary

    def enrich_context(self, chunks_context: str) -> str:
        if not self.summary:
            return chunks_context
        return f"Session context: {self.summary}\n\n{chunks_context}"

    def reset(self) -> None:
        self.summary = ""
        self.turn_count = 0
