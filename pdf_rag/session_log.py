import json
from datetime import datetime
from pathlib import Path


class SessionLog:
    def __init__(self, sessions_dir: Path) -> None:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = sessions_dir / f"{ts}.jsonl"

    def append(self, mode: str, question: str, answer: str) -> None:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "question": question,
            "answer": answer,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
