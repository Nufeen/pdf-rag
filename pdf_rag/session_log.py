import json
from datetime import datetime
from pathlib import Path


class SessionLog:
    def __init__(self, sessions_dir: Path) -> None:
        sessions_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = sessions_dir / f"{ts}.jsonl"

    @staticmethod
    def load_latest(sessions_dir: Path) -> list[dict]:
        files = sorted(sessions_dir.glob("*.jsonl"))
        if not files:
            return []
        entries = []
        with files[-1].open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def append(self, mode: str, question: str, answer: str) -> None:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "question": question,
            "answer": answer,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
