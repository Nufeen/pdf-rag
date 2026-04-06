import json

from pdf_rag.session_log import SessionLog


def test_append_creates_file(tmp_path):
    log = SessionLog(tmp_path / "sessions")
    log.append("ask", "What is entropy?", "It is disorder.")
    assert log.path.exists()


def test_append_writes_valid_jsonl(tmp_path):
    log = SessionLog(tmp_path / "sessions")
    log.append("ask", "What is entropy?", "It is disorder.")
    lines = log.path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["mode"] == "ask"
    assert entry["question"] == "What is entropy?"
    assert entry["answer"] == "It is disorder."
    assert "ts" in entry


def test_multiple_appends_stored_as_separate_lines(tmp_path):
    log = SessionLog(tmp_path / "sessions")
    log.append("ask", "Q1", "A1")
    log.append("research", "Q2", "A2")
    lines = log.path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_load_latest_returns_appended_entries(tmp_path):
    sessions = tmp_path / "sessions"
    log = SessionLog(sessions)
    log.append("ask", "What is entropy?", "It is disorder.")
    log.append("research", "What is information?", "Shannon's theory.")
    result = SessionLog.load_latest(sessions)
    assert len(result) == 2
    assert result[0]["question"] == "What is entropy?"
    assert result[1]["question"] == "What is information?"


def test_load_latest_empty_dir_returns_empty_list(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    assert SessionLog.load_latest(sessions) == []


def test_load_latest_corrupted_line_skipped(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    f = sessions / "2024-01-01_00-00-00.jsonl"
    f.write_text('{"mode":"ask","question":"Q","answer":"A","ts":"x"}\nNOT_JSON\n')
    result = SessionLog.load_latest(sessions)
    assert len(result) == 1
    assert result[0]["question"] == "Q"


def test_load_latest_picks_most_recent_file(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "2024-01-01_00-00-00.jsonl").write_text(
        '{"mode":"ask","question":"old","answer":"a","ts":"x"}\n'
    )
    (sessions / "2024-01-02_00-00-00.jsonl").write_text(
        '{"mode":"ask","question":"new","answer":"b","ts":"y"}\n'
    )
    result = SessionLog.load_latest(sessions)
    assert len(result) == 1
    assert result[0]["question"] == "new"


def test_init_creates_sessions_directory(tmp_path):
    sessions = tmp_path / "deep" / "nested" / "sessions"
    assert not sessions.exists()
    SessionLog(sessions)
    assert sessions.exists()
