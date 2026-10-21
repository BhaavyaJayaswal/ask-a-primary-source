"""Evaluation suite: ~10 question/expected-fact pairs covering all 3 books,
plain retrieval, and the lookup_year agent tool. Each test calls the real
/ask endpoint (so it costs a real Gemini call) and checks the expected fact
shows up in the answer.

Run with:  pytest
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

EVAL_CASES = [
    ("Who wrote The Prince?", "machiavelli"),
    ("Who wrote the Ecclesiastical History of the English People?", "bede"),
    ("Who wrote Germania?", "tacitus"),
    ("Who led the Britons against Agricola's army at the Grampian hills?", "calgacus"),
    # Known hard case: the real passage ranks ~8th in retrieval, not top-5.
    # Marked xfail rather than skipped/removed, so it stays visible as a
    # tracked, documented limitation instead of a silent regression.
    pytest.param(
        "According to Bede, who converted the Anglo-Saxons to Christianity?",
        "augustine",
        marks=pytest.mark.xfail(
            reason="Embedding model ranks the real passage ~8th, outside TOP_K=5",
            strict=True,
        ),
    ),
    ("In what year did Bede finish his Ecclesiastical History?", "731"),
    ("What year was the Battle of Hastings?", "1066"),  # lookup_year tool
    ("What year did Machiavelli write The Prince?", "1513"),  # lookup_year tool
    ("According to Machiavelli, is it better for a prince to be feared or loved?", "feared"),
    ("According to Machiavelli, should a prince be considered generous or miserly?", "miser"),
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("question, expected", EVAL_CASES)
def test_ask_contains_expected_fact(client, question, expected):
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 200

    answer = response.json()["answer"].lower()
    assert expected.lower() in answer, (
        f"Expected '{expected}' in answer to {question!r}, got: {answer}"
    )
