import json

import pytest

import ollama_service
from app import app


@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def _evaluation_response():
    return {
        "dimensionScores": {
            "technicalAccuracy": 60,
            "conceptualUnderstanding": 70,
            "completeness": 80,
            "practicalExamples": 50,
            "clarity": 90,
            "reasoning": 70,
        },
        "assessment": "partially_correct",
        "conceptReview": [
            {"concept": "Cache invalidation", "status": "covered", "evidence": "cache invalidation"},
            {"concept": "Consistency", "status": "missing", "evidence": ""},
        ],
        "correctConcepts": ["Cache invalidation"],
        "missingConcepts": ["Consistency"],
        "incorrectConcepts": [],
        "explanation": "The answer describes invalidation but does not explain the consistency guarantee.",
        "feedback": "Explain how cache invalidation keeps reads consistent with the source.",
        "improvedAnswer": "I would cache frequently read values and invalidate them when the source changes.",
        "actionableFeedback": ["Explain the invalidation strategy.", "Give an example of a stale read."],
    }





def test_health_reports_actual_ollama_model_status(client, monkeypatch):
    monkeypatch.setattr(
        ollama_service,
        "_request_json",
        lambda path, timeout_seconds=None: {"models": [{"name": "qwen2.5:3b"}]},
    )

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json["connected"] is True
    assert response.json["modelAvailable"] is True
    assert response.json["model"] == "qwen2.5:3b"


def test_health_does_not_claim_ollama_is_connected_when_unavailable(monkeypatch):
    def offline(path, timeout_seconds=None):
        raise ollama_service.OllamaError("Cannot connect to Ollama.", 503)

    monkeypatch.setattr(ollama_service, "_request_json", offline)

    status = ollama_service.connection_status()

    assert status["connected"] is False
    assert status["modelAvailable"] is False
    assert status["status"] == "offline"


def test_question_generation_uses_model_result(client, monkeypatch):
    monkeypatch.setattr(
        "app.generate_question",
        lambda context: {
            "question": "How would you avoid stale data in a distributed cache?",
            "expectedConcepts": ["invalidation", "consistency", "TTL"],
        },
    )

    response = client.post("/api/interview", json={
        "action": "start",
        "role": "Backend Developer",
        "interviewType": "Technical",
        "difficulty": "Hard",
    })

    assert response.status_code == 200
    assert response.json["question"].startswith("How would you")
    assert len(response.json["expectedConcepts"]) == 3
    assert response.json["mode"] == "ollama"


def test_question_generation_uses_an_alternative_when_first_candidate_repeats(monkeypatch):
    prompts = []
    responses = [{
        "questions": [
            {"question": "How would you design a rate limiter?", "expectedConcepts": ["limits", "tokens", "fairness"]},
            {"question": "How would you investigate a slow database query?", "expectedConcepts": ["indexes", "plans", "metrics"]},
            {"question": "How would you monitor a service outage?", "expectedConcepts": ["alerts", "metrics", "recovery"]},
        ],
    }]

    def fake_request(path, payload):
        prompts.append(payload["messages"][-1]["content"])
        return {"message": {"content": json.dumps(responses[0])}}

    monkeypatch.setattr(ollama_service, "_request_json", fake_request)
    monkeypatch.setattr(ollama_service.time, "sleep", lambda _: None)

    result = ollama_service.generate_question({
        "role": "Backend Developer",
        "questionNumber": 1,
        "previousQuestions": ["How would you design a rate limiter?"],
    })

    assert result["question"] == "How would you investigate a slow database query?"
    assert len(prompts) == 1
    assert "questions" in ollama_service._question_schema()["properties"]


def test_question_generation_retries_when_all_candidates_repeat(monkeypatch):
    prompts = []
    repeated = {
        "question": "How would you design a rate limiter?",
        "expectedConcepts": ["limits", "tokens", "fairness"],
    }
    new_question = {
        "question": "How would you investigate a slow database query?",
        "expectedConcepts": ["indexes", "plans", "metrics"],
    }

    def fake_request(path, payload):
        prompts.append(payload["messages"][-1]["content"])
        questions = [repeated, repeated, repeated] if len(prompts) == 1 else [
            new_question,
            {"question": "How would you monitor a service outage?", "expectedConcepts": ["alerts", "metrics", "recovery"]},
            {"question": "How would you handle database migrations?", "expectedConcepts": ["rollbacks", "testing", "locking"]},
        ]
        return {"message": {"content": json.dumps({"questions": questions})}}

    monkeypatch.setattr(ollama_service, "_request_json", fake_request)
    monkeypatch.setattr(ollama_service.time, "sleep", lambda _: None)

    result = ollama_service.generate_question({
        "role": "Backend Developer",
        "questionNumber": 1,
        "previousQuestions": ["How would you design a rate limiter?"],
    })

    assert result["question"] == new_question["question"]
    assert len(prompts) == 2
    assert "different skill area" in prompts[1]


def test_evaluation_scores_are_valid_and_overall_is_dimension_mean(monkeypatch):
    requests = []

    def fake_request(path, payload):
        requests.append(payload)
        return {"message": {"content": json.dumps(_evaluation_response())}}

    monkeypatch.setattr(ollama_service, "_request_json", fake_request)
    result = ollama_service.evaluate_answer({
        "question": "How do you keep cached data accurate?",
        "expectedConcepts": ["Cache invalidation", "Consistency"],
        "answer": "I use cache invalidation when source data changes.",
    })

    scores = result["dimensionScores"]
    assert all(0 <= score <= 100 for score in scores.values())
    assert result["overallScore"] == round(sum(scores.values()) / len(scores))
    assert result["correctConcepts"] == ["Cache invalidation"]
    assert result["missingConcepts"] == ["Consistency"]
    assert result["assessment"] == "partially_correct"
    schema = requests[0]["format"]
    assert schema["properties"]["dimensionScores"]["properties"]["technicalAccuracy"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 100,
    }
    assert schema["properties"]["conceptReview"]["minItems"] == 2
    assert requests[0]["options"]["num_predict"] > 0


def test_invalid_model_json_is_retried_and_then_rejected(monkeypatch):
    attempts = []

    def fake_request(path, payload):
        attempts.append(path)
        return {"message": {"content": "not valid json"}}

    monkeypatch.setattr(ollama_service, "_request_json", fake_request)
    monkeypatch.setattr(ollama_service.time, "sleep", lambda _: None)

    with pytest.raises(ollama_service.OllamaError, match="invalid response twice"):
        ollama_service.evaluate_answer({
            "question": "How do caches stay consistent?",
            "expectedConcepts": ["Cache invalidation", "Consistency"],
            "answer": "I invalidate a cache when its source changes.",
        })

    assert len(attempts) == 2


def test_final_summary_uses_bounded_json_schema(monkeypatch):
    payloads = []
    summary = {
        "headline": "A useful interview practice session.",
        "strengths": ["Clear communication and relevant examples."],
        "weaknesses": ["Add more detail about performance trade-offs."],
        "recommendations": ["Practice explaining implementation decisions."],
    }

    def fake_request(path, payload):
        payloads.append(payload)
        return {"message": {"content": json.dumps(summary)}}

    monkeypatch.setattr(ollama_service, "_request_json", fake_request)

    result = ollama_service.generate_summary([{
        "question": "How do you keep cached data accurate?",
        "evaluation": _evaluation_response(),
    }])

    assert result == summary
    assert payloads[0]["format"]["required"] == [
        "headline",
        "strengths",
        "weaknesses",
        "recommendations",
    ]


def test_schema_retry_asks_model_to_return_omitted_fields(monkeypatch):
    prompts = []
    incomplete = _evaluation_response()
    incomplete.pop("explanation")

    def fake_request(path, payload):
        prompts.append(payload["messages"][-1]["content"])
        response = incomplete if len(prompts) == 1 else _evaluation_response()
        return {"message": {"content": json.dumps(response)}}

    monkeypatch.setattr(ollama_service, "_request_json", fake_request)
    monkeypatch.setattr(ollama_service.time, "sleep", lambda _: None)
    result = ollama_service.evaluate_answer({
        "question": "How do you keep cached data accurate?",
        "expectedConcepts": ["Cache invalidation", "Consistency"],
        "answer": "I use cache invalidation when its source changes.",
    })

    assert result["actionableFeedback"]
    assert "schema validation" in prompts[1]
    assert "Do not omit any field" in prompts[1]


def test_missing_actionable_list_uses_model_feedback_text(monkeypatch):
    response = _evaluation_response()
    response.pop("actionableFeedback")
    monkeypatch.setattr(
        ollama_service,
        "_request_json",
        lambda path, payload: {"message": {"content": json.dumps(response)}},
    )

    with pytest.raises(ollama_service.OllamaError, match="invalid response twice"):
        ollama_service.evaluate_answer({
            "question": "How do you keep cached data accurate?",
            "expectedConcepts": ["Cache invalidation", "Consistency"],
            "answer": "I use cache invalidation when source data changes.",
        })


def test_model_can_mark_a_named_but_unexplained_concept_missing(monkeypatch):
    response = _evaluation_response()
    response["conceptReview"][0] = {
        "concept": "Cache invalidation",
        "status": "missing",
        "evidence": "",
    }
    monkeypatch.setattr(
        ollama_service,
        "_request_json",
        lambda path, payload: {"message": {"content": json.dumps(response)}},
    )
    result = ollama_service.evaluate_answer({
        "question": "How do you keep cached data accurate?",
        "expectedConcepts": ["Cache invalidation", "Consistency"],
        "answer": "Cache invalidation matters, but I would not use it without checking the source data.",
    })

    assert result["correctConcepts"] == []
    assert result["missingConcepts"] == ["Cache invalidation", "Consistency"]


def test_evaluation_preserves_model_concept_evidence(monkeypatch):
    response = _evaluation_response()
    response["conceptReview"] = [
        {
            "concept": "market trends",
            "status": "covered",
            "evidence": "I would compare the needs users describe with market trends and competitor offerings.",
        },
        {"concept": "Consistency", "status": "missing", "evidence": ""},
    ]
    monkeypatch.setattr(
        ollama_service,
        "_request_json",
        lambda path, payload: {"message": {"content": json.dumps(response)}},
    )

    result = ollama_service.evaluate_answer({
        "question": "How do you include market trends?",
        "expectedConcepts": ["market trends", "Consistency"],
        "answer": (
            "I would start by gathering user feedback through interviews and usage data, then compare the needs "
            "users describe with market trends and competitor offerings."
        ),
    })

    assert result["correctConcepts"] == ["market trends"]
    assert result["conceptReview"][0]["evidence"] == response["conceptReview"][0]["evidence"]


def test_empty_model_evidence_does_not_invalidate_evaluation(monkeypatch):
    response = _evaluation_response()
    response["conceptReview"][0]["evidence"] = ""
    monkeypatch.setattr(
        ollama_service,
        "_request_json",
        lambda path, payload: {"message": {"content": json.dumps(response)}},
    )

    result = ollama_service.evaluate_answer({
        "question": "How do you keep cached data accurate?",
        "expectedConcepts": ["Cache invalidation", "Consistency"],
        "answer": "I use cache invalidation when source data changes.",
    })

    assert result["correctConcepts"] == ["Cache invalidation"]


def test_assessment_is_consistent_with_scores_and_concept_coverage(monkeypatch):
    response = _evaluation_response()
    response["assessment"] = "correct"
    response["conceptReview"][1] = {
        "concept": "Consistency",
        "status": "covered",
        "evidence": "consistent reads",
    }
    monkeypatch.setattr(
        ollama_service,
        "_request_json",
        lambda path, payload: {"message": {"content": json.dumps(response)}},
    )

    result = ollama_service.evaluate_answer({
        "question": "How do you keep cached data accurate?",
        "expectedConcepts": ["Cache invalidation", "Consistency"],
        "answer": "I use cache invalidation when source data changes and ensure consistent reads.",
    })

    assert result["assessment"] == "correct"


def test_truncated_ollama_output_is_retried(monkeypatch):
    attempts = []

    def fake_request(path, payload):
        attempts.append(payload)
        if len(attempts) == 1:
            return {"done": False, "done_reason": "length", "eval_count": 900}
        return {"message": {"content": json.dumps(_evaluation_response())}, "done": True, "done_reason": "stop"}

    monkeypatch.setattr(ollama_service, "_request_json", fake_request)
    monkeypatch.setattr(ollama_service.time, "sleep", lambda _: None)

    result = ollama_service.evaluate_answer({
        "question": "How do you keep cached data accurate?",
        "expectedConcepts": ["Cache invalidation", "Consistency"],
        "answer": "I use cache invalidation when source data changes.",
    })

    assert len(attempts) == 2
    assert result["assessment"] == "partially_correct"


def test_ollama_connection_failure_is_an_explicit_api_error(client, monkeypatch):
    monkeypatch.setattr(
        "app.generate_question",
        lambda context: (_ for _ in ()).throw(ollama_service.OllamaError("Ollama is offline.", 503)),
    )

    response = client.post("/api/interview", json={
        "action": "start",
        "role": "Backend Developer",
    })

    assert response.status_code == 503
    assert response.json["error"] == "Ollama is offline."


def test_report_requires_evaluated_answers(client):
    response = client.post("/api/interview", json={
        "action": "finish",
        "role": "Backend Developer",
        "answers": [{"question": "Question?", "answer": "Candidate response"}],
    })

    assert response.status_code == 400
    assert "completed evaluation" in response.json["error"]


def test_pdf_report_returns_a_downloadable_pdf(client):
    entry = {
        "question": "How do you keep cached data accurate?",
        "answer": "I invalidate the cache when source data changes.",
        "evaluation": {
            **_evaluation_response(),
            "overallScore": 70,
        },
    }
    response = client.post("/api/report", json={
        "candidate": "Candidate",
        "role": "Backend Developer",
        "interviewDate": "2026-11-03",
        "answers": [entry],
        "report": {
            "score": 70,
            "dimensionScores": _evaluation_response()["dimensionScores"],
            "strengths": ["Clear communication"],
            "weaknesses": ["Explain edge cases"],
            "missingConcepts": ["Cache invalidation"],
            "recommendations": ["Practice cache design questions"],
            "headline": "A useful practice session.",
        },
    })

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF-")
    assert "interview-performance-report.pdf" in response.headers["Content-Disposition"]
