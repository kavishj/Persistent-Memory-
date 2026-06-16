from core.writer import extractor


def test_parse_response_filters_invalid_and_low_confidence_facts():
    raw = """
    [
      {
        "fact": "The API enforces a 100 requests per minute limit.",
        "fact_type": "constraint",
        "entities": ["API"],
        "confidence": 0.91,
        "source": "rate limit error"
      },
      {
        "fact": "This one is too uncertain.",
        "fact_type": "constraint",
        "entities": [],
        "confidence": 0.40,
        "source": "guess"
      },
      {
        "fact": "",
        "fact_type": "environment",
        "entities": [],
        "confidence": 0.85,
        "source": "empty"
      }
    ]
    """

    facts = extractor._parse_response(raw)

    assert len(facts) == 1
    assert facts[0].fact == "The API enforces a 100 requests per minute limit."
    assert facts[0].fact_type == "constraint"


def test_extract_facts_prefers_groq(monkeypatch):
    monkeypatch.setattr(extractor, "GROQ_API_KEY", "groq-test")
    monkeypatch.setattr(extractor, "ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setattr(extractor, "GEMINI_API_KEY", "gemini-test")

    calls = []

    def groq_stub(user_prompt: str) -> str:
        calls.append("groq")
        return """
        [
          {
            "fact": "Groq extracted a valid fact.",
            "fact_type": "capability",
            "entities": ["Groq"],
            "confidence": 0.88,
            "source": "session log"
          }
        ]
        """

    def anthropic_stub(user_prompt: str) -> str:
        calls.append("anthropic")
        raise AssertionError("Anthropic should not be called when Groq succeeds")

    def gemini_stub(user_prompt: str) -> str:
        calls.append("gemini")
        raise AssertionError("Gemini should not be called when Groq succeeds")

    monkeypatch.setattr(extractor, "_call_groq", groq_stub)
    monkeypatch.setattr(extractor, "_call_anthropic", anthropic_stub)
    monkeypatch.setattr(extractor, "_call_gemini", gemini_stub)

    facts = extractor.extract_facts("session log", "agent-1", task_type="debug")

    assert calls == ["groq"]
    assert len(facts) == 1
    assert facts[0].fact == "Groq extracted a valid fact."


def test_extract_facts_falls_back_to_anthropic_then_gemini(monkeypatch):
    monkeypatch.setattr(extractor, "GROQ_API_KEY", "groq-test")
    monkeypatch.setattr(extractor, "ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setattr(extractor, "GEMINI_API_KEY", "gemini-test")

    calls = []

    def groq_stub(user_prompt: str) -> str:
        calls.append("groq")
        raise RuntimeError("groq unavailable")

    def anthropic_stub(user_prompt: str) -> str:
        calls.append("anthropic")
        raise RuntimeError("anthropic unavailable")

    def gemini_stub(user_prompt: str) -> str:
        calls.append("gemini")
        return """
        [
          {
            "fact": "Gemini fallback produced a fact.",
            "fact_type": "environment",
            "entities": ["Gemini"],
            "confidence": 0.82,
            "source": "fallback"
          }
        ]
        """

    monkeypatch.setattr(extractor, "_call_groq", groq_stub)
    monkeypatch.setattr(extractor, "_call_anthropic", anthropic_stub)
    monkeypatch.setattr(extractor, "_call_gemini", gemini_stub)

    facts = extractor.extract_facts("session log", "agent-1", task_type="debug")

    assert calls == ["groq", "anthropic", "gemini"]
    assert len(facts) == 1
    assert facts[0].fact == "Gemini fallback produced a fact."
