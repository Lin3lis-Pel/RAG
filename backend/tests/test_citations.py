"""Citations: the model cites chunk ids, the backend checks them. Fake model and retrieval: no API calls, no database."""

import pytest

from app import citations, embeddings, evaluation, query_translation, rag, retrieval, tool_calling
from app.evaluation import EvalItem, RunRecord
from app.query_translation import TranslatedQuery
from app.retrieval import RetrievedChunk


def chunk(chunk_id: int, title: str = "Prisoners", year: int = 2013) -> RetrievedChunk:
    return RetrievedChunk(id=chunk_id, movie_id="tt1", movie_title=title, year=year,
                          content=f"Movie: {title}\n\nDark and tense.", distance=0.3, metadata={})


# --- checking ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("answer, text, cited, invalid", [
    ("Tense [1]. Long [2].", "Tense [1]. Long [2].", [1, 2], []),
    ("Both agree [2, 1].", "Both agree [2, 1].", [2, 1], []),
    ("Both agree [1; 2].", "Both agree [1, 2].", [1, 2], []),
    ("Tense [1] and tense [1].", "Tense [1] and tense [1].", [1], []),
    ("Tense [99].", "Tense.", [], ["99"]),  # not sent to the model: removed, with its space
    ("Tense [1, 99].", "Tense [1].", [1], ["99"]),
    ("No citation.", "No citation.", [], []),
    ("The [sic] film, see [this](https://x.y).", "The [sic] film, see [this](https://x.y).", [], []),
])
def test_only_ids_of_chunks_sent_to_the_model_count(answer, text, cited, invalid):
    result = citations.check(answer, [1, 2])

    assert (result.text, result.cited, result.invalid) == (text, cited, invalid)


def test_strip_removes_every_citation():
    assert citations.strip("Tense [1]. Long [2, 3].") == "Tense. Long."


# --- in the pipeline --------------------------------------------------------------------------------------------


@pytest.fixture
def answer_with(monkeypatch):
    def run(text):
        monkeypatch.setattr(query_translation, "translate_query", lambda m, h=None: TranslatedQuery.passthrough(m))
        monkeypatch.setattr(embeddings, "embed_query", lambda t: [0.1] * 1536)
        monkeypatch.setattr(retrieval, "search_chunks", lambda s, v, k: [chunk(1), chunk(2)])
        monkeypatch.setattr(retrieval, "keyword_search", lambda s, kw, k=None: [])
        monkeypatch.setattr(tool_calling, "run_with_tools", lambda messages, tools: (text, []))
        return rag.answer_question("q", session=None)

    return run


def test_the_answer_reports_which_sources_it_cites(answer_with):
    result = answer_with("Tense [2].")

    assert result.answer == "Tense [2]." and result.cited_ids == [2]
    assert [c.id for c in result.sources] == [1, 2]  # every chunk the model saw is still listed
    assert result.debug.invalid_citations == []


def test_an_invented_citation_is_removed_and_reported(answer_with):
    result = answer_with("Tense [2]. A masterpiece [4711].")

    assert result.answer == "Tense [2]. A masterpiece." and result.cited_ids == [2]
    assert result.debug.invalid_citations == ["4711"]


def test_the_system_prompt_asks_for_citations():
    assert "chunk_id in square brackets" in rag.SYSTEM_PROMPT


def test_the_api_marks_cited_sources_and_history_stores_the_answer_without_ids(api_without_database, monkeypatch):
    monkeypatch.setattr(rag, "answer_question", lambda message, session, history=None: rag.RagAnswer(
        answer="Tense [2].", sources=[chunk(1), chunk(2)], cited_ids=[2]))

    body = api_without_database.post("/api/chat", json={"message": "q"}).json()

    assert body["answer"] == "Tense [2]."
    assert [(s["chunk_id"], s["cited"]) for s in body["sources"]] == [("1", False), ("2", True)]
    assert api_without_database.saved[0][2] == "Tense."


# --- in the evaluation -------------------------------------------------------------------------------------------

ITEM = EvalItem(id="O1", type="opinion", question="q", expected_answer="a", expected_movies=["Prisoners (2013)"],
                must_contain_any=["2281"])


def test_citation_precision_is_the_share_of_cited_chunks_about_an_expected_film():
    rec = RunRecord(answer="A [1]. B [2].", chunks=[chunk(1), chunk(2, "Zodiac", 2007), chunk(3)], cited_ids=[1, 2],
                    invalid_citations=["9"])

    assert evaluation.citation_metrics(ITEM, rec) == {"cited": 2, "invalid": 1, "precision": 0.5, "sent_precision": 0.667}


def test_citation_precision_needs_an_expected_film_and_a_citation():
    no_film = ITEM.model_copy(update={"expected_movies": [], "no_answer": True})

    assert evaluation.citation_metrics(no_film, RunRecord(answer="x", chunks=[chunk(1)], cited_ids=[1]))["precision"] is None
    assert evaluation.citation_metrics(ITEM, RunRecord(answer="x", chunks=[chunk(1)]))["precision"] is None


def test_judge_and_word_checks_see_the_answer_without_ids():
    rec = RunRecord(answer="Tense [2281].", chunks=[chunk(2281)], cited_ids=[2281])

    judged = evaluation.build_judge_messages(ITEM, rec)[1].content
    assert "<answer>\nTense.\n</answer>" in judged
    assert evaluation.rule_checks(ITEM, rec)["must_contain_hit"] is False  # an id is not an answer


def test_the_summary_averages_citations_over_answered_questions():
    def row(**citations):
        return {"id": "x", "type": "opinion", "error": None, "judged": False, "scores": None, "retrieval": None,
                "rules": {"must_contain_hit": None, "expected_tool_used": None}, "latency_ms": 1, "cost_usd": 0.1,
                "citations": citations}

    rows = [row(cited=2, invalid=1, precision=0.5, sent_precision=0.25),
            row(cited=1, invalid=0, precision=1.0, sent_precision=0.75),
            row(cited=0, invalid=0, precision=None, sent_precision=0.5)]
    summary = evaluation.summarise("hybrid", rows, 0.0, "judge")

    assert summary["citations"] == {"answers_with_citation": 2, "invalid": 1, "precision": 0.75, "sent_precision": 0.5}
