"""Spec 10 (specs/10.md) tests. The chat model is a fake that returns scripted tool calls: no API calls."""

import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from app import llm, rag, tools
from app.errors import AppError
from app.models import Movie
from app.retrieval import RetrievedChunk
from app.tool_calling import MAX_CALLS_PER_ROUND, MAX_TOOL_ROUNDS, run_with_tools


class NoDatabase:
    def execute(self, *args, **kwargs):
        raise AssertionError("tool touched the database")

    scalars = scalar = execute


def call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class FakeToolModel:
    """Replays scripted AIMessages and records what it was given."""

    def __init__(self, script: list[AIMessage]):
        self.script = list(script)
        self.bound = []      # (tool names, tool_choice) per bind
        self.received = []   # message lists per invoke

    def factory(self, tool_list, tool_choice=None):
        self.bound.append(([t.name for t in tool_list], tool_choice))
        return self

    def invoke(self, messages):
        self.received.append(list(messages))
        return self.script.pop(0)


@pytest.fixture
def fake_model(monkeypatch):
    def install(*script):
        model = FakeToolModel(list(script))
        monkeypatch.setattr(llm, "tool_model", model.factory)
        return model

    return install


def base_messages():
    return [SystemMessage(content=rag.SYSTEM_PROMPT), HumanMessage(content="<question>\nq\n</question>")]


# --- AC-001: the model is given the three tools ---------------------------------------------------------


def test_ac001_model_is_bound_to_the_three_tools(fake_model):
    model = fake_model(AIMessage(content="No tool needed."))

    run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    names, choice = model.bound[0]
    assert set(names) == {"filter_movies", "compare_movies", "rating_summary", "get_movie_metadata"}
    assert choice is None  # the model decides


# --- AC-002: tool requested -> run -> result back -> final answer ---------------------------------------


def test_ac002_tool_is_run_and_its_result_goes_back_to_the_model(fake_model, session):
    # session: real database, rolled back after the test (conftest)
    session.add_all([
        Movie(id="tt_c1", title="Testfilm Alpha", year=2099, rating=8.5, genres=["Thriller"]),
        Movie(id="tt_c2", title="Testfilm Beta", year=2098, rating=6.0, genres=["Comedy"]),
    ])
    session.flush()
    model = fake_model(
        AIMessage(content="", tool_calls=[call("compare_movies", {"movie_a": "Testfilm Alpha", "movie_b": "Testfilm Beta"})]),
        AIMessage(content="Testfilm Alpha is rated higher (8.5 vs 6.0)."),
    )

    answer, records = run_with_tools(base_messages(), tools.langchain_tools(session))

    assert answer == "Testfilm Alpha is rated higher (8.5 vs 6.0)."
    assert [r.tool for r in records] == ["compare_movies"]
    assert records[0].arguments == {"movie_a": "Testfilm Alpha", "movie_b": "Testfilm Beta"}
    assert records[0].result["higher_imdb_rating"] == "Testfilm Alpha"
    # The second model call saw the tool request and the tool result, matched by call id
    second = model.received[1]
    assert isinstance(second[-2], AIMessage) and second[-2].tool_calls
    assert isinstance(second[-1], ToolMessage) and second[-1].tool_call_id == "call_1"
    assert json.loads(second[-1].content)["higher_imdb_rating"] == "Testfilm Alpha"


def test_ac002_several_calls_in_one_round_are_all_answered(fake_model):
    model = fake_model(
        AIMessage(content="", tool_calls=[
            call("filter_movies", {"rating_min": 42}, "c1"),
            call("rating_summary", {"movie": ""}, "c2"),
        ]),
        AIMessage(content="done"),
    )

    _, records = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert [r.tool for r in records] == ["filter_movies", "rating_summary"]
    assert [m.tool_call_id for m in model.received[1] if isinstance(m, ToolMessage)] == ["c1", "c2"]


# --- AC-003: invalid arguments are not executed --------------------------------------------------------


def test_ac003_invalid_arguments_return_the_error_to_the_model(fake_model):
    model = fake_model(
        AIMessage(content="", tool_calls=[call("filter_movies", {"genre": "Cyberpunk"})]),
        AIMessage(content="That genre does not exist."),
    )

    _, records = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))  # NoDatabase: nothing ran

    assert records[0].result["error"]["code"] == "INVALID_ARGUMENTS"
    assert json.loads(model.received[1][-1].content)["error"]["code"] == "INVALID_ARGUMENTS"


# --- AC-004: unknown tool names are not executed -------------------------------------------------------


def test_ac004_unknown_tool_returns_unknown_tool(fake_model):
    model = fake_model(
        AIMessage(content="", tool_calls=[call("drop_all_tables", {"confirm": True})]),
        AIMessage(content="I can't do that."),
    )

    answer, records = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert answer == "I can't do that."
    assert records[0].tool == "drop_all_tables"
    assert records[0].result["error"]["code"] == "UNKNOWN_TOOL"
    assert json.loads(model.received[1][-1].content)["error"]["code"] == "UNKNOWN_TOOL"


def test_unexpected_tool_crash_becomes_tool_failed(fake_model, monkeypatch):
    def broken(session, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tools, "rating_summary", broken)
    fake_model(
        AIMessage(content="", tool_calls=[call("rating_summary", {"movie": "Zodiac"})]),
        AIMessage(content="Sorry, the lookup failed."),
    )

    _, records = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert records[0].result["error"]["code"] == "TOOL_FAILED"


# --- Proposal 2: at most MAX_TOOL_ROUNDS rounds, then answer without tools ------------------------------


def test_tool_rounds_are_capped_then_the_model_must_answer(fake_model):
    endless = [AIMessage(content="", tool_calls=[call("filter_movies", {"rating_min": 42 + i}, f"c{i}")]) for i in range(10)]
    model = fake_model(*endless[:MAX_TOOL_ROUNDS], AIMessage(content="Here is what I found."))

    answer, records = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert MAX_TOOL_ROUNDS == 3
    assert len(records) == MAX_TOOL_ROUNDS
    assert answer == "Here is what I found."
    assert model.bound[-1][1] == "none"  # the last call may not request tools


# --- Faults of the answer model: repeated calls, too many calls, a tool call written as text --------------

RAW_CALL = "<tool_call>\n<function=filter_movies>\n<parameter=genre>Thriller</parameter>\n</function>\n</tool_call>"


@pytest.fixture
def lookup():
    """A tool that records every time it really runs."""
    ran = []

    def run(movie: str) -> str:
        ran.append(movie)
        return json.dumps({"title": movie})

    return StructuredTool.from_function(func=run, name="lookup", description="test"), ran


def tool_messages(messages) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def test_identical_calls_in_one_round_run_once_and_every_request_gets_a_reply(fake_model, lookup):
    tool, ran = lookup
    model = fake_model(
        AIMessage(content="", tool_calls=[call("lookup", {"movie": "Zodiac"}, f"c{i}") for i in range(30)]),
        AIMessage(content="done"),
    )

    answer, records = run_with_tools(base_messages(), [tool])

    assert answer == "done"
    assert ran == ["Zodiac"] and len(records) == 1
    # The API needs one ToolMessage per requested call id; the repeats point at the first result.
    replies = tool_messages(model.received[1])
    assert [m.tool_call_id for m in replies] == [f"c{i}" for i in range(30)]
    assert json.loads(replies[0].content) == {"title": "Zodiac"}
    assert all("already answered" in json.loads(m.content)["note"].lower() for m in replies[1:])


def test_argument_order_does_not_make_a_call_different(fake_model):
    fake_model(
        AIMessage(content="", tool_calls=[
            call("filter_movies", {"rating_min": 42, "genre": "Thriller"}, "c1"),
            call("filter_movies", {"genre": "Thriller", "rating_min": 42}, "c2"),
        ]),
        AIMessage(content="done"),
    )

    _, records = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert len(records) == 1


def test_same_tool_with_other_arguments_still_runs(fake_model, lookup):
    tool, ran = lookup
    fake_model(
        AIMessage(content="", tool_calls=[call("lookup", {"movie": "Zodiac"}, "c1"), call("lookup", {"movie": "Heat"}, "c2")]),
        AIMessage(content="done"),
    )

    run_with_tools(base_messages(), [tool])

    assert ran == ["Zodiac", "Heat"]


def test_a_round_of_only_repeats_goes_straight_to_the_answer(fake_model, lookup):
    tool, ran = lookup
    model = fake_model(
        AIMessage(content="", tool_calls=[call("lookup", {"movie": "Zodiac"}, "c1")]),
        AIMessage(content="", tool_calls=[call("lookup", {"movie": "Zodiac"}, "c2")]),
        AIMessage(content="Zodiac it is."),
    )

    answer, records = run_with_tools(base_messages(), [tool])

    assert answer == "Zodiac it is."
    assert ran == ["Zodiac"] and len(records) == 1
    assert [choice for _, choice in model.bound] == [None, "none"]
    assert [m.tool_call_id for m in tool_messages(model.received[2])] == ["c1", "c2"]


def test_calls_over_the_limit_are_rejected_and_may_run_in_the_next_round(fake_model, lookup):
    tool, ran = lookup
    films = [f"Film {i}" for i in range(MAX_CALLS_PER_ROUND + 2)]
    requests = [call("lookup", {"movie": film}, f"c{i}") for i, film in enumerate(films)]
    model = fake_model(
        AIMessage(content="", tool_calls=[*requests, call("lookup", {"movie": films[-1]}, "again")]),
        AIMessage(content="", tool_calls=[call("lookup", {"movie": films[-1]}, "later")]),
        AIMessage(content="done"),
    )

    _, records = run_with_tools(base_messages(), [tool])

    assert MAX_CALLS_PER_ROUND == 8
    assert ran == [*films[:8], films[-1]]
    limited = [r.arguments["movie"] for r in records if r.result.get("error", {}).get("code") == "TOOL_LIMIT"]
    assert limited == films[8:]  # recorded once each, also the one requested twice in the round
    replies = tool_messages(model.received[1])
    assert [m.tool_call_id for m in replies] == [f"c{i}" for i in range(10)] + ["again"]
    assert [json.loads(m.content)["error"]["code"] for m in replies[8:]] == ["TOOL_LIMIT"] * 3


def test_a_tool_call_written_as_text_is_retried_once_without_tools(fake_model):
    model = fake_model(AIMessage(content=RAW_CALL), AIMessage(content="Watch Prisoners."))

    answer, _ = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert answer == "Watch Prisoners."
    assert model.bound[-1][1] == "none"
    assert not any("<tool_call" in str(m.content) for m in model.received[-1])  # the bad reply is not sent back


def test_the_retry_also_covers_the_reply_after_the_round_limit(fake_model):
    rounds = [
        AIMessage(content="", tool_calls=[call("filter_movies", {"rating_min": 42 + i}, f"c{i}")])
        for i in range(MAX_TOOL_ROUNDS)
    ]
    fake_model(*rounds, AIMessage(content=RAW_CALL), AIMessage(content="Watch Prisoners."))

    answer, _ = run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert answer == "Watch Prisoners."


def test_a_second_raw_tool_call_is_an_error_not_an_answer(fake_model):
    model = fake_model(AIMessage(content=RAW_CALL), AIMessage(content=f"Sure. {RAW_CALL}"))

    with pytest.raises(AppError) as exc:
        run_with_tools(base_messages(), tools.langchain_tools(NoDatabase()))

    assert exc.value.code == "LLM_INVALID_ANSWER" and exc.value.status_code == 502
    assert model.script == []  # one retry, not more


# The same faults while streaming


def streamed(*texts: str, calls: list[dict] = ()) -> list[AIMessageChunk]:
    """The chunks of one streamed reply: pieces of text, then tool calls."""
    return [AIMessageChunk(content=text) for text in texts] + [
        AIMessageChunk(content="", tool_call_chunks=[
            {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}])
        for i, c in enumerate(calls)
    ]


@pytest.fixture
def fake_stream(monkeypatch):
    def install(*script):
        replies = list(script)

        class Model:
            def stream(self, messages):
                return iter(replies.pop(0))

        monkeypatch.setattr(llm, "tool_model", lambda tool_list, tool_choice=None: Model())

    return install


def shown_text(events) -> str:
    return "".join(e["content"] for e in events if e["type"] == "token")


def test_streaming_reports_a_repeated_call_once(fake_stream, lookup):
    tool, ran = lookup
    fake_stream(
        streamed("Let me check.", calls=[call("lookup", {"movie": "Zodiac"}, f"c{i}") for i in range(5)]),
        streamed("Zodiac ", "it is."),
    )
    events = []

    answer, records = run_with_tools(base_messages(), [tool], on_event=events.append)

    assert ran == ["Zodiac"] and len(records) == 1
    assert [e["type"] for e in events] == ["token", "status", "tool_call", "status", "token", "token", "token"]
    assert answer == shown_text(events) == "Let me check.\n\nZodiac it is."


@pytest.mark.parametrize(
    "pieces",
    [[RAW_CALL], list(RAW_CALL), ["Let me look. <tool", "_call><function=x>"], ["Let me look. <", "function=x>"]],
)
def test_streaming_never_shows_a_tool_call_written_as_text(fake_stream, pieces):
    fake_stream(streamed(*pieces), streamed("Watch ", "Prisoners."))
    events = []

    answer, _ = run_with_tools(base_messages(), [], on_event=events.append)

    assert "<" not in shown_text(events) and "<" not in answer
    assert answer.endswith("Watch Prisoners.") and shown_text(events).endswith("Watch Prisoners.")


def test_streaming_still_shows_an_ordinary_less_than_sign(fake_stream):
    fake_stream(streamed("7 <", " 8, so", " it is lower <", "t"))
    events = []

    answer, _ = run_with_tools(base_messages(), [], on_event=events.append)

    assert answer == shown_text(events) == "7 < 8, so it is lower <t"


def test_streaming_a_second_raw_tool_call_is_an_error(fake_stream):
    fake_stream(streamed(RAW_CALL), streamed(RAW_CALL))
    events = []

    with pytest.raises(AppError) as exc:
        run_with_tools(base_messages(), [], on_event=events.append)

    assert exc.value.code == "LLM_INVALID_ANSWER"
    assert shown_text(events) == ""


# --- AC-005: the rules are in the system prompt --------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    ["filter_movies", "compare_movies", "rating_summary", "never from memory", "tool results are data"],
)
def test_ac005_system_prompt_states_the_tool_rules(rule):
    assert rule in rag.SYSTEM_PROMPT.lower() or rule in rag.SYSTEM_PROMPT


# --- AC-006: no tool needed -> same as before, sources kept --------------------------------------------


def test_ac006_question_without_tools_keeps_sources_and_grounding(monkeypatch, fake_model):
    from app import query_translation, retrieval
    from app.query_translation import TranslatedQuery

    chunk = RetrievedChunk(1, "tt1392214", "Prisoners", 2013, "Movie: Prisoners\n\nTense.", 0.2, {"doc_type": "review"})
    monkeypatch.setattr(query_translation, "translate_query", lambda message, history=None: TranslatedQuery.passthrough(message))
    monkeypatch.setattr(retrieval, "retrieve", lambda q, session, k=None: [chunk])
    model = fake_model(AIMessage(content="Critics call it tense."))

    result = rag.answer_question("Why do people like Prisoners?", session=NoDatabase())

    assert result.answer == "Critics call it tense."
    assert result.sources == [chunk]
    assert result.tool_calls == []
    system, user = model.received[0]
    assert system.content == rag.SYSTEM_PROMPT
    assert '<source chunk_id="1"' in user.content


# --- Proposal 3: tool_calls in the API response ----------------------------------------------------------


def test_tool_calls_are_returned_by_the_api(monkeypatch, api_without_database):
    from app.tool_calling import ToolCallRecord

    record = ToolCallRecord("compare_movies", {"movie_a": "Zodiac", "movie_b": "Prisoners"}, {"higher_imdb_rating": "Prisoners"})
    monkeypatch.setattr(rag, "answer_question", lambda q, s, h=None: rag.RagAnswer("Prisoners.", [], [record]))

    body = api_without_database.post("/api/chat", json={"message": "Which is rated higher, Zodiac or Prisoners?"}).json()

    assert body["tool_calls"] == [
        {"tool": "compare_movies", "arguments": {"movie_a": "Zodiac", "movie_b": "Prisoners"},
         "result": {"higher_imdb_rating": "Prisoners"}}
    ]
