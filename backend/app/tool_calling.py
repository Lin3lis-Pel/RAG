"""Tool calling: the model decides, our code executes.

The model never runs anything. It is given the tools' names, descriptions and argument schemas, and when it
thinks one would help it replies with a *request* ("call compare_movies with movie_a=Zodiac, movie_b=Prisoners")
instead of an answer. This loop:

1. sends the conversation to the model, bound to the tools;
2. if the reply contains tool requests, runs each one through `app/tools.py` (which validates the arguments
   first), and appends the result as a ToolMessage linked to the request by its call id;
3. sends everything back, so the model now sees the real numbers, and repeats;
4. stops when the model replies without a tool request: that reply is the answer.

At most MAX_TOOL_ROUNDS rounds; after that the model is called once more with tools switched off and must answer
with what it has. This caps both the wait and the cost if a model keeps asking for tools.

The answer model sometimes misbehaves in two ways, and the loop does not rely on it behaving:
- It requests the same call over and over (28-34 in one question). Each (name, arguments) pair runs once per question;
  a repeat gets a short "already answered" reply, because the API wants a reply to every requested call, and is not
  reported. At most MAX_CALLS_PER_ROUND calls run per round, which also limits calls that differ only slightly.
- It writes a tool call as plain text ("<tool_call><function=...>") instead of requesting one. That text is never shown:
  the model is asked once more with tools switched off, and if it does it again the question fails with
  LLM_INVALID_ANSWER.
"""

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from app import deadline, llm
from app.errors import AppError

# Implements: specs/10.md (proposal 2)
MAX_TOOL_ROUNDS = 3
MAX_CALLS_PER_ROUND = 8  # no question in the saved evaluation runs needed 6 calls in total

# How a tool call looks when a model writes it as text instead of requesting it.
MARKUP_STARTS = ("<tool_call", "</tool_call", "<function=", "</function")
MARKUP = re.compile("|".join(MARKUP_STARTS), re.IGNORECASE)

ALREADY_ANSWERED = {"note": "Already answered: this exact call was made earlier for this question. Use that result."}


@dataclass
class ToolCallRecord:
    """One executed (or rejected) tool call, for the API response and the page."""

    tool: str
    arguments: dict
    result: dict


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


# Implements: specs/10.md#AC-003, #AC-004
def execute(call: dict, tools_by_name: dict[str, BaseTool]) -> dict:
    """Run one requested call. Never raises: every problem becomes an error result the model can read."""
    tool = tools_by_name.get(call["name"])
    if tool is None:
        return _error("UNKNOWN_TOOL", f"There is no tool named {call['name']!r}. Available: {', '.join(tools_by_name)}.")
    try:
        # The LangChain wrapper validates the arguments against the tool's Pydantic schema and returns
        # INVALID_ARGUMENTS without running anything when they do not fit.
        return json.loads(tool.invoke(call.get("args") or {}))
    except Exception:  # an unexpected bug in a tool must not crash the chat
        return _error("TOOL_FAILED", f"The {call['name']} tool failed unexpectedly.")


def _call_key(call: dict) -> str:
    """Two requests are the same call when the name and the arguments match, whatever the argument order."""
    return json.dumps([call["name"], call.get("args") or {}], sort_keys=True)


def _before_markup(text: str) -> str:
    """The text up to the first piece of tool-call markup (all of it when there is none)."""
    found = MARKUP.search(text)
    return text[: found.start()] if found else text


class MarkupFilter:
    """Passes streamed text on, but never tool-call markup. A piece may end in the middle of a tag ("<tool" + "_call>"),
    so an ending that could be the start of one is held back until the next piece decides."""

    def __init__(self, emit: Callable[[str], None]):
        self.emit = emit
        self.held = ""
        self.blocked = False  # markup was found: nothing more of this reply is shown

    def add(self, text: str) -> None:
        if self.blocked:
            return
        text = self.held + text
        self.blocked = MARKUP.search(text) is not None
        safe = _before_markup(text)
        cut = len(safe)
        if not self.blocked:
            # Only the last "<" can be the start of an unfinished tag: no tag contains a second one.
            start = safe.rfind("<")
            if start >= 0 and any(tag.startswith(safe[start:].lower()) for tag in MARKUP_STARTS):
                cut = start
        self.held = safe[cut:]
        if safe[:cut]:
            self.emit(safe[:cut])

    def finish(self) -> None:
        """The reply ended: what was held back was ordinary text after all."""
        if self.held:
            self.emit(self.held)
            self.held = ""


# Implements: specs/10.md#AC-001, #AC-002
def run_with_tools(
    messages: list[BaseMessage], tools: list[BaseTool], on_event: Callable[[dict], None] | None = None
) -> tuple[str, list[ToolCallRecord]]:
    """Run the model with tools until it answers. Returns the answer text and every tool call made.

    With on_event, every model round is streamed and the loop reports what happens as events: each piece
    of answer text ("token"), each tool about to run ("status") and each tool result ("tool_call"). MiMo sometimes
    writes a sentence before calling a tool ("let me look that up"); it has already been shown, so it stays in the
    answer, as its own paragraph. Without on_event it works as before (only the final reply's text).
    """
    messages = list(messages)
    tools_by_name = {tool.name: tool for tool in tools}
    model = llm.tool_model(tools)
    records: list[ToolCallRecord] = []
    answered: set[str] = set()  # calls already run for this question

    texts: list[str] = []  # everything the user saw: a model sometimes writes a sentence before a tool call

    def call_model(runnable) -> AIMessage:
        if on_event is None:
            return llm.invoke(runnable, messages)
        started_round = False

        def on_text(text: str) -> None:
            nonlocal started_round
            if not started_round and any(t.strip() for t in texts):
                on_event({"type": "token", "content": "\n\n"})  # new paragraph after an earlier round's text
            started_round = True
            on_event({"type": "token", "content": text})

        markup_filter = MarkupFilter(on_text)
        reply = llm.stream(runnable, messages, markup_filter.add)
        markup_filter.finish()
        return reply

    def shown() -> str:
        return "\n\n".join(t for t in texts if t.strip())

    def record(call: dict, result: dict) -> None:
        entry = ToolCallRecord(tool=call["name"], arguments=call.get("args") or {}, result=result)
        records.append(entry)
        if on_event:
            on_event({"type": "tool_call", "data": asdict(entry)})

    def run_round(calls: list[dict]) -> bool:
        """Answer every requested call with a ToolMessage. False when none of them was new."""
        ran = 0
        over_limit: set[str] = set()  # only this round: the model may ask for these again
        for call in calls:
            deadline.check()  # No new tool after the question's time limit
            key = _call_key(call)
            if key in answered:
                result = ALREADY_ANSWERED
            elif ran >= MAX_CALLS_PER_ROUND:
                result = _error("TOOL_LIMIT", f"Not run: at most {MAX_CALLS_PER_ROUND} tool calls per round.")
                if key not in over_limit:
                    over_limit.add(key)
                    record(call, result)
            else:
                if on_event:
                    on_event({"type": "status", "stage": "tool", "message": f"Calling {call['name']}"})
                result = execute(call, tools_by_name)
                answered.add(key)
                ran += 1
                record(call, result)
            messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call["id"]))
        if on_event:
            on_event({"type": "status", "stage": "answer", "message": "Writing the answer"})
        return ran > 0

    def call_without_tools() -> AIMessage:
        # Same tools in view (the history refers to them), but the model may not call any.
        return call_model(llm.tool_model(tools, tool_choice="none"))

    reply = None
    for _ in range(MAX_TOOL_ROUNDS):
        reply = call_model(model)
        if not reply.tool_calls:
            break  # this reply is the answer
        texts.append(_before_markup(reply.text))
        messages.append(reply)
        made_progress = run_round(reply.tool_calls)
        reply = None
        if not made_progress:
            break  # only repeats: another round with tools would bring the same again
    if reply is None:  # round limit reached, or only repeats
        reply = call_without_tools()

    if MARKUP.search(reply.text):  # a tool call written as text is not an answer; the bad reply is not sent back
        texts.append(_before_markup(reply.text))  # when streaming, the text before it has been shown
        reply = call_without_tools()
        if MARKUP.search(reply.text):
            raise AppError("LLM_INVALID_ANSWER", "The language model did not give a usable answer. Please try again.", 502)

    texts.append(reply.text)
    return shown() if on_event else reply.text, records
