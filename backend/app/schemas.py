"""Pydantic models for every request and response body. Add fields; never rename them.

The frontend's TypeScript types are generated from these (frontend/README.md), so there is no second copy to
keep in step.
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, RootModel


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)
    # The browser keeps one UUID per conversation. Omitted -> a new conversation is started.
    conversation_id: UUID | None = None


class ResponseModel(BaseModel):
    """A body we send. Fields with a default are always present in it, so the schema (and the generated TypeScript
    types) list them as required."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class Source(ResponseModel):
    """One retrieved chunk the model was given. `cited` says whether the answer refers to it ([chunk_id] in the
    text); the other fields let the page show who said it and link to the original review."""

    movie: str
    year: int | None = None
    review_id: str | None = None  # None for a movie-profile chunk
    chunk_id: str
    source: str  # "rotten_tomatoes" (critic review) or "tmdb_imdb" (movie profile)
    critic: str | None = None
    publication: str | None = None
    url: str | None = None
    excerpt: str
    cited: bool = False


class ToolCall(ResponseModel):
    """One tool call the model made, shown under the answer."""

    tool: str
    arguments: dict
    result: dict


class VectorResult(ResponseModel):
    """One chunk returned by vector search, for the RAG process panel."""

    rank: int
    chunk_id: str
    movie: str
    year: int | None = None
    doc_type: str | None = None  # "review" or "profile"
    critic: str | None = None
    distance: float  # cosine distance: 0 = same direction as the question's embedding
    similarity: float  # 1 - distance, easier to read: higher = closer
    excerpt: str


class KeywordResult(ResponseModel):
    """One chunk returned by full-text keyword search."""

    rank: int
    chunk_id: str
    movie: str
    year: int | None = None
    doc_type: str | None = None
    critic: str | None = None
    score: float  # PostgreSQL ts_rank_cd, normalised to 0..1: more and closer matches = higher
    excerpt: str


class FusedResult(ResponseModel):
    """One chunk of the hybrid ranking: where each search put it, and its Reciprocal Rank Fusion score."""

    rank: int
    chunk_id: str
    movie: str
    year: int | None = None
    doc_type: str | None = None
    found_by: list[str]  # "vector", "keyword" or both
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float  # sum of 1 / (k + rank) over the lists the chunk appears in
    selected: bool  # sent to the model


class RagDebugOut(ResponseModel):
    """What retrieval did for this answer."""

    original_query: str
    translated_query: str  # the semantic_query that was embedded
    vector_results: list[VectorResult] = []
    selected_chunks: list[str] = []  # chunk ids sent to the model
    keywords: list[str] = []
    filters: dict = {}
    translation_origin: str  # "model", "fallback" (translation failed) or "disabled"
    history_messages: int = 0
    timings_ms: dict[str, int] = {}
    keyword_results: list[KeywordResult] = []
    strategy: str = "vector"  # "hybrid" or "vector"
    fused_results: list[FusedResult] = []  # The hybrid ranking (empty for "vector")
    # "mcp" (tools through the MCP server) or "local" (in-process); "mcp, then local" when the
    # server failed mid-answer and the remaining calls ran in-process
    tool_backend: str = "local"
    warnings: list[str] = []  # What failed while the answer could still be produced
    per_film: list[str] = []  # Films retrieved separately (a question naming two or more)
    filters_applied: bool = False  # `filters` narrowed both searches
    invalid_citations: list[str] = []  # Ids the model cited that were not among its sources (removed)


class ModelUsageOut(ResponseModel):
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0  # part of output_tokens: the model's hidden "thinking", billed as output
    cached_input_tokens: int = 0  # part of input_tokens served from the provider's prompt cache, billed at a discount
    total_tokens: int
    cost_usd: float | None = None
    cost_source: str  # "reported" (by OpenRouter), "estimated" (embedding), "not reported"


class Usage(ResponseModel):
    """Tokens and cost of one answer; by_model breaks the totals down per model."""

    model: str  # the answer model
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    by_model: list[ModelUsageOut] = []


class ChatResponse(ResponseModel):
    answer: str  # cites its sources by chunk id in square brackets: "Tense and bleak [2281]."
    sources: list[Source] = []
    tool_calls: list[ToolCall] = []
    conversation_id: str | None = None
    debug: RagDebugOut | None = None
    usage: Usage | None = None


class StoredMessage(ResponseModel):
    role: str
    content: str
    created_at: datetime


class ConversationOut(ResponseModel):
    """GET /api/conversations/{id}."""

    id: str
    created_at: datetime
    messages: list[StoredMessage] = []


class ErrorDetail(ResponseModel):
    code: str
    message: str


class ErrorResponse(ResponseModel):
    error: ErrorDetail


# --- the events of POST /api/chat/stream (app/streaming.py) ----------------------------------------------------


class StatusEvent(ResponseModel):
    type: Literal["status"]
    stage: str  # "translation", "search", "answer" or "tool"
    message: str


class TokenEvent(ResponseModel):
    type: Literal["token"]
    content: str


class ToolCallEvent(ResponseModel):
    type: Literal["tool_call"]
    data: ToolCall


class SourcesEvent(ResponseModel):
    type: Literal["sources"]
    data: list[Source]


class StreamMetadata(ResponseModel):
    """Everything /api/chat returns besides the sources."""

    answer: str
    conversation_id: str | None
    tool_calls: list[ToolCall]
    debug: RagDebugOut | None
    usage: Usage | None


class MetadataEvent(ResponseModel):
    type: Literal["metadata"]
    data: StreamMetadata


class DoneEvent(ResponseModel):
    type: Literal["done"]


class ErrorEvent(ResponseModel):
    type: Literal["error"]
    error: ErrorDetail


class StreamEvent(RootModel):
    root: Annotated[
        StatusEvent | TokenEvent | ToolCallEvent | SourcesEvent | MetadataEvent | DoneEvent | ErrorEvent,
        Field(discriminator="type"),
    ]
