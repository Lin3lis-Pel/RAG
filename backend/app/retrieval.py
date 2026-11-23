"""Retrieval over rag_chunks: vector search and keyword search.

Vector search = embed the question with the same model as the chunks, then ask PostgreSQL for the K chunks
whose embeddings are closest by cosine distance (`embedding <=> query`, 0 = same direction). It matches meaning:
"a heist inside dreams" finds Inception without the word "Inception".

Keyword search = PostgreSQL full-text search over the chunk text. It matches words: each keyword from query
translation must appear as a phrase ("Hugh Jackman" = 'hugh' followed by 'jackman'), after stemming
("thrillers" -> "thriller"). It is strong where embeddings are weak: exact names of people and films.

Neither query ever puts user or model text into the SQL itself: the embedding and the keywords are bound
parameters. PostgreSQL chooses how to run the vector query and uses the HNSW index: approximate, but with the same
top 8 as an exact scan on 200 real queries, in ~5 ms instead of ~72 ms.

Either search can be narrowed to some films, by id or by filters (genres, years, rating). HNSW finds the nearest
chunks first and filters them afterwards, so a narrow condition could leave fewer than K. pgvector's iterative scan
(0.8+) then keeps reading further along the index until it has K rows; `strict_order` keeps them nearest first.
"""

import re
from dataclasses import dataclass, field

from sqlalchemy import BigInteger, Select, any_, func, literal, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import embeddings
from app.config import get_settings
from app.errors import AppError
from app.models import Movie, RagChunk, Review
from app.query_translation import QueryFilters

MAX_KEYWORDS_SEARCHED = 8


@dataclass
class RetrievedChunk:
    id: int
    movie_id: str
    movie_title: str
    year: int | None
    content: str
    distance: float | None  # cosine distance from vector search; None for a keyword-only result
    metadata: dict = field(default_factory=dict)
    url: str | None = None  # link to the original review; None for profile chunks
    keyword_score: float | None = None  # full-text rank from keyword search


def _review_url():
    """Review chunks point at their review by metadata->>'review_id'; the review row holds the original URL."""
    review_id = RagChunk.metadata_["review_id"].astext.cast(BigInteger)
    return Review.id == review_id, Review.metadata_["url"].astext.label("url")


def _run(session: Session, statement, iterative_scan: bool = False) -> list:
    try:
        if iterative_scan:  # for this transaction only
            session.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
        return session.execute(statement).all()
    except SQLAlchemyError as exc:
        raise AppError("RAG_RETRIEVAL_FAILED", "Unable to retrieve movie information.", 503) from exc


def _chunk(row, distance: float | None = None, keyword_score: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        id=row.id,
        movie_id=row.movie_id,
        movie_title=row.metadata_.get("movie_title", row.movie_id),
        year=row.metadata_.get("year"),
        content=row.content,
        distance=distance,
        metadata=row.metadata_,
        url=row.url,
        keyword_score=keyword_score,
    )


def _narrow(statement: Select, movie_ids: list[str] | None, filters: QueryFilters | None) -> Select:
    """Only these films' chunks, and/or only chunks of films inside the filters (every genre, the years, the rating)."""
    if movie_ids:
        statement = statement.where(RagChunk.movie_id.in_(movie_ids))
    if filters is not None and not filters.is_empty():
        statement = statement.join(Movie, Movie.id == RagChunk.movie_id)
        for genre in filters.genres:
            statement = statement.where(literal(genre) == any_(Movie.genres))  # :genre = ANY(genres)
        if filters.year_min is not None:
            statement = statement.where(Movie.year >= filters.year_min)
        if filters.year_max is not None:
            statement = statement.where(Movie.year <= filters.year_max)
        if filters.rating_min is not None:
            statement = statement.where(Movie.rating >= filters.rating_min)
    return statement


# Implements: specs/6.md#AC-002, #AC-006, #AC-007
def search_chunks(
    session: Session, query_vector: list[float], k: int, movie_ids: list[str] | None = None,
    filters: QueryFilters | None = None,
) -> list[RetrievedChunk]:
    """The K nearest chunks. movie_ids: only those films' chunks (per-film retrieval for multi-film questions);
    filters: only chunks of films inside them."""
    distance = RagChunk.embedding.cosine_distance(query_vector).label("distance")
    join_on, url = _review_url()
    statement = (
        select(RagChunk.id, RagChunk.movie_id, RagChunk.content, RagChunk.metadata_, distance, url)
        .outerjoin(Review, join_on)
        .where(RagChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(k)
    )
    narrowed = _narrow(statement, movie_ids, filters)
    rows = _run(session, narrowed, iterative_scan=narrowed is not statement)
    return [_chunk(row, distance=float(row.distance)) for row in rows]


# Implements: specs/6.md#AC-001
def retrieve(
    question: str, session: Session, k: int | None = None, filters: QueryFilters | None = None
) -> list[RetrievedChunk]:
    query_vector = embeddings.embed_query(question)
    limits = {"filters": filters} if filters is not None else {}
    return search_chunks(session, query_vector, k or get_settings().retrieval_top_k, **limits)


_TITLE_WITH_YEAR = re.compile(r"^(?P<title>.+?)\s*\((?P<year>\d{4})\)$")


def named_movies(session: Session, titles: list[str]) -> list[tuple[str, list[str]]]:
    """The films behind titles from query translation, as (label, movie ids), in the order given. Only
    exact (case-insensitive) title matches count; "Title (Year)" picks one year, a shared title keeps all its films;
    a title that is not in the database is dropped."""
    found, seen = [], set()
    for name in titles:
        match = _TITLE_WITH_YEAR.match(name)
        title, year = (match["title"], int(match["year"])) if match else (name, None)
        statement = select(Movie.id, Movie.title, Movie.year).where(func.lower(Movie.title) == title.lower())
        if year is not None:
            statement = statement.where(Movie.year == year)
        rows = [r for r in _run(session, statement.order_by(Movie.year)) if r.id not in seen]
        if rows:
            seen.update(r.id for r in rows)
            label = rows[0].title if len(rows) > 1 else f"{rows[0].title} ({rows[0].year})"
            found.append((label, [r.id for r in rows]))
    return found


def keyword_search(
    session: Session, keywords: list[str], k: int | None = None, movie_ids: list[str] | None = None,
    filters: QueryFilters | None = None,
) -> list[RetrievedChunk]:
    """Chunks containing any of the keywords as a phrase, best full-text rank first. No keywords -> no results.
    movie_ids and filters narrow the films as in search_chunks."""
    terms = [kw.strip() for kw in keywords if kw and kw.strip()][:MAX_KEYWORDS_SEARCHED]
    if not terms:
        return []

    # phraseto_tsquery('english', 'Hugh Jackman') = 'hugh' <-> 'jackman'; || joins the phrases with OR.
    query = None
    for term in terms:
        phrase = func.phraseto_tsquery("english", term)
        query = phrase if query is None else query.op("||")(phrase)

    # ts_rank_cd scores how often and how close together the matches are; normalisation 32 maps it to 0..1.
    score = func.ts_rank_cd(RagChunk.search_vector, query, 32).label("score")
    join_on, url = _review_url()
    statement = (
        select(RagChunk.id, RagChunk.movie_id, RagChunk.content, RagChunk.metadata_, score, url)
        .outerjoin(Review, join_on)
        .where(RagChunk.search_vector.op("@@")(query))
        .order_by(score.desc(), RagChunk.id)
        .limit(k or get_settings().keyword_top_k)
    )
    statement = _narrow(statement, movie_ids, filters)
    return [_chunk(row, keyword_score=float(row.score)) for row in _run(session, statement)]
