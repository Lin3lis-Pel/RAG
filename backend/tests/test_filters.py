"""Filters from query translation (genres, years, minimum rating) applied in retrieval, and filter_movies' year_max.

The SQL tests run on the real (rolled-back) database; translation, embedding and the answer model are faked.
Test films use years 2096-2099, so the filters below never match a real film.
"""

import pytest
from pydantic import ValidationError
from sqlalchemy import event, select, text

from app import embeddings, query_translation, rag, retrieval, tool_calling, tools
from app.db import get_engine
from app.models import Movie, RagChunk
from app.query_translation import QueryFilters, TranslatedQuery
from app.retrieval import RetrievedChunk
from app.tools import FilterMoviesInput

DIMS = 1536


def toy_vector(*values: float) -> list[float]:
    return list(values) + [0.0] * (DIMS - len(values))


@pytest.fixture
def films(session):
    session.add_all([
        Movie(id="tt_a", title="Testfilm Alpha", year=2096, rating=8.5, genres=["Thriller", "Crime"]),
        Movie(id="tt_b", title="Testfilm Beta", year=2098, rating=6.0, genres=["Comedy"]),
        Movie(id="tt_c", title="Testfilm Gamma", year=2099, rating=8.0, genres=["Thriller"]),
    ])
    session.flush()
    for i, movie_id in enumerate(["tt_a", "tt_b", "tt_c"]):
        session.add(RagChunk(movie_id=movie_id, content=f"Quokka glimmerwort {movie_id}",
                             embedding=toy_vector(1.0, i / 10),
                             metadata_={"movie_title": movie_id, "chunk_key": f"t:{i}"}))
    session.flush()
    return session


FILTER_CASES = [
    (QueryFilters(year_min=2097, year_max=2098), {"tt_b"}),
    (QueryFilters(year_min=2096, year_max=2099, rating_min=8), {"tt_a", "tt_c"}),
    (QueryFilters(year_min=2096, genres=["Thriller"]), {"tt_a", "tt_c"}),
    (QueryFilters(year_min=2096, genres=["Thriller", "Crime"]), {"tt_a"}),  # every genre named, not any of them
    (QueryFilters(year_min=2100), set()),
]


@pytest.mark.parametrize("filters, expected", FILTER_CASES)
def test_vector_search_returns_only_films_inside_the_filters(films, filters, expected):
    films.execute(text("SET LOCAL enable_indexscan = off"))  # exact scan: HNSW is approximate for toy vectors

    found = retrieval.search_chunks(films, toy_vector(1.0), k=10, filters=filters)

    assert {c.movie_id for c in found} == expected


@pytest.mark.parametrize("filters, expected", FILTER_CASES)
def test_keyword_search_returns_only_films_inside_the_filters(films, filters, expected):
    found = retrieval.keyword_search(films, ["glimmerwort"], k=10, filters=filters)

    assert {c.movie_id for c in found} == expected


@pytest.mark.parametrize("narrowed", [{"filters": QueryFilters(year_min=2000)}, {"movie_ids": ["tt1392214"]}])
def test_iterative_scan_is_on_only_for_a_narrowed_vector_search(session, narrowed):
    statements = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(get_engine(), "before_cursor_execute", record)
    try:
        retrieval.search_chunks(session, toy_vector(1.0), k=3)
        assert not any("iterative_scan" in s for s in statements)
        retrieval.search_chunks(session, toy_vector(1.0), k=3, **narrowed)
        assert any("iterative_scan" in s for s in statements)
    finally:
        event.remove(get_engine(), "before_cursor_execute", record)
    assert session.execute(text("SHOW hnsw.iterative_scan")).scalar() == "strict_order"


@pytest.mark.real_data
def test_the_hnsw_index_fills_k_rows_for_a_filter_few_chunks_pass(session):
    """Without iterative scan the index collects its ~40 nearest chunks and filters afterwards, so a filter few of
    them pass returns too few rows. The vector belongs to a film outside the filter, and the planner is forced onto
    the index."""
    vector = session.scalars(
        select(RagChunk.embedding).join(Movie).where(Movie.title == "La La Land", RagChunk.embedding.is_not(None)).limit(1)
    ).one()
    session.execute(text("SET LOCAL enable_seqscan = off"))
    session.execute(text("SET LOCAL enable_sort = off"))

    found = retrieval.search_chunks(session, list(vector), k=10, filters=QueryFilters(genres=["War"], year_max=2000))

    assert len(found) == 10
    for chunk in found:
        movie = session.get(Movie, chunk.movie_id)
        assert "War" in movie.genres and movie.year <= 2000
    assert [c.distance for c in found] == sorted(c.distance for c in found)


# --- when the pipeline applies them ----------------------------------------------------------------------


def chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(id=chunk_id, movie_id="tt1", movie_title="Prisoners", year=2013,
                          content="Movie: Prisoners\n\nx", distance=0.2, metadata={})


@pytest.fixture
def pipeline(monkeypatch):
    """rag.answer_question with a given translation; also returns the filters each search received."""

    def run(translation, matches_filters=True):
        seen = []

        def search(session, vector, k, filters=None):
            seen.append(filters)
            return [chunk(1)] if filters is None or matches_filters else []

        def keyword(session, keywords, k=None, filters=None):
            seen.append(filters)
            return []

        monkeypatch.setattr(query_translation, "translate_query", lambda m, h=None: translation)
        monkeypatch.setattr(embeddings, "embed_query", lambda t: [0.1] * DIMS)
        monkeypatch.setattr(retrieval, "search_chunks", search)
        monkeypatch.setattr(retrieval, "keyword_search", keyword)
        monkeypatch.setattr(tool_calling, "run_with_tools", lambda messages, tools: ("Answer.", []))
        return rag.answer_question("q", session=None), seen

    return run


FILTERS = QueryFilters(genres=["Thriller"], year_min=2010, year_max=2019)


def test_a_question_that_names_no_film_is_searched_with_its_filters(pipeline):
    result, seen = pipeline(TranslatedQuery(semantic_query="thriller", filters=FILTERS))

    assert seen == [FILTERS, FILTERS]
    assert result.debug.filters_applied is True and result.debug.warnings == []


def test_a_question_that_names_a_film_ignores_the_filters(pipeline):
    """The model sometimes derives a genre or year from the film itself; that must not hide the film's chunks."""
    result, seen = pipeline(TranslatedQuery(semantic_query="Prisoners", movies=["Prisoners"], filters=FILTERS))

    assert seen == [None, None] and result.debug.filters_applied is False


def test_no_filters_means_an_ordinary_search(pipeline):
    result, seen = pipeline(TranslatedQuery(semantic_query="slow burn"))

    assert seen == [None, None] and result.debug.filters_applied is False


def test_filters_nothing_matches_are_dropped_with_a_warning(pipeline):
    result, seen = pipeline(TranslatedQuery(semantic_query="thriller", filters=FILTERS), matches_filters=False)

    assert seen == [FILTERS, FILTERS, None, None]
    assert [c.id for c in result.sources] == [1]
    assert result.debug.filters_applied is False
    assert "Thriller" in result.debug.warnings[0] and "2010" in result.debug.warnings[0]


def test_an_impossible_year_range_from_the_model_is_dropped_not_searched():
    filters = QueryFilters(year_min=2019, year_max=2010, genres=["Thriller"])

    assert filters.year_min is None and filters.year_max is None and filters.genres == ["Thriller"]


# --- filter_movies: year_max --------------------------------------------------------------------------------


def test_filter_movies_year_range_is_inclusive(films):
    result = tools.filter_movies(films, year_min=2096, year_max=2098)

    assert [m["title"] for m in result["movies"]] == ["Testfilm Alpha", "Testfilm Beta"]
    assert result["filters"] == {"year_min": 2096, "year_max": 2098}


def test_year_max_alone_is_a_filter_and_its_range_is_checked():
    assert FilterMoviesInput(year_max=1990).year_max == 1990
    for bad in ({"year_max": 1500}, {"year_min": 2019, "year_max": 2010}):
        with pytest.raises(ValidationError):
            FilterMoviesInput(**bad)


@pytest.mark.real_data
def test_thrillers_from_the_2010s_exclude_the_2020s(session):
    result = tools.filter_movies(session, genre="Thriller", year_min=2010, year_max=2019, limit=25)

    assert result["total_matches"] > 0
    assert all(2010 <= m["year"] <= 2019 for m in result["movies"])
