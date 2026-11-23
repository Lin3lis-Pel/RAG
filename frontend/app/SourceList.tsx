// The chunks the model was given: first the ones the answer cites, numbered as in the text, then, collapsed, the rest.
// The rest is kept because it is an honest record of what retrieval found, including what the model did not use.

import type { Source } from "./api";

// Only http(s) links are rendered; anything else in the data is shown as plain text.
function safeUrl(url: string | null): string | null {
  return url && /^https?:\/\//i.test(url) ? url : null;
}

function attribution(source: Source): string {
  if (source.source === "tmdb_imdb") return "Movie profile (TMDB / IMDb)";
  return [source.critic ?? "Unknown critic", source.publication].filter(Boolean).join(", ");
}

// "Prisoners (2013), A.O. Scott, New York Times": also the tooltip of a citation in the text.
export function sourceLabel(source: Source): string {
  return `${source.movie}${source.year ? ` (${source.year})` : ""}, ${attribution(source)}`;
}

function SourceCard({ source, number, id }: { source: Source; number?: number; id?: string }) {
  const url = safeUrl(source.url);
  return (
    <li
      id={id}
      tabIndex={id ? -1 : undefined}
      className="flex min-w-0 flex-col rounded-lg border border-line p-2.5 text-xs focus:border-accent focus:outline-none"
    >
      <div className="flex items-baseline gap-1.5">
        {number !== undefined && <span className="font-medium text-accent">{number}</span>}
        <span className="truncate text-sm font-medium" title={source.movie}>
          {source.movie}
          {source.year ? ` (${source.year})` : ""}
        </span>
      </div>
      <div className="truncate text-muted" title={attribution(source)}>
        {attribution(source)}
      </div>
      <p className="mt-1 text-foreground/80">{source.excerpt}</p>
      <div className="mt-auto flex gap-2 pt-1.5 text-muted">
        {url && (
          <a href={url} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
            Original review
          </a>
        )}
        <span className="ml-auto" title={source.review_id ? `review ${source.review_id}` : undefined}>
          chunk {source.chunk_id}
        </span>
      </div>
    </li>
  );
}

const GRID = "grid gap-2 sm:grid-cols-3";

// citedOrder: the chunk ids the answer cites, in the order first cited. idPrefix makes a card's id unique per answer.
export default function SourceList({
  sources,
  citedOrder,
  idPrefix,
}: {
  sources: Source[];
  citedOrder: string[];
  idPrefix: string;
}) {
  if (sources.length === 0) return null;
  const cited = sources
    .filter((s) => s.cited)
    .sort((a, b) => citedOrder.indexOf(a.chunk_id) - citedOrder.indexOf(b.chunk_id));
  const rest = sources.filter((s) => !s.cited);

  return (
    <section className="mt-3" aria-label="Sources">
      <h3 className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted">Sources</h3>
      {cited.length > 0 ? (
        <ol className={GRID}>
          {cited.map((source) => (
            <SourceCard
              key={source.chunk_id}
              source={source}
              number={citedOrder.indexOf(source.chunk_id) + 1}
              id={`${idPrefix}${source.chunk_id}`}
            />
          ))}
        </ol>
      ) : (
        <p className="text-xs text-muted">This answer cites none of the retrieved reviews or profiles.</p>
      )}
      {rest.length > 0 && (
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-muted hover:text-foreground">
            Also retrieved, not cited ({rest.length})
          </summary>
          <ul className={`mt-1.5 ${GRID}`}>
            {rest.map((source) => (
              <SourceCard key={source.chunk_id} source={source} />
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
