// The answer cites its sources by chunk id: "Tense and bleak [2281]." The reader sees numbers instead, in the order
// the answer first cites each source, and the source cards carry the same numbers. The backend has already removed ids
// that were not among the answer's sources (backend/app/citations.py).

const MARKER = /[ \t]*\[(\d+(?:\s*[,;]\s*\d+)*)\]/g;
// The start of a citation that is still streaming in, so "[22" never flashes up before "81]" arrives.
const UNFINISHED = /[ \t]*\[[\d\s,;]*$/;

export const CITE = "#cite-";

// The ids the text cites, in the order first cited: the number shown for an id is its place in this list.
export function citedIds(text: string): string[] {
  const ids = [...text.matchAll(MARKER)].flatMap((match) => match[1].match(/\d+/g) ?? []);
  return [...new Set(ids)];
}

// [2281] becomes a Markdown link "[1](#cite-2281)", which the page renders as a superscript number.
export function numberCitations(text: string, streaming: boolean): string {
  const shown = streaming ? text.replace(UNFINISHED, "") : text;
  const order = citedIds(shown);
  return shown.replace(MARKER, (_, ids: string) =>
    (ids.match(/\d+/g) ?? []).map((id) => `[${order.indexOf(id) + 1}](${CITE}${id})`).join(" "),
  );
}
