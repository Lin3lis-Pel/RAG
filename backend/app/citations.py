"""Citations: which of the retrieved chunks the answer actually used.

The model gets each chunk inside a <source chunk_id="2281"> tag and is asked to write that id after every claim it
takes from the chunk: "Critics praised its tension [2281]." That makes the link between a claim and its evidence
explicit, but it is still model output, so it is checked like any other: an id only counts if that chunk was really
among the chunks sent for this question. Anything else (an invented id, one from an earlier answer) is removed from
the text and reported, never shown as a source.

This proves that the cited chunk exists and was given to the model, not that it supports the claim. The latter is
measured by the evaluation.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# [2281] or [2281, 2282], with the space before it, so that removing one does not leave " ." behind.
MARKER = re.compile(r"[ \t]*\[(\d+(?:\s*[,;]\s*\d+)*)\]")


@dataclass
class CitedAnswer:
    text: str  # the answer with invalid ids removed
    cited: list[int] = field(default_factory=list)  # valid chunk ids, in the order first cited
    invalid: list[str] = field(default_factory=list)  # ids that were not among the chunks sent to the model


def check(answer: str, sent_ids: Iterable[int]) -> CitedAnswer:
    """Keep the ids of chunks the model was given, drop the rest."""
    sent = {str(i) for i in sent_ids}
    cited: dict[str, None] = {}
    invalid: dict[str, None] = {}

    def keep_valid(match: re.Match) -> str:
        ids = list(dict.fromkeys(re.findall(r"\d+", match[1])))
        valid = [i for i in ids if i in sent]
        cited.update(dict.fromkeys(valid))
        invalid.update(dict.fromkeys(i for i in ids if i not in sent))
        space = match[0][: match[0].index("[")]
        return f"{space}[{', '.join(valid)}]" if valid else ""

    text = MARKER.sub(keep_valid, answer)
    return CitedAnswer(text=text, cited=[int(i) for i in cited], invalid=list(invalid))


def strip(answer: str) -> str:
    """The answer without any citation: for the stored conversation (the ids mean nothing to a later question) and
    for the evaluation's judge and word checks."""
    return MARKER.sub("", answer)
