"""Render data/TAXONOMY.md from data/taxonomy.json.

    python -m src.etl.render_taxonomy           # rewrite the file
    python -m src.etl.render_taxonomy --check   # exit 1 if it is stale

The JSON is the source of truth. The Markdown exists so the mapping is readable
on GitHub; keeping it generated means the two cannot disagree.
"""

from __future__ import annotations

import argparse
import sys

from src.config import TAXONOMY_MD_PATH
from src.taxonomy import Taxonomy, load_taxonomy

_PREAMBLE = """<!-- GENERATED FILE — do not edit by hand.
     Source: data/taxonomy.json
     Regenerate: python -m src.etl.render_taxonomy -->

# Label taxonomy

Two label levels are in play, and they are not the same thing.

**Fine intent (77 classes).** The label that ships with BANKING77. This is what
the models are actually trained to predict, and it is what every accuracy and
F1 figure over "intents" refers to.

**Coarse category (9 classes).** A triage queue. This is *our* mapping, defined
in `data/taxonomy.json` and written by hand for this project — it is not part of
the dataset and has no external validation. A category prediction is produced by
mapping the predicted intent through the table at the bottom of this file, not
by a second model. Category metrics therefore inherit every intent error that
crosses a category boundary, and are unaffected by intent errors that stay
inside one.

Each category carries the operational metadata the routing layer needs: the
queue it lands in, the SLA the business promises, and a base priority. The
routing rules in `src/routing/rules.py` may raise the priority above the base
(never lower it) — see that module for the escalation conditions.

## Categories

"""

_TABLE_NOTE = """
`security_critical` categories bypass the low-confidence hold: a record routed
there is escalated to a human immediately rather than parked in a review queue,
because the cost of a delayed fraud report is higher than the cost of a false
alarm.

## Intent to category

All {n_intents} intents, grouped by the category they map to.

"""


def render(taxonomy: Taxonomy) -> str:
    lines = [_PREAMBLE]

    lines.append(
        "| Category | Queue | SLA (min) | Base priority | Security critical | Intents |\n"
        "| --- | --- | ---: | ---: | :---: | ---: |\n"
    )
    for key in taxonomy.category_keys:
        cat = taxonomy.categories[key]
        lines.append(
            f"| `{cat.key}` | `{cat.queue}` | {cat.sla_minutes} | P{cat.base_priority} "
            f"| {'yes' if cat.security_critical else 'no'} "
            f"| {len(taxonomy.intents_in_category(key))} |\n"
        )

    lines.append("\n")
    for key in taxonomy.category_keys:
        cat = taxonomy.categories[key]
        lines.append(f"**`{cat.key}`** — {cat.description}\n\n")

    lines.append(_TABLE_NOTE.format(n_intents=len(taxonomy.intents)))

    for key in taxonomy.category_keys:
        cat = taxonomy.categories[key]
        members = taxonomy.intents_in_category(key)
        lines.append(f"### `{cat.key}` ({len(members)} intents)\n\n")
        for intent in members:
            lines.append(f"- `{intent}`\n")
        lines.append("\n")

    return "".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit non-zero if the file is stale")
    args = parser.parse_args(argv)

    content = render(load_taxonomy())

    if args.check:
        current = (
            TAXONOMY_MD_PATH.read_text(encoding="utf-8")
            if TAXONOMY_MD_PATH.exists()
            else ""
        )
        if current != content:
            print(
                "data/TAXONOMY.md is out of date with data/taxonomy.json.\n"
                "Run: python -m src.etl.render_taxonomy",
                file=sys.stderr,
            )
            return 1
        print("data/TAXONOMY.md is up to date")
        return 0

    TAXONOMY_MD_PATH.write_text(content, encoding="utf-8")
    print(f"wrote {TAXONOMY_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
