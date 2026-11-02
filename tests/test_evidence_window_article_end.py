"""The judge's window stops at the cited article's own end.

The window was a fixed slice around the cited title (300 characters before, 2,500 after). When the
cited item's own text was short, the slice ran on into the NEXT item of the same feed, and a
citation of the short item was backed by a sentence from the next article. The characters before
the title could likewise reach the PREVIOUS item. These pin, for an RSS feed, an Atom feed, an HTML
page split by headings and an HTML page of <article> elements:

  * the window of a short item holds its own text and none of the next or the previous item's;
  * a claim whose only support is in the next item is not judged supported from it, even when the
    judge quotes that sentence (the quote is located outside the cited article: not located);
  * a claim supported inside the short item still is.

The feeds and pages below are synthetic (example.test hosts, invented titles and text).
"""
from __future__ import annotations

import unittest

import helpers  # noqa: F401  (puts the repo on sys.path)

from claim_seal import evidence as ev
from claim_seal.evidence import (EVIDENCE_WINDOW_AHEAD, build_claim_evidence, capture_sources,
                                 evidence_candidates, evidence_window)
from claim_seal.normalize import NORMALIZER_ID, normalize

FETCHED_AT = "2025-01-15T09:30:00Z"
TITLE_PREV = "Post zero: a compiler cache"
TITLE_SHORT = "Post alpha: release notes"
TITLE_NEXT = "Post beta: speech on the edge"
TITLES = [TITLE_PREV, TITLE_SHORT, TITLE_NEXT]
PREV_ONLY = "Zero halves warm start time with a persistent compiler cache."
SHORT_OWN = "Alpha adds float8 training to the core library."
NEXT_ONLY = "Beta cuts streaming speech latency on edge phones by forty percent."
FILLER = " ".join(["Beta measured the decoder on three handsets and two boards."] * 60)  # > 2,500 chars

RSS = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Synthetic lab blog</title><link>https://example.test/</link>
<item><title>{TITLE_PREV}</title><link>https://example.test/0</link>
<description>{PREV_ONLY}</description></item>
<item><title>{TITLE_SHORT}</title><link>https://example.test/1</link>
<description>{SHORT_OWN}</description></item>
<item><title>{TITLE_NEXT}</title><link>https://example.test/2</link>
<description>{NEXT_ONLY} {FILLER}</description></item>
</channel></rss>
"""

ATOM = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Synthetic lab blog</title><id>urn:example:lab</id>
<entry><title>{TITLE_PREV}</title><link href="https://example.test/0"/><id>urn:example:0</id>
<summary>{PREV_ONLY}</summary></entry>
<entry><title>{TITLE_SHORT}</title><link href="https://example.test/1"/><id>urn:example:1</id>
<summary>{SHORT_OWN}</summary></entry>
<entry><title>{TITLE_NEXT}</title><link href="https://example.test/2"/><id>urn:example:2</id>
<summary>{NEXT_ONLY} {FILLER}</summary></entry>
</feed>
"""

HTML_HEADINGS = f"""<!doctype html><html><head><title>Synthetic lab index</title></head><body>
<h1>Synthetic lab</h1>
<h2>{TITLE_PREV}</h2><p>{PREV_ONLY}</p>
<h2>{TITLE_SHORT}</h2><p>{SHORT_OWN}</p>
<h3>Notes</h3><p>Alpha also fixes two crashes.</p>
<h2>{TITLE_NEXT}</h2><p>{NEXT_ONLY} {FILLER}</p>
</body></html>
"""

HTML_ARTICLES = f"""<!doctype html><html><head><title>Synthetic lab index</title></head><body><main>
<article><h3>{TITLE_PREV}</h3><p>{PREV_ONLY}</p></article>
<article><h3>{TITLE_SHORT}</h3><p>{SHORT_OWN}</p></article>
<article><h3>{TITLE_NEXT}</h3><p>{NEXT_ONLY} {FILLER}</p></article>
</main></body></html>
"""


def shared_sources(body: str):
    """Three items that all carry the same captured document (one feed or one index page)."""
    items = [{"id": "doc-1", "key": title, "title": title, "text": "parsed text", "body": body,
              "url": f"https://example.test/{n}", "retrieved_at": FETCHED_AT}
             for n, title in enumerate(TITLES)]
    return capture_sources({"lab": {"label": "Lab", "items": items}})


def brief(ref):
    return {"items": [{"insight": "A release can ship a capability in a few words.",
                       "sources": [{"id": "lab", "ref": ref, "label": ref}]}]}


def supported_by(span):
    return {0: {"verdict": "supported", "relation": "says",
                "evidence_spans": [{"citation_index": 0, "support_span": span}]}}


class _ArticleEndCases:
    BODY = ""

    def window(self, ref):
        window = evidence_candidates(brief(ref), shared_sources(self.BODY)).get((0, 0))
        self.assertIsNotNone(window, f"no evidence window for {ref!r}")
        return window

    def test_short_item_window_holds_its_own_text_only(self):
        window = self.window(TITLE_SHORT)
        self.assertIn(SHORT_OWN, window)
        self.assertNotIn(NEXT_ONLY, window, "the window ran past the cited item into the NEXT one")
        self.assertNotIn(TITLE_NEXT, window)
        self.assertNotIn(PREV_ONLY, window, "the window reached back into the PREVIOUS item")

    def test_behind_side_stops_at_the_items_start(self):
        window = self.window(TITLE_NEXT)
        self.assertIn(NEXT_ONLY, window)
        self.assertNotIn(SHORT_OWN, window, "the characters behind reached the previous item")

    def test_support_only_in_the_next_item_is_not_support(self):
        record = build_claim_evidence(brief(TITLE_SHORT), shared_sources(self.BODY), supported_by(NEXT_ONLY))
        (row,) = record["outcomes"]
        self.assertNotEqual(row["support_state"], "supported")
        self.assertEqual(row["evidence_spans"], [])
        self.assertEqual(row["state_reason"], "EVIDENCE-SPAN-NOT-LOCATED")

    def test_support_inside_the_short_item_is_still_support(self):
        record = build_claim_evidence(brief(TITLE_SHORT), shared_sources(self.BODY), supported_by(SHORT_OWN))
        (row,) = record["outcomes"]
        self.assertEqual(row["support_state"], "supported")
        self.assertEqual(row["state_reason"], "EVIDENCE-SPANS-LOCATED")
        (span,) = row["evidence_spans"]
        self.assertEqual(normalize(NORMALIZER_ID, self.BODY.encode("utf-8"))[span["start"]:span["end"]], SHORT_OWN)


class RssArticleEndTests(_ArticleEndCases, unittest.TestCase):
    BODY = RSS


class AtomArticleEndTests(_ArticleEndCases, unittest.TestCase):
    BODY = ATOM


class HtmlHeadingArticleEndTests(_ArticleEndCases, unittest.TestCase):
    BODY = HTML_HEADINGS

    def test_a_lower_heading_inside_the_article_does_not_end_it(self):
        self.assertIn("Alpha also fixes two crashes.", self.window(TITLE_SHORT))


class HtmlArticleElementEndTests(_ArticleEndCases, unittest.TestCase):
    BODY = HTML_ARTICLES


class UnboundedShapesTests(unittest.TestCase):
    """What the clip leaves alone."""

    def test_a_long_item_keeps_the_full_window_ahead(self):
        window = evidence_candidates(brief(TITLE_NEXT), shared_sources(RSS))[(0, 0)]
        ix = RSS.index(TITLE_NEXT)
        item_start = RSS.rindex("<item>", 0, ix)
        # the item runs past the characters ahead, so only the behind side is clipped
        self.assertEqual(window, RSS[item_start:ix + EVIDENCE_WINDOW_AHEAD])

    def test_text_with_no_article_markup_is_not_clipped(self):
        text = "x" * 400 + "Plain title here" + "y" * 3000
        self.assertEqual(evidence_window(text, "Plain title here", 300, 2500), text[100:400 + 2500])

    def test_article_bounds_shapes(self):
        article_bounds = ev.article_bounds
        doc = "<feed><entry><title>A</title></entry><entry><title>B</title><summary>x</summary></entry></feed>"
        a, b = doc.index(">A<") + 1, doc.index(">B<") + 1
        self.assertEqual(doc[slice(*article_bounds(doc, a))], "<entry><title>A</title></entry>")
        self.assertEqual(doc[slice(*article_bounds(doc, b))], "<entry><title>B</title><summary>x</summary></entry>")
        self.assertEqual(article_bounds(doc, 0), (0, len(doc)))   # outside every item: not clipped
        page = "<h1>T</h1><p>a</p><h2>s</h2><p>b</p><h1>U</h1>"
        self.assertEqual(page[slice(*article_bounds(page, page.index("T")))],
                         "<h1>T</h1><p>a</p><h2>s</h2><p>b</p>")


if __name__ == "__main__":
    unittest.main()
