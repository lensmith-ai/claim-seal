"""The judge's window finds a cited title in every written form, and never borrows another item's.

A feed writes a title as XML text: "What? <br> Data" arrives as "What? &lt;br&gt; Data", "R & D" as
"R &amp; D", a line break may be "&#10;", and a feed may escape twice ("&amp;amp;"). The window used
to search for the parsed title verbatim and, on a miss, start at offset 0, which in a feed is the
FIRST item: a claim was judged against an unrelated article. These pin the fix: the exact written
forms first, then the plain form on both sides, and no window at all when the title is not there.

The feed below is synthetic (example.test hosts, invented titles and text).
"""
from __future__ import annotations

import copy
import unittest

import helpers  # noqa: F401  (puts the repo on sys.path)

from claim_seal.evidence import build_claim_evidence, capture_sources, evidence_candidates, plain_title
from claim_seal.resolve import resolve

FETCHED_AT = "2025-01-15T09:30:00Z"
FIRST_DESCRIPTION = ("Unrelated first item. Filler about tuning compute kernels for a new chip, long enough "
                     "that a window anchored on the second title does not reach back to the first title. "
                     "Tile sizes, memory layouts, launch parameters and benchmark tables would follow here; "
                     "this sentence stands in for them, and so does this one: tile sizes, memory layouts, "
                     "launch parameters and benchmark tables.")
FEED = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Synthetic lab blog</title>
<item><title>Post one: kernels on new silicon</title>
<description>&lt;p&gt;{FIRST_DESCRIPTION}&lt;/p&gt;</description>
<link>https://example.test/1</link></item>
<item><title>Post two: free &amp; cheap models? &lt;br&gt; Data systems for agents</title>
<description>&lt;p&gt;When model calls are cheap, the data systems around agents decide quality.&lt;/p&gt;</description>
<link>https://example.test/2</link></item>
<item><title>Post three: line one&#10;line two &amp;amp; more</title>
<description>&lt;p&gt;Third item evidence about numeric line breaks in titles.&lt;/p&gt;</description>
<link>https://example.test/3</link></item>
</channel></rss>
"""
# the titles as a feed parser hands them over (entities decoded once), and the clean keys a brief cites
PARSED = ["Post one: kernels on new silicon",
          "Post two: free & cheap models? <br> Data systems for agents",
          "Post three: line one\nline two &amp; more"]
KEYS = ["Post one: kernels on new silicon",
        "Post two: free & cheap models? Data systems for agents",
        "Post three: line one line two & more"]


def feed_sources(titles=PARSED):
    items = [{"id": "feed-response-1", "key": key, "title": title, "text": "parsed text", "body": FEED,
              "url": f"https://example.test/{n + 1}", "retrieved_at": FETCHED_AT}
             for n, (key, title) in enumerate(zip(KEYS, titles))]
    return capture_sources({"feed": {"label": "Feed", "items": items}})


def brief(ref):
    return {"items": [{"insight": "Cheap model calls make data systems decide quality.",
                       "sources": [{"id": "feed", "ref": ref, "label": ref}]}]}


class TitleAnchorTests(unittest.TestCase):
    def window(self, ref, sources=None):
        return evidence_candidates(brief(ref), sources or feed_sources()).get((0, 0))

    def test_escaped_line_break_and_ampersand_anchor_on_the_cited_item(self):
        window = self.window(KEYS[1])
        self.assertIsNotNone(window)
        self.assertIn("data systems around agents decide quality", window)
        self.assertNotIn("<?xml", window, "the window starts at the head of the feed")
        self.assertNotIn(KEYS[0], window, "the window shows the feed's first item, not the cited one")

    def test_numeric_line_break_and_double_escaped_ampersand_anchor_by_plain_form(self):
        self.assertNotIn(PARSED[2], FEED)
        window = self.window(KEYS[2])
        self.assertIsNotNone(window)
        self.assertIn("Third item evidence about numeric line breaks", window)
        self.assertNotIn(KEYS[0], window)

    def test_a_title_the_feed_does_not_carry_gets_no_window_and_is_unresolved(self):
        titles = list(PARSED)
        titles[1] = "A title this feed never carried"
        sources = feed_sources(titles)
        self.assertIsNone(self.window(KEYS[1], sources))
        rec = build_claim_evidence(brief(KEYS[1]), sources, {0: {"verdict": "supported", "evidence_spans": [
            {"citation_index": 0, "support_span": "Unrelated first item."}]}})
        row = rec["outcomes"][0]
        self.assertEqual((row["support_state"], row["state_reason"]), ("unresolved", "EVIDENCE-TITLE-NOT-FOUND"))
        self.assertEqual(row["evidence_spans"], [])

    def test_plain_title_form(self):
        self.assertEqual(plain_title("Free &amp;amp; cheap?&lt;br&gt;\n Data <i>x</i>"), "Free & cheap? Data x")


class ResolveEscapedReferenceTests(unittest.TestCase):
    def test_escaped_reference_forms_resolve_exactly_and_an_unknown_one_stays_unresolved(self):
        sources = {"feed": {"label": "Feed", "items": [
            {"title": "Unrelated first post", "text": "Kernel tuning."},
            {"title": "Free & cheap models? Data systems", "text": "Data decides."}]}}
        doc = {"items": [{"insight": "c", "sources": [
            {"id": "feed", "ref": "Free &amp; cheap models? &lt;br&gt; Data systems"},
            {"id": "feed", "ref": "Free & cheap models? <br>\nData systems"},
            {"id": "feed", "ref": "A title nobody published"}]}]}
        cites = resolve(doc, sources)["claims"][0]["citations"]
        self.assertEqual([c["resolution"] for c in cites], ["exact", "exact", "unresolved"])
        self.assertEqual({c["matched_title"] for c in cites[:2]}, {"Free & cheap models? Data systems"})
        self.assertIsNone(cites[2]["matched_title"])


if __name__ == "__main__":
    unittest.main()
