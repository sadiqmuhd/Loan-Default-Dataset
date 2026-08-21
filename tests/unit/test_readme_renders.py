"""Structural checks on README.md, so it renders correctly on GitHub.

The README is the first and often only thing a reader sees, and GitHub degrades
silently rather than erroring: a broken Mermaid block becomes "Unable to render
rich display", a truncated link still looks like a link. Neither shows up
locally, which is what makes them worth a test.

Both failure modes checked here are ones this file has actually hit.
"""

from __future__ import annotations

import re

import pytest

from loan_default.config import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
TEXT = README.read_text(encoding="utf-8")

MERMAID_BLOCKS = re.findall(r"```mermaid\n(.*?)```", TEXT, re.S)

# GitHub's Mermaid build renders <br> but chokes on other inline HTML in node
# labels, which is what produced "Unable to render rich display".
ALLOWED_HTML_TAGS = {"br"}


def test_there_is_a_mermaid_diagram():
    assert MERMAID_BLOCKS, "The architecture diagram is missing from README.md"


@pytest.mark.parametrize("index", range(len(MERMAID_BLOCKS)))
def test_mermaid_delimiters_are_balanced(index: int):
    block = MERMAID_BLOCKS[index]
    for opener, closer in (("[", "]"), ("{", "}"), ("(", ")")):
        assert block.count(opener) == block.count(closer), (
            f"Mermaid block {index}: {opener}{closer} unbalanced "
            f"({block.count(opener)} vs {block.count(closer)})"
        )
    assert block.count('"') % 2 == 0, f"Mermaid block {index}: unbalanced quotes"


@pytest.mark.parametrize("index", range(len(MERMAID_BLOCKS)))
def test_mermaid_uses_no_unsupported_inline_html(index: int):
    """`<i>` tags in node labels are what broke the diagram on GitHub."""
    block = MERMAID_BLOCKS[index]
    found = re.findall(r"<(/?[a-zA-Z][a-zA-Z0-9]*)", block)
    tags = {tag.lstrip("/").rstrip("/").lower() for tag in found}
    unsupported = tags - ALLOWED_HTML_TAGS
    assert not unsupported, (
        f"Mermaid block {index} uses inline HTML GitHub may not render: "
        f"{sorted(unsupported)}. Only {sorted(ALLOWED_HTML_TAGS)} is safe."
    )


@pytest.mark.parametrize("index", range(len(MERMAID_BLOCKS)))
def test_mermaid_declares_a_diagram_type(index: int):
    first = next(line.strip() for line in MERMAID_BLOCKS[index].split("\n") if line.strip())
    known = ("flowchart", "graph", "sequenceDiagram", "classDiagram", "stateDiagram", "erDiagram")
    assert first.startswith(known), f"Mermaid block {index} starts with {first!r}"


def test_delimiters_are_balanced_across_the_file():
    """Whole-file balance. Prose may legitimately wrap a parenthetical across
    lines, so a per-line check would false-positive; a whole-file count still
    catches a genuinely dropped character."""
    for opener, closer in (("(", ")"), ("[", "]"), ("{", "}")):
        assert TEXT.count(opener) == TEXT.count(closer), (
            f"{opener}{closer} unbalanced across README.md "
            f"({TEXT.count(opener)} vs {TEXT.count(closer)})"
        )


def test_delimiters_are_balanced_on_every_line_carrying_a_link():
    """Catches truncated markdown links, which is where imbalance does damage.

    The README shipped once with seven dropped ")" characters, which turned
    "[Kaggle](https://...dataset) into" into "[Kaggle](https://...datasetinto" -
    a link whose URL had swallowed the following word. It still rendered as a
    link, just pointing at a 404, so nothing looked wrong until you clicked it.

    Restricted to lines containing a link, because a wrapped parenthetical in
    ordinary prose is fine and only the link case is a silent failure.
    """
    offenders = []
    for number, line in enumerate(TEXT.split("\n"), 1):
        if "](" not in line:
            continue
        for opener, closer in (("(", ")"), ("[", "]")):
            if line.count(opener) != line.count(closer):
                offenders.append(
                    f"line {number}: {opener}{closer} unbalanced - {line.strip()[:70]}"
                )
    assert not offenders, "Unbalanced delimiters on link lines:" + "".join(
        f"\n  {o}" for o in offenders
    )


def test_no_markdown_link_url_runs_into_following_text():
    """A URL whose final path segment ends in a common English word is usually a
    dropped closing bracket rather than a real address."""
    suspicious = []
    for url in re.findall(r"\]\((https?://[^)\s]+)\)", TEXT):
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        for word in ("into", "and", "the", "with", "from"):
            if tail.endswith(word) and not tail.endswith(f"-{word}"):
                suspicious.append(f"{url} (ends with {word!r})")
    assert not suspicious, "URLs that look truncated:" + "".join(f"\n  {s}" for s in suspicious)


def test_every_local_image_exists():
    """A broken image is invisible until someone opens the page."""
    missing = []
    for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", TEXT):
        if path.startswith(("http://", "https://")):
            continue
        if not (PROJECT_ROOT / path).exists():
            missing.append(path)
    assert not missing, "README references images that do not exist: " + ", ".join(missing)


def test_every_relative_link_resolves():
    missing = []
    for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", TEXT):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.split("#")[0]
        if path and not (PROJECT_ROOT / path).exists():
            missing.append(target)
    assert not missing, "README links to paths that do not exist: " + ", ".join(missing)


def test_code_fences_are_balanced():
    assert TEXT.count("```") % 2 == 0, "Unbalanced code fences will swallow the rest of the page"


def test_no_leaked_placeholder_text():
    """Guards against shipping a template value in the most-read file."""
    for placeholder in ("your-app.up.railway.app", "TODO", "FIXME", "<your-"):
        assert placeholder not in TEXT, f"README still contains placeholder {placeholder!r}"
