"""Convert a markdown file to a styled PDF, resolving relative image paths.

Usage:
    python blog/scripts/md_to_pdf.py blog/01_player_accuracy.md blog/pdf/01_player_accuracy.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from weasyprint import CSS, HTML


CSS_BODY = """
@page {
    size: A4;
    margin: 1.8cm 1.6cm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #888;
    }
}
html { font-size: 11pt; }
body {
    font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
    line-height: 1.5;
    color: #222;
    max-width: 100%;
}
h1, h2, h3, h4 {
    font-family: -apple-system, "Helvetica Neue", Helvetica, sans-serif;
    color: #111;
    line-height: 1.25;
    page-break-after: avoid;
}
h1 { font-size: 1.9em; margin-top: 0; border-bottom: 2px solid #222; padding-bottom: 6px; }
h2 { font-size: 1.4em; margin-top: 1.4em; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
h3 { font-size: 1.15em; margin-top: 1.1em; }
h4 { font-size: 1.0em; margin-top: 1em; }
p { margin: 0.5em 0; text-align: justify; }
code, pre {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.92em;
    background: #f5f5f5;
    border-radius: 3px;
    padding: 1px 4px;
}
pre { padding: 8px 10px; overflow-x: auto; }
pre code { background: transparent; padding: 0; }
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0.6em auto;
    page-break-inside: avoid;
}
em { font-style: italic; }
em img + em, p > em:first-child { display: block; text-align: center; font-size: 0.92em; color: #555; }
blockquote {
    margin: 0.8em 0;
    padding: 0.4em 1em;
    border-left: 3px solid #c0c0c0;
    background: #f9f9f9;
    color: #444;
}
table {
    border-collapse: collapse;
    margin: 0.8em 0;
    font-size: 0.92em;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #ccc;
    padding: 4px 8px;
    text-align: left;
    vertical-align: top;
}
th { background: #f0f0f0; font-weight: bold; }
sup {
    font-size: 0.7em;
    vertical-align: super;
}
hr { border: 0; border-top: 1px solid #ccc; margin: 1.4em 0; }
a { color: #1a5fb4; text-decoration: none; }
"""


def main(src_path: str, out_path: str) -> None:
    src = Path(src_path).resolve()
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    md_text = src.read_text(encoding="utf-8")

    html_body = markdown.markdown(
        md_text,
        extensions=[
            "extra",          # tables, footnotes, etc.
            "fenced_code",
            "codehilite",
            "sane_lists",
            "smarty",
        ],
        extension_configs={
            "codehilite": {"guess_lang": False, "noclasses": False},
        },
    )

    title = src.stem.replace("_", " ").title()
    html_full = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title></head>
<body>{html_body}</body></html>"""

    HTML(string=html_full, base_url=str(src.parent)).write_pdf(
        target=str(out),
        stylesheets=[CSS(string=CSS_BODY)],
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
