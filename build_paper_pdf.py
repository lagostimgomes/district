"""
Convert research_paper_nature.md → research_paper_nature.pdf
using Python-Markdown + WeasyPrint.

Run:  python build_paper_pdf.py
Requires: python generate_paper_figures.py to have been run first.
"""
import re, base64
from pathlib import Path
import markdown
from weasyprint import HTML, CSS

SRC      = Path("research_paper_nature.md")
CSS_FILE = Path("paper_style.css")
OUT      = Path("docs/research_paper_nature.pdf")
FIGS_DIR = Path("figures")

def img_tag(path: Path, alt: str = "", width: str = "100%") -> str:
    """Embed a PNG as a base64 data URI so WeasyPrint resolves it without
    needing access to the filesystem at render time."""
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<img src="data:image/png;base64,{data}" '
            f'alt="{alt}" style="width:{width};display:block;'
            f'margin:0 auto 4pt auto;page-break-inside:avoid;"/>')

md_text = SRC.read_text(encoding="utf-8")

# ── Pre-process Markdown ────────────────────────────────────────────────────

# Wrap the Abstract paragraph in a styled box
md_text = re.sub(
    r'\*\*Abstract\*\*\n\n(.*?)(?=\n\n---)',
    lambda m: (
        '**Abstract**\n\n'
        '<div class="abstract-box">\n\n'
        + m.group(1) +
        '\n\n</div>'
    ),
    md_text,
    flags=re.DOTALL
)

# Wrap **Keywords** line
md_text = re.sub(
    r'\*\*Keywords\*\*:(.*)',
    r'<p class="keywords"><strong>Keywords:</strong>\1</p>',
    md_text
)

# Convert the equation block to a styled pre
md_text = md_text.replace(
    "w(e) = base × W\\_SAME\\_COUNTY^β × W\\_SAME\\_PLACE^β × W\\_SAME\\_COUSUB^β / W\\_ROAD^β",
    "`w(e) = base × W_SAME_COUNTY^β × W_SAME_PLACE^β × W_SAME_COUSUB^β / W_ROAD^β`"
)

# ── Inject figure images BEFORE the bold→<strong> conversion ────────────────
FIG_MAP = {
    "Fig. 1": FIGS_DIR / "fig1.png",
    "Fig. 2": FIGS_DIR / "fig2.png",
    "Fig. 3": FIGS_DIR / "fig3.png",
    "Fig. 4": FIGS_DIR / "fig4.png",
}

def inject_fig(m):
    fig_key  = m.group(1)   # e.g. "Fig. 1"
    rest     = m.group(2)   # " | National comparison..."
    img_path = FIG_MAP.get(fig_key)
    if img_path and img_path.exists():
        tag = img_tag(img_path, alt=fig_key)
        # Raw HTML block (blank lines required for Markdown to pass through)
        # then the caption re-rendered as bold
        return f'\n\n<div class="figure-block">\n{tag}\n</div>\n\n**{fig_key}{rest}**'
    return m.group(0)

md_text = re.sub(
    r'\*\*(Fig\. \d+)(\s*\|[^*]+)\*\*',
    inject_fig,
    md_text
)

# Convert remaining figure/extended-data legend bold spans → <strong>
md_text = re.sub(
    r'\*\*((?:Fig\.|Extended Data Fig\.)\s*\d+\s*\|.*?)\*\*',
    r'<strong>\1</strong>',
    md_text
)

# Extended Data heading anchor
md_text = md_text.replace(
    "## Extended Data",
    "## Extended Data {#extended-data}"
)

# ── Render Markdown → HTML ──────────────────────────────────────────────────
body_html = markdown.markdown(
    md_text,
    extensions=[
        "tables",
        "attr_list",
        "fenced_code",
        "nl2br",
        "sane_lists",
        "md_in_html",   # allows markdown inside raw HTML blocks
    ]
)

# Post-process: wrap figure legend paragraphs
body_html = re.sub(
    r'(<p><strong>(?:Fig\.|Extended Data Fig\.)\s*\d+\s*\|.*?</p>)',
    r'<div class="figure-legend">\1</div>',
    body_html,
    flags=re.DOTALL
)

# ── Full HTML document ──────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Geography determines partisan representation</title>
</head>
<body>
{body_html}
</body>
</html>"""

# ── Render to PDF ───────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)

css = CSS(filename=str(CSS_FILE))
HTML(string=html, base_url=str(Path.cwd())).write_pdf(
    str(OUT),
    stylesheets=[css],
    presentational_hints=True,
)

print(f"✓  PDF written → {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
