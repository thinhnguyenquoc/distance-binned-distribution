"""
Script: convert_paper_to_pdf.py
Converts academic paper Markdown files (e.g. paper/full_paper_en.md, paper/full_paper_vi.md)
to publication-quality PDF documents with full LaTeX math typesetting, booktabs-style tables,
and embedded vector/raster figures.

Requirements:
    - markdown
    - playwright (or system Chrome/Edge as fallback)

Usage:
    python scripts/convert_paper_to_pdf.py --input paper/full_paper_en.md --output paper/full_paper_en.pdf
    python scripts/convert_paper_to_pdf.py --input paper/full_paper_vi.md --output paper/full_paper_vi.pdf
    python scripts/convert_paper_to_pdf.py --all
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    print("Error: 'markdown' package is required. Install via: pip install markdown")
    sys.exit(1)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="{lang}">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    
    <!-- MathJax 3 Configuration -->
    <script>
    window.MathJax = {{
        tex: {{
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true,
            packages: {{'[+]': ['ams', 'color']}}
        }},
        options: {{
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
        }},
        svg: {{
            fontCache: 'global'
        }},
        startup: {{
            typeset: true
        }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/{math_script}"></script>

    <style>
        @page {{
            size: A4;
            margin: 22mm 18mm 22mm 18mm;
            @bottom-center {{
                content: counter(page);
                font-family: 'Times New Roman', 'Cambria', serif;
                font-size: 9pt;
                color: #555;
            }}
        }}

        body {{
            font-family: 'Cambria', 'Georgia', 'Times New Roman', serif;
            font-size: 10.5pt;
            line-height: 1.55;
            color: #1a1a1a;
            text-align: justify;
            text-justify: inter-word;
            margin: 0;
            padding: 0;
        }}

        /* Document Title and Headings */
        h1 {{
            font-size: 19pt;
            line-height: 1.3;
            text-align: center;
            margin-top: 0;
            margin-bottom: 1.2rem;
            color: #111;
            font-weight: bold;
            page-break-after: avoid;
            break-after: avoid;
        }}

        h2 {{
            font-size: 14pt;
            line-height: 1.35;
            margin-top: 1.8rem;
            margin-bottom: 0.6rem;
            color: #222;
            border-bottom: 1px solid #ddd;
            padding-bottom: 3px;
            page-break-after: avoid;
            break-after: avoid;
        }}

        h3 {{
            font-size: 11.5pt;
            line-height: 1.35;
            margin-top: 1.3rem;
            margin-bottom: 0.4rem;
            color: #333;
            font-weight: bold;
            page-break-after: avoid;
            break-after: avoid;
        }}

        h4 {{
            font-size: 10.5pt;
            line-height: 1.35;
            margin-top: 1rem;
            margin-bottom: 0.3rem;
            color: #444;
            font-style: italic;
            page-break-after: avoid;
            break-after: avoid;
        }}

        p {{
            margin-top: 0.4rem;
            margin-bottom: 0.6rem;
            text-indent: 1.5em;
        }}

        p:first-of-type, h1 + p, h2 + p, h3 + p, h4 + p, hr + p {{
            text-indent: 0;
        }}

        hr {{
            border: 0;
            height: 1px;
            background: #e0e0e0;
            margin: 1.5rem 0;
        }}

        /* Academic Booktabs Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.2rem 0;
            font-size: 9pt;
            line-height: 1.4;
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        table caption {{
            font-weight: bold;
            text-align: left;
            margin-bottom: 0.5rem;
            font-size: 9.5pt;
            color: #222;
        }}

        thead tr {{
            border-top: 1.5pt solid #111;
            border-bottom: 1pt solid #111;
        }}

        th {{
            padding: 6px 8px;
            font-weight: bold;
            text-align: left;
            background-color: #fafafa;
        }}

        td {{
            padding: 5px 8px;
            vertical-align: top;
            border-bottom: 0.5pt solid #e5e5e5;
        }}

        tbody tr:last-child td {{
            border-bottom: 1.5pt solid #111;
        }}

        /* Figures & Images */
        figure, .figure-container {{
            margin: 1.5rem auto;
            text-align: center;
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        img {{
            max-width: 96%;
            height: auto;
            display: block;
            margin: 0 auto 0.6rem auto;
            border: 0.5pt solid #e0e0e0;
            border-radius: 2px;
        }}

        img[src$=".svg"] {{
            border: none;
        }}

        figcaption, .caption {{
            font-size: 9pt;
            line-height: 1.4;
            color: #333;
            margin-top: 0.4rem;
            margin-bottom: 1rem;
            text-align: justify;
            text-indent: 0;
        }}

        /* Math styling */
        mjx-container {{
            font-size: 105% !important;
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        mjx-container[display="true"] {{
            margin: 0.8rem 0 !important;
        }}

        /* Blockquotes and Callouts */
        blockquote {{
            border-left: 3px solid #2b6cb0;
            margin: 1rem 0;
            padding: 0.5rem 1rem;
            background-color: #f7fafc;
            font-size: 9.5pt;
            color: #2d3748;
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        blockquote p {{
            text-indent: 0;
            margin: 0.3rem 0;
        }}

        /* Code Blocks */
        pre, code {{
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 8.5pt;
        }}

        code {{
            background-color: #f3f4f6;
            padding: 1px 4px;
            border-radius: 3px;
            color: #b91c1c;
        }}

        /* Citation Links */
        a.citation-link {{
            color: #1a365d;
            text-decoration: none;
            font-weight: 500;
        }}
        a.citation-link:hover {{
            text-decoration: underline;
            color: #2b6cb0;
        }}

        pre {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 8px 12px;
            overflow-x: auto;
            page-break-inside: avoid;
            break-inside: avoid;
        }}

        pre code {{
            background: none;
            padding: 0;
            color: #1e293b;
        }}

        /* Lists */
        ul, ol {{
            margin: 0.5rem 0 0.8rem 1.5rem;
            padding: 0;
        }}

        li {{
            margin-bottom: 0.25rem;
        }}
    </style>
</head>
<body>
{body}
</body>
</html>
"""


def load_bibliography(bib_path: Path) -> dict[str, int]:
    """Loads BibTeX entries from file and returns mapping from key to 1-based index."""
    if not bib_path.exists():
        return {}
    bib_text = bib_path.read_text(encoding="utf-8")
    keys = re.findall(r"@\w+\s*\{\s*([^,]+),", bib_text)
    return {k.strip(): idx for idx, k in enumerate(keys, 1)}


def protect_and_convert_markdown(md_text: str, base_dir: Path) -> str:
    """
    Protects math blocks and citations from markdown parser transformations,
    converts to HTML, and fixes relative image links.
    """
    # 0. Load bibliography and protect citations
    bib_candidates = [
        base_dir / "references.bib",
        base_dir / "paper" / "references.bib",
        base_dir.parent / "references.bib",
        base_dir.parent / "paper" / "references.bib",
        Path("paper/references.bib"),
    ]
    bib_path = next((p for p in bib_candidates if p.exists()), None)
    key_to_idx = load_bibliography(bib_path) if bib_path else {}

    citation_store: list[str] = []

    def citation_replacer(match: re.Match) -> str:
        raw_content = match.group(1)
        raw_keys = [k.strip().lstrip("@").strip() for k in raw_content.split(";")]
        nums = []
        for k in raw_keys:
            if not k:
                continue
            if k not in key_to_idx:
                raise ValueError(f"Undefined citation key: '{k}' in '[@{raw_content}]'")
            nums.append(key_to_idx[k])
        
        sorted_nums = sorted(list(set(nums)))
        links = [f'<a href="#ref-{n}" class="citation-link">{n}</a>' for n in sorted_nums]
        formatted = f"[{', '.join(links)}]"
        idx = len(citation_store)
        citation_store.append(formatted)
        return f"@@CITATION_{idx}@@"

    if key_to_idx:
        protected = re.sub(r"\[@([^\]]+)\]", citation_replacer, md_text)
    else:
        protected = md_text

    # 1. Protect block math $$ ... $$
    block_math_store: list[str] = []

    def block_math_replacer(match: re.Match) -> str:
        idx = len(block_math_store)
        block_math_store.append(match.group(0))
        return f"\n\n<div class='math-block'>@@DISPLAY_MATH_{idx}@@</div>\n\n"

    protected = re.sub(r"\$\$(.*?)\$\$", block_math_replacer, protected, flags=re.DOTALL)

    # 2. Protect inline math $ ... $
    inline_math_store: list[str] = []

    def inline_math_replacer(match: re.Match) -> str:
        idx = len(inline_math_store)
        inline_math_store.append(match.group(0))
        return f"@@INLINE_MATH_{idx}@@"

    # Match $...$ where $ is not preceded by backslash and content is not empty
    protected = re.sub(r"(?<!\\)\$(?!\s)(.+?)(?<!\s)(?<!\\)\$", inline_math_replacer, protected)

    # 3. Convert markdown to HTML
    html = markdown.markdown(
        protected,
        extensions=[
            "tables",
            "fenced_code",
            "toc",
            "attr_list",
            "def_list",
            "sane_lists",
        ],
    )

    # 4. Restore citations and math
    for idx, cite_content in enumerate(citation_store):
        html = html.replace(f"@@CITATION_{idx}@@", cite_content)

    for idx, math_content in enumerate(block_math_store):
        html = html.replace(f"@@DISPLAY_MATH_{idx}@@", math_content)

    for idx, math_content in enumerate(inline_math_store):
        html = html.replace(f"@@INLINE_MATH_{idx}@@", math_content)

    # 5. Add id="ref-{n}" to each <li> in References / Tài liệu tham khảo
    def add_reference_ids(html_str: str) -> str:
        m = re.search(
            r"(<h[1-3][^>]*>.*?(?:Tài liệu tham khảo|References).*?</h[1-3]>\s*<ol>)(.*?)(</ol>)",
            html_str,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not m:
            return html_str
        header_ol = m.group(1)
        items_block = m.group(2)
        end_ol = m.group(3)
        idx = 1

        def li_replacer(lim: re.Match) -> str:
            nonlocal idx
            res = f'<li id="ref-{idx}">'
            idx += 1
            return res

        new_items = re.sub(r"<li>", li_replacer, items_block)
        return html_str[:m.start()] + header_ol + new_items + end_ol + html_str[m.end():]

    html = add_reference_ids(html)

    # 6. Fix relative image paths to absolute file:/// URLs
    def img_url_replacer(match: re.Match) -> str:
        src = match.group(1)
        if not src.startswith("http://") and not src.startswith("https://") and not src.startswith("data:"):
            img_path = (base_dir / src).resolve()
            if img_path.exists():
                return f'src="{img_path.as_uri()}"'
        return match.group(0)

    html = re.sub(r'src=["\']([^"\']+)["\']', img_url_replacer, html)

    # Wrap figures and following bold captions into a figure block for clean layout
    def figure_caption_replacer(match: re.Match) -> str:
        img_tag = match.group(1)
        caption_content = match.group(2)
        return (
            f"<figure class='figure-container'>{img_tag}"
            f"<figcaption class='caption'>{caption_content}</figcaption></figure>"
        )

    html = re.sub(
        r"<p>(<img[^>]+>)</p>\s*<p>(<strong>(?:Figure|Hình|Table|Bảng)[^<]+</strong>.*?)</p>",
        figure_caption_replacer,
        html,
        flags=re.DOTALL,
    )

    return html


def render_html_to_pdf_playwright(html_path: Path, output_pdf_path: Path) -> None:
    """Renders an HTML file to PDF via Playwright Chromium."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Load file
        file_url = html_path.resolve().as_uri()
        page.goto(file_url, wait_until="networkidle")

        # Wait for MathJax to finish typesetting if present
        try:
            page.wait_for_function(
                "() => !window.MathJax || !window.MathJax.startup || window.MathJax.startup.document.getMathML ? true : true",
                timeout=15000,
            )
            # Give MathJax a moment to finish rendering complex SVGs
            page.wait_for_timeout(2500)
        except Exception as e:
            print(f"Warning during MathJax wait: {e}")

        # Print PDF
        page.pdf(
            path=str(output_pdf_path),
            format="A4",
            print_background=True,
            margin={
                "top": "22mm",
                "bottom": "22mm",
                "left": "18mm",
                "right": "18mm",
            },
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                '<div style="font-family: \'Cambria\', serif; font-size: 8.5pt; width: 100%; '
                'text-align: center; color: #666;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>'
            ),
        )

        browser.close()


def render_html_to_pdf_chrome(html_path: Path, output_pdf_path: Path) -> None:
    """Fallback: Renders an HTML file to PDF via system Chrome/Edge CLI."""
    import subprocess
    import time

    candidates = [
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    ]
    browser_exe = next((p for p in candidates if p.exists()), None)
    if not browser_exe:
        raise RuntimeError("Neither Playwright nor Chrome/Edge executable found on system.")

    file_url = html_path.resolve().as_uri()
    cmd = [
        str(browser_exe),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf_path.resolve()}",
        file_url,
    ]
    time.sleep(1)
    subprocess.run(cmd, check=True)


def render_markdown_to_pdf_pandoc(input_path: Path, output_pdf_path: Path) -> bool:
    """Renders a Markdown academic paper to PDF via Pandoc + XeLaTeX."""
    import os
    import shutil
    import subprocess

    env = dict(os.environ)
    current_path = env.get("PATH", "")
    tex_paths = ["/Library/TeX/texbin", "/opt/homebrew/bin", "/usr/local/bin"]
    for p in tex_paths:
        if p not in current_path and os.path.exists(p):
            current_path = f"{p}:{current_path}"
    env["PATH"] = current_path

    pandoc_exe = shutil.which("pandoc", path=env["PATH"])
    xelatex_exe = shutil.which("xelatex", path=env["PATH"])

    if not (pandoc_exe and xelatex_exe):
        return False

    bib_candidates = [
        input_path.parent / "references.bib",
        Path("paper/references.bib"),
    ]
    bib_file = next((b for b in bib_candidates if b.exists()), None)

    cmd = [
        pandoc_exe,
        "-f", "markdown+tex_math_single_backslash",
        str(input_path.resolve()),
        f"--resource-path=.:{input_path.parent.resolve()}:{(input_path.parent / 'figures').resolve()}",
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=20mm",
        "-V", "mainfont=Times New Roman",
        "-V", "papersize=a4",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue",
        "-V", "urlcolor=blue",
        "-V", "citecolor=blue",
        "-o", str(output_pdf_path.resolve()),
    ]
    if bib_file:
        cmd.extend(["--citeproc", f"--bibliography={bib_file.resolve()}"])

    try:
        subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Pandoc/XeLaTeX warning: {e.stderr}")
        return False


def convert_paper_to_pdf(
    input_path: Path,
    output_path: Path | None = None,
    keep_html: bool = False,
    math_renderer: str = "chtml",
) -> Path:
    """Converts a Markdown academic paper to PDF."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".pdf")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading Markdown: {input_path}")
    print("Attempting compilation via Pandoc + XeLaTeX...")
    if render_markdown_to_pdf_pandoc(input_path, output_path):
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Success! Generated publication-quality PDF via Pandoc/XeLaTeX: {output_path} ({size_mb:.2f} MB)")
        return output_path

    print("Pandoc/XeLaTeX unavailable or failed; proceeding to HTML + Chromium pipeline...")
    temp_html_path = output_path.with_suffix(".html")
    md_text = input_path.read_text(encoding="utf-8")

    title_match = re.search(r"^#\s+(.+)$", md_text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else input_path.stem
    lang = "vi" if "vi" in input_path.name.lower() else "en"

    math_script = "tex-chtml.js" if math_renderer == "chtml" else "tex-svg.js"
    print(f"Parsing Markdown and typesetting LaTeX math via MathJax ({math_renderer.upper()})...")
    body_html = protect_and_convert_markdown(md_text, base_dir=input_path.parent)
    full_html = HTML_TEMPLATE.format(title=title, lang=lang, body=body_html, math_script=math_script)

    temp_html_path.write_text(full_html, encoding="utf-8")
    print(f"Generated HTML preview: {temp_html_path}")

    print("Rendering PDF via headless Chromium...")
    try:
        render_html_to_pdf_playwright(temp_html_path, output_path)
    except Exception as e:
        print(f"Playwright rendering failed ({e}), attempting system Chrome/Edge fallback...")
        render_html_to_pdf_chrome(temp_html_path, output_path)

    if not keep_html and temp_html_path.exists():
        try:
            temp_html_path.unlink()
        except OSError:
            pass

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Success! Generated PDF: {output_path} ({size_mb:.2f} MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert academic paper Markdown to PDF with MathJax and Playwright.")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("paper/full_paper_en.md"),
        help="Path to input markdown file (default: paper/full_paper_en.md)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to output PDF file (default: same name with .pdf extension)",
    )
    parser.add_argument(
        "--math",
        choices=["chtml", "svg"],
        default="chtml",
        help="MathJax render engine: 'chtml' (compact ~1.5MB using web fonts, default) or 'svg' (pure vector ~4.5MB)",
    )
    parser.add_argument(
        "--keep-html",
        action="store_true",
        help="Retain intermediate HTML file for web viewing or debugging",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert both English and Vietnamese papers (paper/full_paper_en.md and paper/full_paper_vi.md)",
    )

    args = parser.parse_args()

    if args.all:
        for md_file in [Path("paper/full_paper_en.md"), Path("paper/full_paper_vi.md")]:
            if md_file.exists():
                out_pdf = md_file.with_suffix(".pdf")
                convert_paper_to_pdf(md_file, out_pdf, keep_html=args.keep_html, math_renderer=args.math)
    else:
        convert_paper_to_pdf(args.input, args.output, keep_html=args.keep_html, math_renderer=args.math)


if __name__ == "__main__":
    main()