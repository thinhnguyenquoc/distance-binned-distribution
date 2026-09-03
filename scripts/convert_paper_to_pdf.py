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


def protect_and_convert_markdown(md_text: str, base_dir: Path) -> str:
    """
    Protects math blocks from markdown parser transformations,
    converts to HTML, and fixes relative image links.
    """
    # 1. Protect block math $$ ... $$
    block_math_store: list[str] = []

    def block_math_replacer(match: re.Match) -> str:
        idx = len(block_math_store)
        block_math_store.append(match.group(0))
        return f"\n\n<div class='math-block'>@@DISPLAY_MATH_{idx}@@</div>\n\n"

    protected = re.sub(r"\$\$(.*?)\$\$", block_math_replacer, md_text, flags=re.DOTALL)

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

    # 4. Restore math
    for idx, math_content in enumerate(block_math_store):
        html = html.replace(f"@@DISPLAY_MATH_{idx}@@", math_content)

    for idx, math_content in enumerate(inline_math_store):
        html = html.replace(f"@@INLINE_MATH_{idx}@@", math_content)

    # 5. Fix relative image paths to absolute file:/// URLs
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
    temp_html_path = output_path.with_suffix(".html")

    print(f"Reading Markdown: {input_path}")
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