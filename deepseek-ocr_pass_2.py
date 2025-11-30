# deepseek-ocr_pass_2.py
# Pass 2: Uses Ollama deepseek-ocr to re-OCR only "complex" tables
# identified in pass 1, and patches them into the Docling markdown + HTML.

import base64
import io
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import markdown  # pip install markdown
import requests
from pdf2image import convert_from_path
from PIL import Image


# === CONFIGURATION ===
PDF_PATH = Path("/Users/charliegansky/AI/OCR-Finance/MSFT_FY25q4_10K.pdf")
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-ocr"


# ---------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------
def encode_image_to_base64(image: Image.Image, fmt: str = "png") -> str:
    image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format=fmt.upper())
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ollama_generate(image: Image.Image, prompt: str) -> Dict[str, Any]:
    """
    Call Ollama with the given prompt and a single page image.
    Returns the full JSON response.
    """
    b64 = encode_image_to_base64(image)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
    }
    resp = requests.post(OLLAMA_BASE_URL, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def extract_inner_table_html(html_snippet: str) -> str:
    """
    Given an HTML snippet, return only the <table>...</table> part.
    """
    lower = html_snippet.lower()
    start = lower.find("<table")
    if start == -1:
        return html_snippet
    end = lower.find("</table>", start)
    if end == -1:
        return html_snippet
    end += len("</table>")
    return html_snippet[start:end]


def generate_image_variants(image: Image.Image) -> List[Tuple[str, Image.Image]]:
    """
    Create cropped variants of the page to help OCR when the full page fails.
    Modified to prioritize full page and use table-detection friendly crops.
    """
    width, height = image.size
    variants = [("full page", image)]

    # Define crop regions that are more likely to contain tables
    # Instead of simple halves, use overlapping regions that are more likely to contain tables
    candidate_boxes = [
        # Full page regions (with some margin adjustments)
        ("top two-thirds", (0, 0, width, 2 * height // 3)),
        ("bottom two-thirds", (0, height // 3, width, height)),
        ("middle half", (0, height // 4, width, 3 * height // 4)),
        # Left and right sections (in case tables are in specific areas)
        ("left two-thirds", (0, 0, 2 * width // 3, height)),
        ("right two-thirds", (width // 3, 0, width, height)),
        # Quadrants in case table is in a corner
        ("top-left", (0, 0, width // 2, height // 2)),
        ("top-right", (width // 2, 0, width, height // 2)),
        ("bottom-left", (0, height // 2, width // 2, height)),
        ("bottom-right", (width // 2, height // 2, width, height)),
    ]

    seen_boxes = {(0, 0, width, height)}
    for label, (x0, y0, x1, y1) in candidate_boxes:
        x0 = max(0, min(int(x0), width - 1))
        y0 = max(0, min(int(y0), height - 1))
        x1 = max(x0 + 1, min(int(x1), width))
        y1 = max(y0 + 1, min(int(y1), height))

        # Increase minimum area to avoid tiny crops that don't contain full tables
        if (x1 - x0) * (y1 - y0) < 50000:  # Increased from 20000 to 50000
            continue

        box = (x0, y0, x1, y1)
        if box in seen_boxes:
            continue
        seen_boxes.add(box)

        variants.append((label, image.crop(box)))

    return variants


# ---------------------------------------------------------
# Markdown table parsing
# ---------------------------------------------------------
def is_valid_markdown_table_block(lines: List[str]) -> bool:
    """
    Check if the lines form a markdown table block.
    """
    table_lines = [ln for ln in lines if ln.strip().startswith("|")]
    if len(table_lines) < 2:
        return False
    if not any("---" in ln for ln in lines):
        return False
    return True


def extract_markdown_tables(text: str) -> List[str]:
    """
    Extract contiguous markdown tables from a text block.
    """
    lines = text.splitlines()
    tables = []
    current = []

    for line in lines:
        if line.strip().startswith("|"):
            current.append(line.rstrip())
        else:
            if current and is_valid_markdown_table_block(current):
                tables.append("\n".join(current).strip())
                current = []
            else:
                current = []

    if current and is_valid_markdown_table_block(current):
        tables.append("\n".join(current).strip())

    return tables


def table_shape(table_md: str) -> Tuple[int, int]:
    """
    Estimate number of rows and columns in a markdown table.
    """
    rows = [ln for ln in table_md.splitlines() if ln.strip().startswith("|")]
    row_count = len(rows)
    if not rows:
        return 0, 0

    col_counts = [max(len(r.split("|")) - 2, 0) for r in rows]
    col_count = max(col_counts) if col_counts else 0
    return row_count, col_count


def choose_best_table_by_shape(candidates: List[str], orig_table_md: str) -> str:
    """
    Pick the closest table to the Docling structure based on shape.
    Enhanced to prioritize complete tables over cropped ones when possible.
    """
    if not candidates:
        return ""

    o_rows, o_cols = table_shape(orig_table_md)
    if o_rows == 0 or o_cols == 0:
        return candidates[0]

    best_table = None
    best_score = float("-inf")

    for tbl in candidates:
        r, c = table_shape(tbl)
        # Score based on how close the dimensions are to the original
        score = -abs(r - o_rows) - abs(c - o_cols)
        # Penalize tables with fewer columns than original
        if c < o_cols:
            score -= 10
        # Prioritize tables with more rows/columns when original was incomplete
        if r >= o_rows and c >= o_cols:
            score += 5
        if score > best_score:
            best_score = score
            best_table = tbl

    return best_table or ""


def is_likely_complete_table(table_md: str, orig_table_md: str) -> bool:
    """
    Check if the OCR-extracted table is likely more complete than the original.
    This helps determine if we should prefer a result from a full page vs cropped version.
    """
    if not table_md or not orig_table_md:
        return False
        
    new_rows, new_cols = table_shape(table_md)
    orig_rows, orig_cols = table_shape(orig_table_md)
    
    # If the new table has more rows and columns, it's likely more complete
    return new_rows >= orig_rows and new_cols >= orig_cols


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main() -> None:
    pdf_path = PDF_PATH
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    stem = pdf_path.stem
    out_dir = pdf_path.parent / f"{stem}_docling"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_md_path = out_dir / f"{stem}_docling_raw.md"
    raw_html_path = out_dir / f"{stem}_docling_raw.html"
    complex_json_path = out_dir / f"{stem}_complex_tables.json"
    final_md_path = out_dir / f"{stem}_docling_ocr_fixed.md"
    final_html_path = out_dir / f"{stem}_docling_ocr_fixed.html"

    print("=== PASS 2: DEEPSEEK OCR ON COMPLEX TABLES ===")
    print(f"PDF_PATH: {pdf_path}")
    print(f"Output directory: {out_dir}")
    print(f"Expecting complex table metadata at: {complex_json_path}\n")

    if not raw_md_path.exists() or not raw_html_path.exists():
        raise SystemExit(
            "Raw Docling outputs not found.\n"
            f"Expected:\n {raw_md_path}\n {raw_html_path}\n"
            "Run docling_pass_1.py first."
        )

    if not complex_json_path.exists():
        print("Complex table JSON not found. Searching for fallback...")
        candidates = list(out_dir.glob("*_complex_tables.json"))
        if candidates:
            complex_json_path = candidates[0]
            print(f"Using fallback file: {complex_json_path}")
        else:
            raise SystemExit("Complex table metadata not found.")

    markdown_content = raw_md_path.read_text(encoding="utf-8")
    html_content = raw_html_path.read_text(encoding="utf-8")
    complex_tables = json.loads(complex_json_path.read_text(encoding="utf-8"))

    if not complex_tables:
        print("No complex tables. Copying raw output as final.")
        final_md_path.write_text(markdown_content, encoding="utf-8")
        final_html_path.write_text(html_content, encoding="utf-8")
        print("Done.")
        return

    print(f"Re-OCR'ing {len(complex_tables)} complex tables with '{OLLAMA_MODEL}'...")

    print(f"Converting PDF '{pdf_path}' to images...")
    pages = convert_from_path(str(pdf_path), dpi=200)
    print(f"Converted {len(pages)} pages.\n")

    for idx, tbl in enumerate(complex_tables, start=1):
        page_index = tbl.get("page_index")
        heading = (tbl.get("heading") or "").strip() or f"table on page {page_index + 1}"
        orig_md = tbl.get("orig_table_md", "")
        orig_html = tbl.get("orig_table_html", "")

        print(f"[{idx}/{len(complex_tables)}] Page {page_index + 1}, heading '{heading}'")

        if page_index is None or page_index >= len(pages):
            print(" Warning: out-of-range page index, skipping.")
            continue

        page_image = pages[page_index]

        prompts = [
            (
                "heading-aware",
                f"""
You are an OCR engine specialised in financial reports.
From this page image, find the table closest to this heading:
\"\"\"{heading}\"\"\"
Extract ONLY the table and output it as a GitHub-flavoured Markdown table.
No explanations. No extra text.
""".strip(),
            ),
            (
                "simple",
                "Extract the table from this image as a GitHub-flavoured Markdown table. Only output the table."
            )
        ]

        image_variants = generate_image_variants(page_image)
        ocr_success = False
        best_full_page_result = None
        best_crop_result = None

        # First, try all prompts on the full page
        print(" Trying full page...")
        for label, prompt in prompts:
            try:
                resp_json = ollama_generate(page_image, prompt)  # Use full page
            except Exception as exc:
                print(f" Error ({label}, full page): {exc}")
                continue

            text = (resp_json.get("response") or "").strip()
            if not text:
                print(f" Warning: empty OCR output ({label}, full page)")
                continue

            tables = extract_markdown_tables(text)
            if not tables:
                snippet = text.replace("\n", " ")[:200]
                print(f" No tables found ({label}, full page): {snippet!r}")
                continue

            new_table_md = choose_best_table_by_shape(tables, orig_md)
            if not new_table_md:
                print(f" Could not select table ({label}, full page).")
                continue

            # Check if this full page result is likely complete
            if is_likely_complete_table(new_table_md, orig_md):
                best_full_page_result = new_table_md
                print(f" Found likely complete table from full page using {label}")
                # If it's a really good result (same or better dimensions), use it immediately
                if table_shape(new_table_md)[0] >= table_shape(orig_md)[0] and \
                   table_shape(new_table_md)[1] >= table_shape(orig_md)[1]:
                    break

        # If full page gave us a good result, use it
        if best_full_page_result:
            # --- Patch Markdown ---
            if orig_md and orig_md.strip() in markdown_content:
                markdown_content = markdown_content.replace(
                    orig_md.strip(), best_full_page_result.strip()
                )
                print(f" Patched markdown using full page result")
            else:
                print(" Warning: original markdown not found. Skipping MD patch.")

            # --- Patch HTML ---
            if orig_html:
                inner_orig_html = extract_inner_table_html(orig_html).strip()
                if inner_orig_html and inner_orig_html in html_content:
                    try:
                        new_table_html_full = markdown.markdown(
                            best_full_page_result, extensions=["tables"]
                        )
                    except Exception:
                        new_table_html_full = ""
                    inner_new_html = extract_inner_table_html(new_table_html_full)
                    html_content = html_content.replace(
                        inner_orig_html, inner_new_html, 1
                    )
                    print(f" Patched HTML using full page result")
                else:
                    print(" Warning: original HTML table not found. Skipping.")
            else:
                print(" No original HTML snippet available.")

            ocr_success = True
        else:
            # If full page didn't work well, try cropped versions
            for variant_label, variant_image in image_variants:
                if variant_label == "full page":
                    continue  # Already tried above
                print(f" Trying crop: {variant_label}...")

                for label, prompt in prompts:
                    try:
                        resp_json = ollama_generate(variant_image, prompt)
                    except Exception as exc:
                        print(f" Error ({label}, {variant_label}): {exc}")
                        continue

                    text = (resp_json.get("response") or "").strip()
                    if not text:
                        print(f" Warning: empty OCR output ({label}, {variant_label})")
                        continue

                    tables = extract_markdown_tables(text)
                    if not tables:
                        snippet = text.replace("\n", " ")[:200]
                        print(f" No tables found ({label}, {variant_label}): {snippet!r}")
                        continue

                    new_table_md = choose_best_table_by_shape(tables, orig_md)
                    if not new_table_md:
                        print(f" Could not select table ({label}, {variant_label}).")
                        continue

                    # Check if this crop result is likely complete
                    if is_likely_complete_table(new_table_md, orig_md):
                        best_crop_result = new_table_md
                        print(f" Found likely complete table from crop using {label} ({variant_label})")
                        
                        # --- Patch Markdown ---
                        if orig_md and orig_md.strip() in markdown_content:
                            markdown_content = markdown_content.replace(
                                orig_md.strip(), new_table_md.strip()
                            )
                            print(f" Patched markdown using {label} ({variant_label})")
                        else:
                            print(" Warning: original markdown not found. Skipping MD patch.")

                        # --- Patch HTML ---
                        if orig_html:
                            inner_orig_html = extract_inner_table_html(orig_html).strip()
                            if inner_orig_html and inner_orig_html in html_content:
                                try:
                                    new_table_html_full = markdown.markdown(
                                        new_table_md, extensions=["tables"]
                                    )
                                except Exception:
                                    new_table_html_full = ""
                                inner_new_html = extract_inner_table_html(new_table_html_full)
                                html_content = html_content.replace(
                                    inner_orig_html, inner_new_html, 1
                                )
                                print(f" Patched HTML using {label} ({variant_label})")
                            else:
                                print(" Warning: original HTML table not found. Skipping.")
                        else:
                            print(" No original HTML snippet available.")

                        ocr_success = True
                        break
                if ocr_success:
                    break

        if not ocr_success:
            print(" All OCR attempts failed. Leaving Docling table unchanged.")

    final_md_path.write_text(markdown_content, encoding="utf-8")
    print(f"\n✅ Final OCR-fixed markdown saved to: {final_md_path}")

    final_html_path.write_text(html_content, encoding="utf-8")
    print(f"✅ Final OCR-fixed HTML saved to: {final_html_path}")


if __name__ == "__main__":
    main()