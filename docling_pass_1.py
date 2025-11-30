# docling_pass_1.py
# Pass 1: Converts a PDF to Markdown/HTML using IBM Granite Docling MLX model,
# and identifies "complex" tables that may need re-OCR in a second pass.

import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from pdf2image import convert_from_path
from transformers.image_utils import load_image

from PIL import Image  # noqa: F401  (kept for type hints / future use)

# === CONFIGURATION ===
MODEL_PATH = "ibm-granite/granite-docling-258M-mlx"
PROMPT = "Convert this page to docling."
PDF_PATH = Path("/Users/charliegansky/AI/OCR-Finance/MSFT_FY25q4_10K.pdf")
SHOW_IN_BROWSER = False

# Simple guard so tiny tables are ignored
MIN_COMPLEX_ROWS = 3


def is_complex_table_from_markdown(table_md: str) -> bool:
    """
    Decide whether a table should be re-OCR'd.

    New heuristic:
      - Parse markdown into cells.
      - A table is 'complex' only if there exists at least one cell that:
          * contains 2 or more numeric values, AND
          * has no alphabetic characters (so dates like 'September 30, 2025'
            do not trigger it).

    This catches jammed numeric cells like:
        '2024 247,442 8,308 11.82'
    but ignores simple two-column tables and date labels.
    """
    import re

    # Keep only non-empty lines
    lines = [ln.rstrip() for ln in table_md.splitlines() if ln.strip()]
    if not lines:
        return False

    # Consider only table rows that start with '|' and are not the separator row
    # (the one full of '---').
    data_rows = [
        ln for ln in lines
        if ln.lstrip().startswith("|") and "---" not in ln
    ]
    if not data_rows:
        return False

    num_pattern = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")

    for row in data_rows:
        # Split markdown row into cells, drop the empty first/last from the leading/trailing '|'
        cells = [c.strip() for c in row.split("|")[1:-1]]
        for cell in cells:
            if not cell:
                continue

            matches = num_pattern.findall(cell)
            if len(matches) < 2:
                # Zero or one numeric value in this cell, not interesting
                continue

            # Ignore things like "September 30, 2025" by requiring no letters
            letters = re.sub(r"[^A-Za-z]", "", cell)
            if letters:
                # Contains alphabetic characters, probably a date or label
                continue

            # At this point the cell has multiple numeric values and no letters:
            # exactly the 'jammed numbers' case we want.
            return True

    # No qualifying cell found, table is not complex
    return False


def find_heading_for_table(page_doc: DoclingDocument, table_md: str) -> str:
    """
    Best effort: look in the page markdown for the nearest heading above the table.
    """
    page_md = page_doc.export_to_markdown()
    page_lines = page_md.splitlines()

    table_lines = [ln for ln in table_md.splitlines() if ln.strip()]
    if not table_lines:
        return ""

    first_table_line = None
    for ln in table_lines:
        if ln.strip().startswith("|"):
            first_table_line = ln.strip()
            break
    if first_table_line is None:
        first_table_line = table_lines[0].strip()

    table_idx = None
    for i, ln in enumerate(page_lines):
        if ln.strip() == first_table_line:
            table_idx = i
            break

    if table_idx is None:
        return ""

    # Walk backwards to last markdown heading
    for j in range(table_idx - 1, -1, -1):
        line = page_lines[j].strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()

    return ""


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

    print("=== PASS 1: DOCLING ONLY ===")
    print(f"PDF_PATH: {pdf_path}")
    print(f"Output directory: {out_dir}")
    print(f"Complex tables JSON will be written to: {complex_json_path}")
    print()

    # Load Docling model
    print("Loading Docling MLX model...")
    model, processor = load(MODEL_PATH)
    config = load_config(MODEL_PATH)

    # Convert PDF to images
    print(f"Converting PDF '{pdf_path}' to images...")
    images = convert_from_path(str(pdf_path), dpi=200)
    print(f"Converted {len(images)} pages.")

    all_doctags: List[str] = []
    all_images_for_doc: List[Any] = []
    complex_tables: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp_img_dir_str:
        tmp_img_dir = Path(tmp_img_dir_str)
        image_paths: List[Path] = []

        # Save page images
        for i, img in enumerate(images):
            img_path = tmp_img_dir / f"page_{i+1:04d}.png"
            img.save(img_path, "PNG")
            image_paths.append(img_path)
            print(f"Saved page {i+1} to {img_path.name}")

        # Run Docling page by page
        for page_idx, img_path in enumerate(image_paths):
            print(f"\n--- Processing Page {page_idx+1}/{len(image_paths)} ---")
            pil_image = load_image(str(img_path))

            formatted_prompt = apply_chat_template(
                processor, config, PROMPT, num_images=1
            )

            print("Generating DocTags...")
            output = ""
            for token in stream_generate(
                model,
                processor,
                formatted_prompt,
                [pil_image],
                max_tokens=4096,
                verbose=False,
            ):
                output += token.text
                if "</doctag>" in token.text:
                    break

            all_doctags.append(output)
            all_images_for_doc.append(pil_image)
            print(f"Page {page_idx+1} DocTags done.")

            # Build a page-level Docling document for table inspection
            page_doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
                [output], [pil_image]
            )
            page_doc = DoclingDocument.load_from_doctags(
                page_doctags_doc, document_name=f"Page {page_idx+1}"
            )

            for table in page_doc.tables:
                try:
                    # Pass doc to avoid deprecation warnings
                    table_md = table.export_to_markdown(doc=page_doc)
                except Exception:
                    continue

                if not is_complex_table_from_markdown(table_md):
                    continue

                try:
                    table_html = table.export_to_html(doc=page_doc)
                except Exception:
                    table_html = ""

                heading = find_heading_for_table(page_doc, table_md)

                complex_tables.append(
                    {
                        "page_index": page_idx,
                        "heading": heading,
                        "orig_table_md": table_md,
                        "orig_table_html": table_html,
                    }
                )
                print(
                    f"Marked complex table on page {page_idx+1} "
                    f"(heading: '{heading or 'UNKNOWN'}')"
                )

    # Merge all pages into one document
    print("\nMerging all pages into one document...")
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
        all_doctags, all_images_for_doc
    )
    full_doc = DoclingDocument.load_from_doctags(
        doctags_doc, document_name=stem
    )

    # Export raw Docling markdown and HTML
    print("Exporting raw Docling markdown and HTML...")
    markdown_content = full_doc.export_to_markdown()
    with open(raw_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"✅ Raw markdown saved to: {raw_md_path}")

    full_doc.save_as_html(str(raw_html_path), image_mode=ImageRefMode.EMBEDDED)
    print(f"✅ Raw HTML saved to: {raw_html_path}")

    # Save complex table metadata for pass 2
    with open(complex_json_path, "w", encoding="utf-8") as f:
        json.dump(complex_tables, f, indent=2)
    print(f"✅ Complex table metadata saved to: {complex_json_path}")
    print(f"Detected {len(complex_tables)} complex tables.")

    print("\nPass 1 complete. You can now run the Deepseek pass 2.")


if __name__ == "__main__":
    main()