import fitz
from pathlib import Path

out_dir = Path("results/pdf_pages_render")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Render VI pages
doc_vi = fitz.open("paper/full_paper_vi.pdf")
print("VI total pages:", len(doc_vi))
for pno in range(len(doc_vi)):
    text = doc_vi[pno].get_text()
    if "Hình S1" in text or "Bảng S3" in text or "Kansas City" in text:
        page = doc_vi[pno]
        pix = page.get_pixmap(dpi=150)
        img_path = out_dir / f"page_vi_{pno+1}.png"
        pix.save(str(img_path))
        print(f"Saved VI page {pno+1} to {img_path}")

# 2. Render EN pages
doc_en = fitz.open("paper/full_paper_en.pdf")
print("EN total pages:", len(doc_en))
for pno in range(len(doc_en)):
    text = doc_en[pno].get_text()
    if "Figure S1" in text or "Table S3" in text or "Kansas City" in text:
        page = doc_en[pno]
        pix = page.get_pixmap(dpi=150)
        img_path = out_dir / f"page_en_{pno+1}.png"
        pix.save(str(img_path))
        print(f"Saved EN page {pno+1} to {img_path}")
