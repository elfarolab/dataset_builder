# test_images.py
import pymupdf

pdf_path = "../pdf/enso_evolution-status-fcsts-web.pdf"
doc = pymupdf.open(pdf_path)
print(f"PDF has {len(doc)} pages")
for i in range(min(3, len(doc))):
    page = doc[i]
    imgs = page.get_images(full=True)
    draws = page.get_drawings()
    print(f"Page {i+1}: images={len(imgs)}, drawings={len(draws)}")
doc.close()

