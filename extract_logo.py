import fitz
import os
import glob
print("Starting extraction...")
pdfs = glob.glob('c:/Users/sonam/Downloads/*.pdf')
if not pdfs:
    print("No PDFs found")
    exit(1)
latest_pdf = max(pdfs, key=os.path.getmtime)
print(f"Opening {latest_pdf}")
doc = fitz.open(latest_pdf)
for i in range(len(doc)):
    for img in doc.get_page_images(i):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha < 4:
            pix.save("static/logo.png")
        else:
            pix1 = fitz.Pixmap(fitz.csRGB, pix)
            pix1.save("static/logo.png")
            pix1 = None
        pix = None
        print("Logo extracted to static/logo.png")
        exit(0)
print("No images found in PDF")
