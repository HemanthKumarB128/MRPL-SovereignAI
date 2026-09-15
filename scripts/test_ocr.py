from pdf2image import convert_from_path
import pytesseract

images = convert_from_path("../data/pdfs/IIRS.pdf")

text = ""

for image in images:
    text += pytesseract.image_to_string(image)

print("Characters extracted:", len(text))
print("\nFirst 1000 characters:\n")
print(text[:1000])
