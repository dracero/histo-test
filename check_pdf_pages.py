import fitz
import sys

def find_pages(pdf_path, search_terms):
    doc = fitz.open(pdf_path)
    found_pages = []
    
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text().lower()
        for term in search_terms:
            if term.lower() in text:
                print(f"Match found on Page {i+1} for term '{term}'")
                print(f"Snippet: {text[text.find(term.lower())-50:text.find(term.lower())+50]}")
                found_pages.append(i+1)
                
    doc.close()
    return found_pages

if __name__ == "__main__":
    pdf_path = "pdf/arch2.pdf"
    terms = ["nervioso", "neurona", "sustancia gris", "piramid", "cerebel"]
    find_pages(pdf_path, terms)
