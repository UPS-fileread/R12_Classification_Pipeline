import sys
import os
from scripts.convert_pdf import pdf_to_text, extract_first_n_pages
from scripts.classify_context import classify_context


def main():
    # Check for input PDF argument
    if len(sys.argv) < 2:
        #print("Usage: python main.py <input.pdf|input.txt>")
        sys.exit(1)
    input_pdf_path = sys.argv[1]
    ext = os.path.splitext(input_pdf_path)[1].lower()

    # Validate file existence
    if not os.path.isfile(input_pdf_path):
        print(f"File not found: {input_pdf_path}") 
        sys.exit(1)

    if ext == '.txt':
        with open(input_pdf_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        words = text_content.split()
        word_count = len(words)
        if word_count > 3000:
            words = words[:3000]
            text_content = ' '.join(words)
            word_count = 3000
        try:
            result = classify_context(text_content)
            print(f"Category   : {result.category}")
            print(f"Subcategory: {result.subcategory}")
            print(f"Summary    : {result.summary}")
            print("Key Themes :")
            for idx, theme in enumerate(result.key_themes, 1):
                print(f"  {idx}. {theme}")
        except Exception as e:
            print(f"Classification failed: {e}")
        sys.exit(0)
    elif ext == '.pdf':
        # Read PDF bytes from the input path
        with open(input_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        # Extract first 15 pages as a new PDF in memory
        first5_pdf_bytes = extract_first_n_pages(pdf_bytes, n=15)

        # Extract text from the new 15-page PDF
        try:
            text_content = pdf_to_text(first5_pdf_bytes, num_pages=15)
        except Exception as e:
            print(f"PDF text extraction failed: {e}")
            sys.exit(1)

        word_count = len(text_content.split())

        try:
            result = classify_context(text_content)
            print(f"Category   : {result.category}")
            print(f"Subcategory: {result.subcategory}")
            print(f"Summary    : {result.summary}")
            print("Key Themes :")
            for idx, theme in enumerate(result.key_themes, 1):
                print(f"  {idx}. {theme}")
        except Exception as e:
            print(f"Classification failed: {e}")

    else:
        #print(f"Unsupported file type: {ext}. Please provide a .pdf or .txt file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
