import sys
import os
import time
from scripts.convert_pdf import pdf_to_text, extract_first_n_pages
from scripts.ocr import ocr_pdf_to_text
from scripts.classify_context import classify_context


def main():
    # Check for input PDF argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <input.pdf|input.txt>")
        sys.exit(1)
    input_pdf_path = sys.argv[1]
    ext = os.path.splitext(input_pdf_path)[1].lower()

    # Validate file existence
    if not os.path.isfile(input_pdf_path):
        print(f"File not found: {input_pdf_path}")
        sys.exit(1)

    if ext == '.txt':
        print("\n--- Text File Classification Pipeline ---\n")
        with open(input_pdf_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        words = text_content.split()
        word_count = len(words)
        print(f"Word count: {word_count}")
        if word_count > 3000:
            print("Text exceeds 3000 words. Truncating to 3000 words.")
            words = words[:3000]
            text_content = ' '.join(words)
            word_count = 3000
        print("\nClassifying document context...")
        try:
            result = classify_context(text_content)
            print("\n--- Classification Results ---")
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
        start_time = time.time()
        print("\n--- Document Classification Pipeline ---\n")

        # Step 1: Extract first 15 pages as a new PDF in memory
        print("Extracting first 15 pages...")
        first5_pdf_bytes = extract_first_n_pages(pdf_bytes, n=15)

        # Step 2: Extract text from the new 15-page PDF
        print("Extracting text from first 15 pages...")
        try:
            text_content = pdf_to_text(first5_pdf_bytes, num_pages=15)
        except Exception as e:
            print(f"PDF text extraction failed: {e}")
            sys.exit(1)

        word_count = len(text_content.split())
        print(f"Word count (first 5 pages): {word_count}")

        # Step 3: If word count < 100, perform OCR on the 15-page PDF
        if word_count < 100:
            print("Unreadable text detected. Performing OCR for better extraction...")
            import tempfile
            with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(first5_pdf_bytes)
                tmp_pdf.flush()
                text_content = ocr_pdf_to_text(tmp_pdf.name)
            word_count = len(text_content.split())
            print(f"OCR word count: {word_count}")

        # Optional: Preview first 200 characters
        # print("Preview:\n" + text_content[:200])

        # Step 4: Classify context using LLM
        print("\nClassifying document context...")
        try:
            result = classify_context(text_content)
            print("\n--- Classification Results ---")
            print(f"Category   : {result.category}")
            print(f"Subcategory: {result.subcategory}")
            print(f"Summary    : {result.summary}")
            print("Key Themes :")
            for idx, theme in enumerate(result.key_themes, 1):
                print(f"  {idx}. {theme}")
        except Exception as e:
            print(f"Classification failed: {e}")

        end_time = time.time()
        print(f"\nTime taken to process: {end_time - start_time:.2f} seconds\n")
    else:
        print(f"Unsupported file type: {ext}. Please provide a .pdf or .txt file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
