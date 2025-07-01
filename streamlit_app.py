import streamlit as st
import os
import time
from scripts.classify_context import classify_context
from scripts.convert_pdf import pdf_to_text, extract_first_n_pages


def main():
    # App title and instructions
    st.title('Fileread Document Classification')
    st.write('Upload one or more PDF files to classify their legal context and subcategory.')

    # File uploader widget
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"{uploaded_file.name}")
            start_time = time.time()

            # Determine file type by extension
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if file_ext == ".pdf":
                pdf_bytes = uploaded_file.read()
                # Step 1: Extract first 5 pages as a new PDF in memory
                first5_pdf_bytes = extract_first_n_pages(pdf_bytes, n=5)
                # Step 2: Extract text from the new 5-page PDF
                with st.spinner('Extracting text'):
                    text_content = pdf_to_text(first5_pdf_bytes, num_pages=5)
            elif file_ext == ".txt":
                # Read text content directly
                text_content = uploaded_file.read().decode("utf-8", errors="ignore")
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue

            # Preview extracted text
            st.text_area("Preview", text_content[:200], height=100, disabled=True)

            # Step 4: Classify context using LLM
            try:
                with st.spinner('Analyzing document...'):
                    result = classify_context(text_content)
                # Display classification results
                col1, col2 = st.columns(2)
                col1.metric(label="Category", value=str(result.category))
                col2.metric(label="Subcategory", value=str(result.subcategory))
                st.markdown("**Summary:**")
                st.info(result.summary)
                st.markdown("**Key Themes:**")
                for theme in result.key_themes:
                    st.markdown(f"- {theme}")
                # Expandable section for full document text
                with st.expander("View Full Document"):
                    st.text_area("Document Text", text_content, height=300, disabled=True)
            except Exception as e:
                st.error(f"Error analyzing {uploaded_file.name}: {e}")
            end_time = time.time()
            st.info(f"Time taken to process: {end_time - start_time:.2f} seconds")
    else:
        st.info('👆 Please upload at least one file to analyze.')

if __name__ == "__main__":
    main()