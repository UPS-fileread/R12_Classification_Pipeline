import streamlit as st
import os
import time
import pandas as pd
import io
import zipfile
import tempfile
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
            file_key = uploaded_file.name

            st.subheader(f"{file_key}")
            start_time = time.time()

            if file_key not in st.session_state:
                file_ext = os.path.splitext(file_key)[1].lower()
                if file_ext == ".pdf":
                    pdf_bytes = uploaded_file.read()
                    first5_pdf_bytes = extract_first_n_pages(pdf_bytes, n=5)
                    with st.spinner('Extracting text'):
                        text_content = pdf_to_text(first5_pdf_bytes, num_pages=5)
                elif file_ext == ".txt":
                    text_content = uploaded_file.read().decode("utf-8", errors="ignore")
                else:
                    st.warning(f"Unsupported file type: {file_key}")
                    continue

                try:
                    with st.spinner('Analyzing document...'):
                        result = classify_context(text_content)
                    st.session_state[file_key] = {
                        "text": text_content,
                        "result": result
                    }
                except Exception as e:
                    st.error(f"Error analyzing {file_key}: {e}")
                    continue
            else:
                text_content = st.session_state[file_key]["text"]
                result = st.session_state[file_key]["result"]

            # Preview extracted text
            st.text_area("Preview", text_content[:200], height=100, disabled=True)

            # Display classification results
            col1, col2 = st.columns(2)
            col1.metric(label="Category", value=str(result.category))
            col2.metric(label="Subcategory", value=str(result.subcategory))

            sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]

            # --- Feedback and sentiment collection ---
            # Category
            col1, col2 = st.columns(2)
            with col1:
                #st.markdown("**Category Sentiment:**")
                selected_cat = st.feedback("thumbs", key=f"cat_{uploaded_file.name}")
                if selected_cat is not None:
                    st.markdown(f"You selected: {sentiment_mapping[selected_cat]} for Category")
            with col2:
                #st.markdown("**Subcategory Sentiment:**")
                selected_subcat = st.feedback("thumbs", key=f"subcat_{uploaded_file.name}")
                if selected_subcat is not None:
                    st.markdown(f"You selected: {sentiment_mapping[selected_subcat]} for Subcategory")

            st.markdown("**Summary:**")
            st.info(result.summary)
            #st.markdown("**Summary Sentiment:**")
            selected_sum = st.feedback("thumbs", key=f"sum_{uploaded_file.name}")
            if selected_sum is not None:
                st.markdown(f"You selected: {sentiment_mapping[selected_sum]} for Summary")

            st.markdown("**Key Themes:**")
            for theme in result.key_themes:
                st.markdown(f"- {theme}")
            #st.markdown("**Key Themes Sentiment:**")
            selected_themes = st.feedback("thumbs", key=f"themes_{uploaded_file.name}")
            if selected_themes is not None:
                st.markdown(f"You selected: {sentiment_mapping[selected_themes]} for Key Themes")

            with st.expander("View Full Document"):
                st.text_area("Document Text", text_content, height=300, disabled=True)

            end_time = time.time()
            st.info(f"Time taken to process: {end_time - start_time:.2f} seconds")

            # --- Save results and sentiments to CSV on button click ---
            # Only enable download if all feedbacks are provided
            if (selected_cat is not None and selected_subcat is not None and
                selected_sum is not None and selected_themes is not None):

                data = {
                    "Category": [str(result.category)],
                    "Category_Sentiment": [selected_cat],
                    "SubCategory": [str(result.subcategory)],
                    "SubCategory_Sentiment": [selected_subcat],
                    "Summary": [result.summary],
                    "Summary_Sentiment": [selected_sum],
                    "KeyThemes": ["; ".join(result.key_themes)],
                    "KeyThemes_Sentiment": [selected_themes]
                }
                df = pd.DataFrame(data)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                base_name = os.path.splitext(uploaded_file.name)[0]
                csv_filename = f"{base_name}.csv"

                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Save CSV file
                    csv_path = os.path.join(tmpdirname, csv_filename)
                    with open(csv_path, "w") as f:
                        f.write(csv_data)

                    # Save uploaded document
                    uploaded_path = os.path.join(tmpdirname, uploaded_file.name)
                    with open(uploaded_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Create zip archive
                    zip_filename = f"{base_name}_results.zip"
                    zip_path = os.path.join(tmpdirname, zip_filename)
                    with zipfile.ZipFile(zip_path, "w") as zipf:
                        zipf.write(csv_path, arcname=csv_filename)
                        zipf.write(uploaded_path, arcname=uploaded_file.name)

                    # Read the zip for download
                    with open(zip_path, "rb") as f:
                        zip_bytes = f.read()

                    st.download_button(
                        label="Download Results",
                        data=zip_bytes,
                        file_name=zip_filename,
                        mime="application/zip"
                    )
    else:
        st.info('👆 Please upload at least one file to analyze.')

if __name__ == "__main__":
    main()