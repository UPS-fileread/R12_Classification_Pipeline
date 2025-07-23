import streamlit as st
import os
import time
import json
import pandas as pd
import io
import zipfile
import tempfile
from scripts.classify_context import classify_context
from scripts.convert_pdf import pdf_to_text, extract_first_n_pages

import json
# Load definitions for categories and subcategories
script_dir = os.path.dirname(__file__)
definitions_path = os.path.join(script_dir, "scripts", "definitions.json")
with open(definitions_path, "r") as f:
    definitions_data = json.load(f)
categories_list = list(definitions_data["context_types"].keys())
subcategories_data = definitions_data["subcategories"]


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
            col2.metric(label="Subcategory", value=result.subcategory.value)

            sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]

            # --- Feedback and sentiment collection ---
            # Category
            col1, col2 = st.columns(2)
            with col1:
                cat_feedback = st.feedback("thumbs", key=f"cat_{uploaded_file.name}")
                if cat_feedback is not None:
                    if cat_feedback == 1:
                        category_value = str(result.category)
                    else:
                        category_value = st.selectbox(
                            "Please select the correct category",
                            categories_list,
                            key=f"cat_select_{uploaded_file.name}"
                        )
                    st.markdown(f"You selected: **{category_value}** for Category")
            with col2:
                # If category was disliked, auto-dislike subcategory and show correction dropdown
                if 'cat_feedback' in locals() and cat_feedback == 0:
                    subcat_feedback = 0
                    current_cat = category_value if 'category_value' in locals() else str(result.category)
                    options = subcategories_data.get(current_cat, [])
                    subcategory_value = st.selectbox(
                        "Please select the correct subcategory",
                        options,
                        key=f"subcat_select_{uploaded_file.name}"
                    )
                    st.markdown(f"You selected: **{subcategory_value}** for Subcategory")
                else:
                    subcat_feedback = st.feedback("thumbs", key=f"subcat_{uploaded_file.name}")
                    if subcat_feedback is not None:
                        if subcat_feedback == 1:
                            subcategory_value = str(result.subcategory)
                        else:
                            current_cat = category_value if 'category_value' in locals() else str(result.category)
                            options = subcategories_data.get(current_cat, [])
                            subcategory_value = st.selectbox(
                                "Please select the correct subcategory",
                                options,
                                key=f"subcat_select_{uploaded_file.name}"
                            )
                        st.markdown(f"You selected: **{subcategory_value}** for Subcategory")

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

                st.download_button(
                    label="Download Results as CSV",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv"
                )
    else:
        st.info('👆 Please upload at least one file to analyze.')

if __name__ == "__main__":
    main()