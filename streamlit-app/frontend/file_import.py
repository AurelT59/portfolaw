import streamlit as st
from backend.analysis_pipeline import process_file
import time

# zone document
with st.container(border=True, height="content"):
    # zone d'upload
    st.header("Import a new regulation file")

    uploaded_file = st.file_uploader(
        "Drag and drop",
        type=["html", "xml", "pdf"],
        label_visibility="hidden"
    )

    # zone d'enclenchement du traitement du document
    if uploaded_file is not None:
        st.success("✅ File uploaded")
    else:
        st.info("No file uploaded yet.")

result = None

with st.container(border=True, height="content"):
    # zone du sommaire d'analyse
    st.header("Analyse the imported file")

    if st.button("📃 Launch analysis", disabled=uploaded_file is None):
        with st.spinner("🔄 Processing ..."):
            # appel analyse backend
            time.sleep(5)
            result = process_file()
        st.success("✅ Analysis completed")

    if result is not None:
        st.write("")
        st.write(result)