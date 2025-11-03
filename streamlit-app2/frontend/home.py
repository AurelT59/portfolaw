import streamlit as st

st.image("frontend/images/homepage.png", use_container_width=True)

with st.container(horizontal_alignment="center"):
    if st.button("📊 Get started !"):
        st.switch_page("frontend/file_import.py")