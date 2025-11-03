import streamlit as st

st.set_page_config(
    page_title="PortfoLaw",
    page_icon="frontend/images/portfolaw_ico.ico",
    layout="wide"
)

pages = [
    st.Page("frontend/home.py", title="Home"),
    st.Page("frontend/file_import.py", title="File Import"),
    st.Page("frontend/dashboard.py", title="Dashboard"),
    st.Page("frontend/portfolio.py", title="Portfolio")
    # st.Page("frontend/history.py", title="History")
]

nav_bar = st.navigation(pages, position="top")
nav_bar.run()