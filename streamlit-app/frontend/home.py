import streamlit as st

left, right = st.columns(2)

with left:
    st.image("frontend/images/portfolaw_classic.png", use_container_width=True)

with right:
    st.header("Welcome to PortfoLaw")

    st.write("**What's PortfoLaw ?**")

    st.write("When regulation changes, markets react. PortfoLaw Maker uses AI and its Regulatory Vulnerability Index to translate
    complex legal shifts into clear financial insights — helping investors see risk before it hits.")



    if st.button("📊 File import !"):
        st.switch_page("frontend/file_import.py")