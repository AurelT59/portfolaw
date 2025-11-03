import streamlit as st
from frontend.utils import gauge_maker as gm

left, right = st.columns([1, 3], gap="medium")

with left:
    with st.container():
        st.header("RVI")
        st.write("Reglementation Vulnerability Index")
        RVI = 3
        gm.gauge(RVI)

    with st.container():
        st.header("Summary")
        st.write("Summary here")

with right:
    st.header("Risk factors")

    with st.expander("Suppliers"):
        st.write("Infos")

    with st.expander("Suppliers"):
        st.write("Infos")

    with st.expander("Suppliers"):
        st.write("Infos")

    with st.expander("Suppliers"):
        st.write("Infos")

    with st.expander("Suppliers"):
        st.write("Infos")

    with st.expander("Sentiment Analysis"):
        st.write("""We've analysed Social Media posts linked to the directive 
        and its impact on S&P500 companies.\n
        Aggregating the results, globally the score is :""")