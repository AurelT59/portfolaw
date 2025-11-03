import streamlit as st
import backend.db_management as db
import backend.yahoo_API_link as yAPI
import pandas as pd

with st.container(border=True, height="content"):
    st.header("""
    Stocks on portfolio
    """)

    with st.container(height="content"):
        if st.button("📈 Refresh S&P 500"):
            db.update_portfolio(yAPI.get_SP_500(db.get_portfolio()))

        uploaded_portfolio = st.file_uploader("💼 Import personalised portfolio", type=["csv"])
        if uploaded_portfolio is not None:
            df = pd.read_csv(uploaded_portfolio)
            db.update_portfolio(df)

with st.container(border=True, height="content"):
    st.header("""
    Portfolio content
    """)

    st.dataframe(
        db.get_portfolio(),
        hide_index=True
    )