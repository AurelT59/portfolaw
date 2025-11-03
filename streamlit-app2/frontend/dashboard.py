import streamlit as st
from frontend.utils import gauge_maker as gm
import backend.db_management as db

history = db.get_history()

option = st.selectbox(
    "Select file to print out dashboard",
    (history),
    index=None
)

if option is not None:

    df_analysis = db.get_analysis(option)

    mapping = {
                'Low': 0,
                'Moderate': 2.5,
                'High': 5
            }

    df_analysis['score_num'] = df_analysis['dependance_score'].map(mapping)

    left, right = st.columns([1, 3], gap="medium")

    with left:
        with st.container():
            st.header("RVI")
            st.write("Reglementation Vulnerability Index")
            df_analysis['RVI'] = df_analysis["score_num"]*df_analysis["sentiment_score"]*5/2
            RVI = df_analysis['RVI'].mean()
            gm.gauge(RVI)

        # with st.container():
        #     st.header("Summary")
        #     st.write("Summary here")

    with right:
        st.header("Risk factors")

        with st.expander("Dependance"):
            # Remplacer les valeurs
            dependance_score = df_analysis['score_num'].mean()
            gm.gauge(dependance_score)
            st.write("Score aggregate of both a supplier and a geographical zone risks scores.")

        # with st.expander("Impacted suppliers"):
        #     supplier_score = df_analysis['supplier_score'].mean()
        #     gm.gauge(supplier_score)
        #     st.write("Infos")

        with st.expander("Sentiment Analysis"):
            sentiment_score = df_analysis['sentiment_score'].mean()
            gm.gauge(sentiment_score)
            st.write("""Score obtained from analysed Social Media posts linked to the directive
            and its impact on S&P500 companies.
            """)

            st.dataframe(
                df_analysis[['name', 'recommandation']],
                hide_index=True
            )