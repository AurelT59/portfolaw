import plotly.graph_objects as go
import streamlit as st


def gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={
            'axis': {'range': [0, 5]},
            'bar': {'color': 'red' if value >= 3.33 else 'orange' if value >= 1.67 else 'green'},
            'threshold': {
                'line': {'color': "black", 'width': 6},
                'value': value
            }
        }
    ))
    st.plotly_chart(fig)