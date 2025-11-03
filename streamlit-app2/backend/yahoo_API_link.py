import yfinance as yf
import pandas as pd


def get_SP_500(df):
    # Récupérer les capitalisations boursières via yfinance
    tickers = yf.Tickers(" ".join(df['Symbol']))
    market_caps = {t: info.info.get('marketCap') for t, info in tickers.tickers.items()}

    # Ajouter la colonne Market Cap au DataFrame
    df['Market Cap'] = df['Symbol'].map(market_caps)

    # Calculer les poids dans l'indice (ignorer les valeurs manquantes)
    if df['Market Cap'].notnull().any():
        total_cap = df['Market Cap'].sum(skipna=True)
        df['Weight'] = df['Market Cap'] / total_cap
    else:
        df['Weight'] = None

    return df
