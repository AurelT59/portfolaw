import pandas as pd
import boto3

s3 = boto3.client('s3')
# bucket
bucket_name = 'csv-file-store-92c995d0'
# chemin bucket
s3_portfolio = 'dzd-4jtjp7vk8pgcb4/63dv567snn4uc0/shared/streamlit-app/data/portfolio.csv'

pf_path = "data/portfolio.csv"


def update_portfolio(updated_df):
    updated_df.to_csv(pf_path, index=False, encoding='utf-8')
    # Upload du fichier local vers S3 (remplace l'ancien fichier)
    s3.upload_file(pf_path, bucket_name, s3_portfolio)
    return None


def reset_portfolio():
    return None


def get_portfolio():
    df = pd.read_csv(pf_path)
    return df


def add_to_history():
    return None


def delete_from_history():
    return None


def get_from_history():
    return None