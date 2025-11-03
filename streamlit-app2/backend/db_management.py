import pandas as pd
import boto3

pf_path = "data/portfolio.csv"
hist_path = "data/history.csv"
ana_path = "data/analysis_result/"

s3 = boto3.client('s3')
# bucket
bucket_name = 'csv-file-store-92c995d0'
# chemin bucket
s3_path = 'dzd-4jtjp7vk8pgcb4/63dv567snn4uc0/shared/streamlit-app/'
s3_portfolio = s3_path + pf_path
s3_history = s3_path + hist_path
s3_ana = s3_path + ana_path


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
    df = pd.read_csv(hist_path)
    df.loc[len(df), 'name'] = 'Charlie'
    s3.upload_file(hist_path, bucket_name, s3_history)
    return None


def delete_from_history():
    return None


def get_history():
    df = pd.read_csv(hist_path)
    names_list = df['file_name'].tolist()
    return names_list


def remove_extension(filename):
    if filename.endswith(".html"):
        return filename[:-5]
    elif filename.endswith(".xml"):
        return filename[:-4]
    else:
        return filename


def get_analysis(file_name):
    df = pd.read_csv(ana_path + file_name[0] + "/" + remove_extension(file_name) + ".csv")
    return df

def add_analysis(file_name):
    return None