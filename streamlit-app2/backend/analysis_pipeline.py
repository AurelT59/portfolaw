import backend.db_management as db
import pandas as pd
import os
import glob
import os.path as osp
from typing import Tuple

import pandas as pd

# Dossier racine où tu ranges les résultats d'analyse
BASE_ANALYSIS_DIR = "/home/sagemaker-user/shared/streamlit-app/data/analysis_result"

# ---------------------------------------------------------------------------
# 1. Fonction principale demandée
# ---------------------------------------------------------------------------
def process_file(uploaded_file):
    """
    uploaded_file : chemin complet OU juste le nom ("1.mon_doc.html")
    Le nom doit commencer par "X." où X est un chiffre ou un entier.
    """

    history = db.get_history()

    if uploaded_file in history:
        return None

    # 1) extraire le préfixe (X) et le nom sans extension
    prefix, doc_stem = _extract_prefix_and_stem(uploaded_file)

    # 2) retrouver le dossier /.../analysis_result/X
    target_dir = _get_target_dir(prefix)

    # 3) charger le csv "analysis*.csv"
    analysis_df = _load_analysis_csv(target_dir)

    # 4) calculer les scores entreprises
    company_scores_df = _compute_company_sentiment_scores(analysis_df)

    # 5) écrire le csv qui commence par X. (celui du document)
    output_path = _write_company_scores_csv(
        target_dir=target_dir,
        prefix=prefix,
        doc_stem=doc_stem,
        df=company_scores_df,
    )

    result = "This document has a huge impact"
    return result


# ---------------------------------------------------------------------------
# 2. Sous-fonctions appelées dans process_file
# ---------------------------------------------------------------------------
def _extract_prefix_and_stem(uploaded_file):
    """
    Prend "1.DIRECTIVE (UE)...html" ou un chemin complet,
    et renvoie ("1", "1.DIRECTIVE (UE)...").
    """
    filename = osp.basename(uploaded_file)
    # on enlève l'extension
    stem, _ext = osp.splitext(filename)

    # le préfixe est avant le premier point
    if "." not in stem:
        raise ValueError(
            f"Le fichier '{filename}' ne respecte pas le format 'X.nom_du_doc'"
        )
    prefix = stem.split(".", 1)[0]

    return prefix, stem


def _get_target_dir(prefix: str) -> str:
    """
    Construit le dossier où se trouvent les résultats d'analyse pour X.
    Exemple : X="1" -> /.../analysis_result/1
    """
    target_dir = osp.join(BASE_ANALYSIS_DIR, str(prefix))
    if not osp.isdir(target_dir):
        raise FileNotFoundError(
            f"Le dossier d'analyse '{target_dir}' n'existe pas. "
            f"Vérifie que le fichier '{prefix}.*' a bien été traité."
        )
    return target_dir


def _load_analysis_csv(target_dir: str) -> pd.DataFrame:
    """
    Cherche un fichier qui commence par 'analysis' dans le dossier.
    Exemple : analysis_sp500.csv, analysis_scores.csv, etc.
    """
    pattern = osp.join(target_dir, "analysis*.csv")
    candidates = glob.glob(pattern)

    if not candidates:
        raise FileNotFoundError(
            f"Aucun fichier 'analysis*.csv' trouvé dans {target_dir}"
        )

    # on prend le premier, ou on pourrait trier si besoin
    analysis_csv_path = sorted(candidates)[0]

    df = pd.read_csv(analysis_csv_path)
    # on s'assure que les colonnes attendues existent bien
    required_cols = ["company", "ticker", "avg_pos", "avg_neg", "avg_neu", "avg_mix"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Le fichier {analysis_csv_path} ne contient pas la colonne '{col}'"
            )
    return df


def _compute_company_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    À partir du df de type :
        directive_slug, directive_title, company, ticker, nb_articles,
        avg_pos, avg_neg, avg_neu, avg_mix
    on produit un DF réduit :
        name, ticker, sentiment_score
    """

    def _row_to_score(row) -> float:
        # on récupère les 4 colonnes
        pos = float(row["avg_pos"])
        neg = float(row["avg_neg"])
        neu = float(row["avg_neu"])
        mix = float(row["avg_mix"])

        # formule décrite dans le header
        raw = 1.0 * pos + 0.5 * mix + 0.25 * neu - 1.0 * neg
        score = (raw + 1.0) / 2.0  # on ramène dans [0, 1] grossièrement
        # on clippe pour être 100% sûr
        score = max(0.0, min(1.0, score))
        return score

    out = pd.DataFrame()
    out["name"] = df["company"]
    out["ticker"] = df["ticker"]
    out["sentiment_score"] = df.apply(_row_to_score, axis=1)

    return out


def _write_company_scores_csv(
    target_dir: str, prefix: str, doc_stem: str, df: pd.DataFrame
) -> str:
    """
    On écrit dans le CSV du même dossier qui porte le nom (sans extension)
    du uploaded_file. Pour faire simple tu m'as dit :
       "cherche le csv qui commence par X"
    donc on fait :
       - si /.../X*.csv existe déjà → on écrase
       - sinon → on crée /.../X.results.csv
    """

    # 1) on cherche un csv qui commence par "X"
    pattern = osp.join(target_dir, f"{prefix}*.csv")
    candidates = [p for p in glob.glob(pattern) if osp.basename(p).startswith(prefix)]

    if candidates:
        output_path = sorted(candidates)[0]
    else:
        # pas trouvé → on crée un nom simple
        # ex: 1.DIRECTIVE (UE) 2019-2161.csv
        output_path = osp.join(target_dir, f"{doc_stem}.csv")

    # on écrit (on peut mettre index=False, c'est plus propre)
    df.to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# 3. Petit main pour tester en local
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Exemple d'appel :
    fake_uploaded = "1.DIRECTIVE (UE) 2019-2161.html"
    try:
        msg = process_file(fake_uploaded)
        print(msg)
    except Exception as e:
        # en prod tu loggeras plutôt que print
        print(f"[ERROR] {e}")