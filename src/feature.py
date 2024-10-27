import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer


def make_features(df):
    def one_hot_encoding(x: pd.Series):
        vectorizer = HashingVectorizer(n_features=2 ** 8)
        X = vectorizer.fit_transform(x)
        return X

    # Vérification de la présence de la colonne 'is_comic' pour éviter l'erreur
    if "is_comic" in df.columns:
        y = df["is_comic"]
    else:
        y = None  # Pas de colonne 'is_comic' dans le cas de prédiction

    X = one_hot_encoding(df["video_name"])

    return X, y
