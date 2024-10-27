import pandas as pd


def make_dataset(filename):
    return pd.read_csv(filename, encoding="ISO-8859-1", delimiter=";")
