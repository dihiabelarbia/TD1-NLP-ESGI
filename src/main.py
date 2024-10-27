import click
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score, cross_validate, KFold

from data import make_dataset
from feature import make_features
from models import make_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/model.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    if y.isnull().any():
        print("y contient des valeurs manquantes. Elles seront supprimées.")
        X = X[~y.isnull()]
        y = y.dropna()
    model = make_model()
    model.fit(X, y)

    joblib.dump(model, model_dump_filename)
    print(f"Modèle sauvegardé dans {model_dump_filename}")


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df)
    model = joblib.load(model_dump_filename)
    predictions = model.predict(X)

    df["is_comic_predicted"] = predictions
    df[["video_name", "is_comic_predicted"]].to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")


@click.command()
@click.option("--input_filename", default="data/train.csv", help="File training data")
def evaluate(input_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    if y.isnull().any():
        print("y contient des valeurs manquantes. Elles seront supprimées.")
        X = X[~y.isnull()]
        y = y.dropna()

    model = make_model()
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    k = 10

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    n_estimators = [50, 100, 150, 200, 250, 300, 350]

    for val in n_estimators:
        score = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
        print(f'Average score({val}): {"{:.3f}".format(score.mean())}')

    cnt = 1
    # La méthode split() génère des indices pour diviser les données en ensembles d'entraînement et de test.
    for train_index, test_index in kf.split(X, y):
        print(f'Fold: {cnt}, Train set: {len(train_index)}, Test set: {len(test_index)}')
        cnt += 1

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    }

    scores = cross_validate(model, X, y, cv=kf, scoring=scoring)

    print(f"Result with {k} splits:")
    for metric in scoring.keys():
        metric_scores = scores[f'test_{metric}']
        print(f"{metric.capitalize()} : {metric_scores.mean():.4f} (+/- {metric_scores.std() * 2:.4f})")

    return scores


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
