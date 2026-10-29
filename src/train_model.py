from pathlib import Path
import json
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "winequality-red.csv"
MODEL = ROOT / "models" / "wine_model.joblib"
RESULTS = ROOT / "models" / "metrics.json"
PREDICTIONS = ROOT / "models" / "test_predictions.csv"

SOURCE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)


def get_data():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        print("Laster ned UCI Wine Quality (red wine) ...")
        urllib.request.urlretrieve(SOURCE, DATA)

    df = pd.read_csv(DATA, sep=";")

    if df.empty or "quality" not in df.columns or df.isna().any().any():
        raise ValueError(
            "Datasettet er tomt, mangler quality, "
            "eller har manglende verdier."
        )

    return df


def metrics(y, predictions):
    return {
        "MAE": round(
            float(mean_absolute_error(y, predictions)), 4
        ),
        "RMSE": round(
            float(np.sqrt(mean_squared_error(y, predictions))), 4
        ),
        "R2": round(
            float(r2_score(y, predictions)), 4
        ),
    }


def main():
    df = get_data()

    X = df.drop(columns="quality")
    y = df["quality"]

    # 60 % trening, 20 % validering, 20 % test.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=42,
    )

    candidates = {
        "Baseline (gjennomsnitt)": DummyRegressor(
            strategy="mean"
        ),
        "Ridge Regression": Ridge(
            alpha=10.0
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    validation = {}

    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)

        validation[name] = metrics(
            y_val,
            estimator.predict(X_val),
        )

    # Velg modellen med lavest validerings-MAE.
    winner = min(
        validation,
        key=lambda name: validation[name]["MAE"],
    )

    final_model = candidates[winner]

    # Tren sluttmodellen på trening + validering.
    final_model.fit(X_trainval, y_trainval)

    # Evaluer på det separate testsettet.
    test_predictions = final_model.predict(X_test)
    test = metrics(y_test, test_predictions)

    # Lagre testprediksjoner for videre analyse.
    prediction_df = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": test_predictions,
        "absolute_error": np.abs(
            y_test.to_numpy() - test_predictions
        ),
    })

    MODEL.parent.mkdir(parents=True, exist_ok=True)

    prediction_df.to_csv(
        PREDICTIONS,
        index=False,
    )

    joblib.dump(
        {
            "model": final_model,
            "features": list(X.columns),
        },
        MODEL,
    )

    report = {
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "model_selected": winner,
        "validation": validation,
        "test": test,
        "feature_means": {
            k: float(v)
            for k, v in X.mean().items()
        },
        "feature_min": {
            k: float(v)
            for k, v in X.min().items()
        },
        "feature_max": {
            k: float(v)
            for k, v in X.max().items()
        },
        "quality_distribution": {
            str(k): int(v)
            for k, v in y.value_counts().sort_index().items()
        },
        "prediction_range": {
            "minimum": float(np.min(test_predictions)),
            "maximum": float(np.max(test_predictions)),
            "mean": float(np.mean(test_predictions)),
        },
    }

    RESULTS.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nTestprediksjoner lagret i:")
    print(PREDICTIONS)

    print("\nFerdig! Kjør: streamlit run app.py")


if __name__ == "__main__":
    main()