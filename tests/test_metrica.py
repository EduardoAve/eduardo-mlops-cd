"""
Prueba 2 — La métrica del modelo no cae por debajo del umbral definido.

Se calcula la accuracy sobre los datos de prueba (descargados desde S3 por el
pipeline) y se exige que sea >= al umbral guardado en baseline.json. Así se
evita promover un modelo cuyo desempeño se haya degradado significativamente.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from inference import predecir

with open(os.environ["BASELINE_PATH"], encoding="utf-8") as f:
    META = json.load(f)


def test_accuracy_supera_el_umbral():
    df = pd.read_csv(os.environ["TEST_DATA_PATH"])
    feats = META["feature_names"]
    X = df[feats].values.astype(np.float32)
    y = df["target"].values

    labels, _ = predecir(X)
    acc = accuracy_score(y, labels)
    umbral = META["accuracy_threshold"]

    assert acc >= umbral, f"accuracy {acc:.4f} por debajo del umbral {umbral}"
