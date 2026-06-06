"""
Prueba 1 — El modelo responde correctamente ante datos de entrada definidos.
"""

import json
import os

import numpy as np

from inference import predecir

with open(os.environ["BASELINE_PATH"], encoding="utf-8") as f:
    META = json.load(f)
N = META["n_features"]


def test_modelo_responde_con_entrada_definida():
    """Ante un vector de entrada definido, el modelo retorna una etiqueta y
    probabilidades válidas."""
    x = np.zeros(N, dtype=np.float32)
    labels, probs = predecir(x)

    assert labels.shape == (1,)
    assert int(labels[0]) in (0, 1)
    assert probs.shape == (1, 2)
    # Las probabilidades de las dos clases suman 1.
    assert abs(float(probs[0].sum()) - 1.0) < 1e-3


def test_modelo_responde_para_un_lote():
    """El modelo procesa varias muestras a la vez (matriz N x features)."""
    X = np.zeros((3, N), dtype=np.float32)
    labels, probs = predecir(X)

    assert labels.shape == (3,)
    assert probs.shape == (3, 2)
