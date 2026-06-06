"""
Inferencia con el modelo ONNX.

El modelo NO está en el repositorio: se obtiene desde S3 durante el pipeline y
se hornea en la imagen Docker. En tiempo de ejecución se carga desde la ruta
indicada por la variable de entorno ``MODEL_PATH`` (por defecto ``modelo.onnx``
en el directorio de trabajo del contenedor).
"""

import os
from typing import Sequence, Tuple

import numpy as np
import onnxruntime as ort

MODEL_PATH = os.environ.get("MODEL_PATH", "modelo.onnx")

_session = None
_input_name = None


def _get_session() -> Tuple[ort.InferenceSession, str]:
    """Carga perezosa de la sesión ONNX (una sola vez por proceso)."""
    global _session, _input_name
    if _session is None:
        _session = ort.InferenceSession(
            MODEL_PATH, providers=["CPUExecutionProvider"]
        )
        _input_name = _session.get_inputs()[0].name
    return _session, _input_name


def predecir(features: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta el modelo sobre una o varias muestras.

    ``features`` puede ser un vector (n_features,) o una matriz (N, n_features).
    Retorna ``(labels, probabilidades)``:
      - labels: np.ndarray de enteros (0 = malignant, 1 = benign).
      - probabilidades: np.ndarray (N, 2) con la probabilidad de cada clase.
    """
    sess, input_name = _get_session()
    X = np.asarray(features, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    outputs = sess.run(None, {input_name: X})
    labels = np.asarray(outputs[0]).ravel()
    probabilidades = np.asarray(outputs[1])
    return labels, probabilidades
