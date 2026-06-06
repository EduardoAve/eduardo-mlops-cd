"""
Entrena un clasificador de cáncer de mama y lo exporta a formato ONNX.

Genera, en la carpeta `artifacts/` (FUERA del control de versiones), los
artefactos que el sistema de despliegue obtendrá desde S3 — nunca desde el repo:

  - modelo.onnx     : el modelo a desplegar.
  - test_data.csv   : datos de prueba para la etapa `test` del pipeline.
  - baseline.json   : métrica de referencia (accuracy) y metadatos del modelo.

Se usa un Pipeline(StandardScaler -> LogisticRegression): al ser un modelo
lineal, su conversión a ONNX es exacta (a diferencia de los modelos de árboles,
cuyos umbrales sufren pérdida al truncarse a float32). El escalado va DENTRO del
modelo, de modo que la app puede enviar las features crudas del paciente.

Este script se ejecuta UNA sola vez para producir los artefactos; luego se
suben a S3 con `scripts/upload_artifacts.sh`. El `.onnx` y los datos NO se
versionan en GitHub (ver .gitignore): el repo solo guarda *cómo* se generan.
"""

import json
import os

import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
THRESHOLD = 0.90  # umbral de accuracy que la etapa `test` exigirá al modelo.


def main() -> None:
    os.makedirs(ART, exist_ok=True)

    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target
    feature_names = [str(n) for n in data.feature_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    )
    clf.fit(X_train, y_train)
    acc_sklearn = accuracy_score(y_test, clf.predict(X_test))

    # --- Exportar a ONNX -----------------------------------------------------
    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(
        clf,
        initial_types=initial_type,
        target_opset=12,
        options={id(clf): {"zipmap": False}},  # salida como tensores, no dicts
    )
    onnx_path = os.path.join(ART, "modelo.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # --- Verificar el ONNX con onnxruntime y medir su accuracy --------------
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_pred = sess.run(None, {input_name: X_test})[0].ravel()
    acc_onnx = accuracy_score(y_test, onnx_pred)

    # --- Guardar datos de prueba (features + target) ------------------------
    df = pd.DataFrame(X_test, columns=feature_names)
    df["target"] = y_test
    df.to_csv(os.path.join(ART, "test_data.csv"), index=False)

    # --- Guardar baseline + metadatos ---------------------------------------
    baseline = {
        "accuracy_sklearn": round(float(acc_sklearn), 4),
        "accuracy_onnx": round(float(acc_onnx), 4),
        "accuracy_threshold": THRESHOLD,
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "classes": [str(c) for c in data.target_names],  # ['malignant', 'benign']
    }
    with open(os.path.join(ART, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)

    print("Modelo entrenado y exportado a ONNX")
    print(f"  accuracy sklearn : {acc_sklearn:.4f}")
    print(f"  accuracy onnx    : {acc_onnx:.4f}")
    print(f"  umbral de prueba : {THRESHOLD}")
    print(f"  features         : {X.shape[1]}  | clases: {baseline['classes']}")
    print(f"  artefactos en    : {os.path.abspath(ART)}")


if __name__ == "__main__":
    main()
