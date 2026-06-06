#!/usr/bin/env bash
#
# Descarga los artefactos del modelo desde S3 a ./artifacts.
# El pipeline llama a este script en las etapas `test` y `build`, de modo que el
# modelo .onnx nunca vive en el repositorio: solo su referencia (estas URIs).
#
# Variables de entorno:
#   MODEL_S3_URI     (requerida)  ej. s3://mi-bucket/modelo.onnx
#   BASELINE_S3_URI  (requerida)  ej. s3://mi-bucket/baseline.json
#   TESTDATA_S3_URI  (opcional)   ej. s3://mi-bucket/test_data.csv
#
set -euo pipefail

: "${MODEL_S3_URI:?Define MODEL_S3_URI}"
: "${BASELINE_S3_URI:?Define BASELINE_S3_URI}"

mkdir -p artifacts
aws s3 cp "$MODEL_S3_URI" artifacts/modelo.onnx
aws s3 cp "$BASELINE_S3_URI" artifacts/baseline.json

if [[ "${TESTDATA_S3_URI:-}" != "" ]]; then
  aws s3 cp "$TESTDATA_S3_URI" artifacts/test_data.csv
fi

echo "Artefactos descargados en ./artifacts"
