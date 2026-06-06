#!/usr/bin/env bash
#
# Sube los artefactos generados por model/train_export.py al bucket de S3.
# Se ejecuta UNA vez (setup inicial / publicación de un nuevo modelo).
#
# Uso:
#   MODEL_BUCKET=mi-bucket ./scripts/upload_artifacts.sh
#
set -euo pipefail

: "${MODEL_BUCKET:?Define MODEL_BUCKET (nombre del bucket de S3)}"

aws s3 cp artifacts/modelo.onnx   "s3://${MODEL_BUCKET}/modelo.onnx"
aws s3 cp artifacts/baseline.json "s3://${MODEL_BUCKET}/baseline.json"
aws s3 cp artifacts/test_data.csv "s3://${MODEL_BUCKET}/test_data.csv"

echo "Artefactos subidos a s3://${MODEL_BUCKET}/"
