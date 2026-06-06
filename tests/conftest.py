"""
Configuración de las pruebas.

- Agrega ``app/`` al path para importar ``inference``.
- Define rutas por defecto de los artefactos (modelo, baseline, datos de prueba).
  En el pipeline, la etapa `test` descarga estos artefactos desde S3 a la carpeta
  ``artifacts/``; en local provienen de ``model/train_export.py``. Si el entorno
  ya define estas variables, se respetan (``setdefault``).
"""

import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "app"))

ART = os.path.join(RAIZ, "artifacts")
os.environ.setdefault("MODEL_PATH", os.path.join(ART, "modelo.onnx"))
os.environ.setdefault("BASELINE_PATH", os.path.join(ART, "baseline.json"))
os.environ.setdefault("TEST_DATA_PATH", os.path.join(ART, "test_data.csv"))
