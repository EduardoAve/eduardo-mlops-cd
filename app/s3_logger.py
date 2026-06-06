"""
Registro de predicciones en S3.

Cada predicción realizada a través del endpoint agrega una línea a un archivo de
texto en S3, distinto por ambiente: ``predicciones_dev.txt`` o
``predicciones_prod.txt``. Estos archivos sirven para monitoreo/análisis futuro.

Variables de entorno:
  - ``LOG_BUCKET``: bucket de S3 donde viven los .txt (si no está definida, el
    registro se omite silenciosamente: útil en local y en las pruebas).
  - ``APP_ENV``: ``dev`` o ``prod`` (define el nombre del archivo).
  - ``AWS_REGION``: región del bucket (por defecto us-east-1).

Nota: S3 no permite "append" nativo, así que se hace leer-modificar-escribir.
Es suficiente para el volumen de este sistema; en alta concurrencia se usaría
un almacén append-only (p. ej. Kinesis/Firehose o DynamoDB).
"""

import os
from datetime import datetime, timezone
from typing import Optional, Sequence

APP_ENV = os.environ.get("APP_ENV", "dev")
LOG_BUCKET = os.environ.get("LOG_BUCKET")
LOG_KEY = f"predicciones_{APP_ENV}.txt"


def _formatear(entrada: Sequence[float], prediccion: str,
               probabilidad: Optional[float]) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    prob = f"{probabilidad:.4f}" if probabilidad is not None else "NA"
    entrada_str = ",".join(f"{float(v):.4f}" for v in entrada)
    return f"{ts}\t{APP_ENV}\t{prediccion}\tprob={prob}\tentrada=[{entrada_str}]"


def registrar(entrada: Sequence[float], prediccion: str,
              probabilidad: Optional[float] = None) -> str:
    """
    Agrega una línea con la predicción al archivo de texto del ambiente en S3.

    Retorna la línea registrada. Si ``LOG_BUCKET`` no está configurado, no
    escribe en S3 (no falla); pensado para ejecución local y pruebas.
    """
    linea = _formatear(entrada, prediccion, probabilidad)
    if not LOG_BUCKET:
        return linea

    import boto3  # import local: solo se necesita cuando hay bucket configurado

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    try:
        actual = s3.get_object(Bucket=LOG_BUCKET, Key=LOG_KEY)["Body"].read().decode("utf-8")
    except s3.exceptions.NoSuchKey:
        actual = ""
    s3.put_object(
        Bucket=LOG_BUCKET,
        Key=LOG_KEY,
        Body=(actual + linea + "\n").encode("utf-8"),
    )
    return linea
