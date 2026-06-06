# Sistema de despliegue automático de modelos (MLOps)

Sistema de **despliegue continuo (CD) de modelos de machine learning**: cuando se
publica un nuevo modelo, un pipeline de **GitHub Actions** lo prueba y lo despliega
automáticamente a un endpoint en la nube (**AWS**), con dos ambientes
independientes —**`dev`** y **`prod`**— para que los usuarios finales interactúen
con él.

El caso de uso es un **clasificador de cáncer de mama** (benigno / maligno)
servido como una app **Streamlit**. El modelo está en formato **ONNX** y **no se
versiona en el repositorio**: vive en un bucket de S3 y el pipeline lo descarga
en cada ejecución.

---

## Arquitectura

```
   git push dev/prod
        │
        ▼
  ┌──────────────────────── GitHub Actions (.github/workflows/deploy.yml) ───────────────────────┐
  │                                                                                               │
  │  [test]   configure-aws → descarga modelo.onnx + test_data.csv de S3 → pytest                 │
  │              · responde ante entrada definida   · accuracy ≥ umbral (0.90)                    │
  │                                   │ (needs: test)                                             │
  │  [build-deploy]  descarga modelo de S3 → docker build (hornea el .onnx) → push a ECR          │
  │                  → SSH a EC2: docker pull + run en el puerto del ambiente                      │
  └───────────────────────────────────────────────────────────────────────────────────────────┘
        │                                                  │
        ▼ dev                                              ▼ prod
  http://<EC2_IP>:8501                              http://<EC2_IP>:8502
        │                                                  │
        └──────────── cada predicción → append a ──────────┘
                   s3://<bucket>/predicciones_dev.txt  |  predicciones_prod.txt
```

| Componente | Servicio AWS | Rol |
| --- | --- | --- |
| Endpoints `dev` / `prod` | **EC2** (1 instancia, Docker) | Contenedores Streamlit en `:8501` (dev) y `:8502` (prod) |
| Almacenamiento del modelo y datos | **S3** | `modelo.onnx`, `baseline.json`, `test_data.csv`, logs de predicciones |
| Registro de imágenes | **ECR** | Imágenes Docker etiquetadas por ambiente y commit |
| CI/CD | **GitHub Actions** | `test` + `build-deploy` en cada push a `dev` / `prod` |

---

## Estructura del repositorio

```
.
├── app/                      # Aplicación que se conteneriza
│   ├── app.py                #   UI Streamlit (carga ejemplos, predice)
│   ├── inference.py          #   Carga del modelo ONNX y predicción
│   ├── s3_logger.py          #   Registra cada predicción en S3 (predicciones_<env>.txt)
│   └── requirements.txt      #   Dependencias de runtime (imagen Docker)
├── model/
│   └── train_export.py       # Entrena y exporta el modelo a ONNX (genera los artefactos)
├── tests/                    # Pruebas del modelo (etapa `test` del pipeline)
│   ├── test_responde.py      #   El modelo responde ante una entrada definida
│   └── test_metrica.py       #   La accuracy no cae por debajo del umbral
├── scripts/
│   ├── download_artifacts.sh # Descarga modelo/datos desde S3 (usado por el pipeline)
│   └── upload_artifacts.sh   # Sube los artefactos a S3 (setup inicial)
├── Dockerfile                # Imagen de la app (hornea el modelo bajado de S3)
├── requirements-test.txt     # Dependencias de la etapa de pruebas
└── .github/workflows/
    └── deploy.yml            # Pipeline CI/CD (test + build/promote)
```

> ⚠️ El archivo `modelo.onnx` **no existe en el repositorio** (ver `.gitignore`).
> El repo solo guarda *cómo se genera* (`model/train_export.py`) y *dónde se
> obtiene* (variables `*_S3_URI`). Lo mismo aplica a los datos de prueba.

---

## Ramas y ambientes

| Rama | Ambiente | Endpoint | Archivo de log |
| --- | --- | --- | --- |
| `dev` | desarrollo | `http://<EC2_IP>:8501` | `predicciones_dev.txt` |
| `prod` | producción | `http://<EC2_IP>:8502` | `predicciones_prod.txt` |

Cada push a una rama dispara el pipeline, que despliega al endpoint de ese
ambiente. Así, `dev` sirve para validar un modelo nuevo antes de promoverlo a
`prod` (un push/merge a `prod`).

---

## El pipeline de CI/CD (`deploy.yml`)

Se ejecuta **en cada push a `dev` o `prod`** y tiene dos etapas:

1. **`test`** — Configura credenciales AWS, **descarga el modelo y los datos de
   prueba desde S3** (no están en el repo) y corre `pytest`:
   - `test_responde.py`: el modelo responde con una salida válida ante una
     entrada definida.
   - `test_metrica.py`: la *accuracy* sobre los datos de prueba es **≥ 0.90**
     (umbral en `baseline.json`); evita promover un modelo degradado.
2. **`build-deploy`** (solo si `test` pasó) — Descarga el modelo de S3, construye
   la imagen Docker (con el `.onnx` adentro), la publica en **ECR** y se conecta
   por **SSH a la EC2** para hacer `docker pull` + `docker run` en el puerto del
   ambiente, **actualizando el endpoint**.

---

## Configuración (una sola vez)

### 1. Artefactos del modelo en S3

```bash
python model/train_export.py                 # genera artifacts/ (modelo, datos, baseline)
MODEL_BUCKET=<tu-bucket> ./scripts/upload_artifacts.sh
```

### 2. Secrets del repositorio (Settings → Secrets and variables → Actions → *Secrets*)

| Secret | Descripción |
| --- | --- |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Credenciales del usuario IAM de CI/CD |
| `EC2_HOST` | IP pública o DNS de la instancia EC2 |
| `EC2_USER` | Usuario SSH (p. ej. `ec2-user`) |
| `EC2_SSH_KEY` | Clave **privada** SSH de la instancia |

### 3. Variables del repositorio (misma pantalla → *Variables*)

| Variable | Ejemplo |
| --- | --- |
| `MODEL_S3_URI` | `s3://<bucket>/modelo.onnx` |
| `BASELINE_S3_URI` | `s3://<bucket>/baseline.json` |
| `TESTDATA_S3_URI` | `s3://<bucket>/test_data.csv` |
| `LOG_BUCKET` | `<bucket>` (donde se guardan los `predicciones_*.txt`) |

La instancia EC2 usa un **IAM instance role** con permisos de *pull* en ECR y de
lectura/escritura en S3 (para que la app registre las predicciones).

---

## Ejecutar localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt -r requirements-test.txt

python model/train_export.py          # genera artifacts/
pytest -v                             # corre las pruebas del modelo

# App (usa el modelo de artifacts/):
MODEL_PATH=artifacts/modelo.onnx BASELINE_PATH=artifacts/baseline.json \
  streamlit run app/app.py
# → http://localhost:8501
```

Sin `LOG_BUCKET` definido, las predicciones no se envían a S3 (modo local).

---

## El modelo

- **Dataset:** Breast Cancer Wisconsin (`sklearn`), 30 características, 2 clases.
- **Modelo:** `StandardScaler → LogisticRegression` (lineal → conversión ONNX
  exacta; el escalado va dentro del modelo, así la app envía features crudas).
- **Métrica:** *accuracy* en test ≈ **0.98**; umbral de promoción **0.90**.
