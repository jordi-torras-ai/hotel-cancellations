# INN Hotel Intelligence · Python + HTML

Equivalente local del [notebook de Jordi Torras](https://colab.research.google.com/drive/1cfzchjeurXm6Y6jJT84Bu1Ioz2g3iffw). Una aplicación Flask con HTML, CSS y JavaScript nativos, sin Colab, Gradio, Node, CDN ni servicios externos en tiempo de ejecución. La interfaz conserva el idioma español del material del curso.

## Iniciar

Después de clonar el repositorio:

```bash
cd hotel-cancellations
source ~/myenv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Abre **http://127.0.0.1:5055**. También puedes usar `./run.sh`, que activa `~/myenv` automáticamente. Puedes lanzarlo desde cualquier carpeta con su ruta completa. Si el servidor de esta aplicación ya está funcionando en ese puerto, el script muestra la URL y termina sin iniciar otro. Para detener el servidor, pulsa Ctrl+C en la terminal que lo inició. Para otro puerto: `./run.sh --port 5056`.

En otro ordenador, utiliza Python 3.11 o posterior, crea un entorno virtual, actívalo e instala `requirements.txt`. En Windows, la activación de un entorno local es `.venv\Scripts\activate`. Ejecuta `python app.py` desde esta carpeta. La primera ejecución entrena y guarda el modelo automáticamente; las siguientes reutilizan el artefacto si coinciden el CSV, la revisión del pipeline y la versión de scikit-learn.

Para macOS/Linux sin el entorno `~/myenv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run.sh
```

El selector **Texto** permite elegir Normal, Grande o Muy grande. La preferencia se conserva en este navegador.

## Qué incluye

| Pantalla | Equivalencia con el notebook |
| --- | --- |
| Vista general | Cancelaciones, precio medio, volumen mensual, antelación y canal |
| Explorar datos | CSV paginado y filtrable, calidad de datos, diccionario y preparación |
| El modelo | Árbol inicial y ajustado, métricas reales, matriz de confusión, importancia, búsqueda de parámetros y reentrenamiento |
| Predecir reservas | Los nueve campos de Gradio, probabilidad de cancelación y reglas del árbol |
| Predicción por lotes | CSV de entrada, ejemplo descargable y CSV de predicciones |
| Guía del proyecto | Caso empresarial, flujo completo y escenario financiero editable |

Los gráficos históricos y la evaluación se calculan con el CSV incluido. Los gráficos de seguimiento del notebook eran ilustrativos: no se presentan como resultados de producción. La simulación financiera está identificada como hipotética.

## Datos y reproducibilidad

`data/INNHotelsGroup.csv` es una copia del CSV original encontrado en los materiales locales del curso EADA, `EADAX/Colab/INNHotelsGroup.csv`. Tiene 36.275 filas y las mismas diez columnas que la vista previa del notebook. Se conserva el orden, los valores y las filas repetidas. No se descargó un dataset alternativo ni se generaron datos sintéticos. El SHA-256 del archivo se registra en `artifacts/report.json`.

La preparación utiliza exactamente nueve variables, en el orden del notebook:

1. `lead_time`
2. `no_of_special_requests`
3. `avg_price_per_room`
4. `no_of_adults`
5. `no_of_weekend_nights`
6. `required_car_parking_space`
7. `no_of_week_nights`
8. `arrival_month`, derivado de `arrival_date`
9. `market_segment_type_Online`: Online = 1, Offline = 0

Objetivo: Canceled = 1, Not_Canceled = 0. División aleatoria 70/30, `random_state=42`, sin estratificación, igual que en Colab. Árbol inicial: `DecisionTreeClassifier(random_state=1)`. Ajuste: árbol con `class_weight="balanced"`, búsqueda sobre `max_leaf_nodes=[150,250]` y `min_samples_split=[10,30]`, `cv=5`, puntuación F1. El conjunto de prueba no participa en la búsqueda. El modelo guardado se entrena con el 70%; no se reajusta después con las etiquetas de prueba.

Resultado verificado con scikit-learn 1.8.0:

| Modelo | F1 prueba |
| --- | --- |
| Inicial | 79,37% |
| Ajustado | 80,20% |

25.392 reservas para entrenamiento y 10.883 para prueba. Mejores parámetros: 250 hojas y un mínimo de 10 muestras para dividir. La pequeña diferencia del árbol inicial respecto al 79,30% guardado en Colab puede depender de la versión de scikit-learn; el ajustado reproduce el 80,20% mostrado.

La probabilidad se obtiene localizando explícitamente la **clase 1** en `classes_`. Esto corrige el cálculo de la función original, que comparaba una lista con `1` y podía mostrar la probabilidad de la clase equivocada. La clase predicha usa `model.predict`, respetando también los empates del clasificador.

## Entrenamiento y archivos

```bash
source ~/myenv/bin/activate
python train.py
python app.py --data /ruta/INNHotelsGroup.csv --artifacts /ruta/artefactos
```

También se puede reentrenar con el botón de la pantalla **El modelo**. Se guardan `artifacts/model.joblib` e `artifacts/report.json`. Los CSV personalizados deben seguir el esquema del original y conservar las dos clases. No se cargan archivos pickle proporcionados por visitantes.

Los artefactos se generan localmente y no se versionan. El repositorio incluye el código, los datos originales y las pruebas; no hace falta descargar un modelo para iniciar.

```text
app.py                  Servidor, vistas y API
domain.py               Esquema compartido, validación y codificación
train.py                Entrenamiento, métricas y artefactos
templates/              Vistas HTML con Jinja
static/                 CSS y JavaScript sin dependencias
data/INNHotelsGroup.csv  Datos originales del curso
artifacts/              Modelo generado e informe de evaluación
tests/test_app.py       Pruebas con el modelo y los datos reales
```

## CSV por lotes y API

El CSV de predicción acepta los nueve campos de entrada sin codificar, con `market_segment_type` = Online/Offline y aparcamiento = 0/1. Puede incluir `arrival_date` en formato `AAAA-MM-DD` en lugar de `arrival_month`. Descarga un ejemplo desde el predictor. El CSV de salida conserva los campos de entrada normalizados y añade `predicted_booking_status`, `cancellation_probability` (0–1) y `booking_value_eur`. Se omiten otras columnas, incluidos identificadores. Límite: 50.000 filas y 10 MB. El archivo se procesa en memoria y no se guarda.

| Ruta | Método | Uso |
| --- | --- | --- |
| `/api/predict` | POST JSON | Una reserva |
| `/api/batch` | POST multipart, campo `file` | CSV de reservas |
| `/api/train` | POST | Reentrenar el CSV configurado |
| `/api/report` | GET | Métricas y metadatos |
| `/api/data?page=1&month=7&status=Canceled` | GET | Filtrar y paginar datos |
| `/download/sample`, `/download/data`, `/download/report`, `/download/model` | GET | Descargas |

## Comprobar

```bash
source ~/myenv/bin/activate
python -m pip install pytest
python -m pytest -q
```

Las pruebas comparan la codificación con la del notebook, verifican el F1 de prueba, los dos resultados posibles de predicción, probabilidades individuales y por lotes, fechas, errores de validación, descargas, filtros y reentrenamiento.

## Interpretación

F1 mide la clase de cancelación y no equivale a exactitud. Las probabilidades de un árbol con pesos equilibrados no están calibradas. La exposición estimada es probabilidad × precio × noches; no mide una pérdida efectiva. La evaluación aleatoria del notebook no estima automáticamente el rendimiento futuro: las reservas repetidas pueden aparecer en ambos conjuntos y aumentar el optimismo. Antes de uso operativo se necesitaría una evaluación temporal, calibración y datos recientes. Esta entrega funciona en localhost como proyecto docente; no se ha publicado ni conectado a un sistema de reservas.
