Proyecto: Detección de Fraude

Este proyecto es un pipeline de machine learning para detección de fraude, utilizando MLflow para el tracking de experimentos, Jupyter para desarrollo y análisis, y una API para servir modelos entrenados. Todo se ejecuta en contenedores Docker para portabilidad y reproducibilidad.

Estructura del proyecto
.
├── mlflow/                 # Dockerfile y configuración de MLflow
│   └── requirements.txt
├── jupyter/                # Dockerfile y notebooks de análisis
│   ├── requirements.txt
│   └── notebooks/
├── api/                    # Dockerfile para API de modelos
│   └── ...
├── data/                   # Datos de entrenamiento y prueba
├── docker-compose.yml
└── README.md

Requisitos

Docker >= 24

Docker Compose >= 3.9

Configuración y despliegue

Construir y levantar contenedores:

docker-compose up --build


Esto creará tres servicios:

mlflow: Servidor MLflow en http://localhost:5000

jupyter: Jupyter Notebook en http://localhost:8888 (token: admin)

api: API para servir modelos ML (puerto 8000)

Verificar directorios y permisos

Asegúrate que la carpeta ./mlflow tenga permisos de lectura/escritura:

mkdir -p ./mlflow/artifacts
chmod -R 777 ./mlflow


Nota: Para producción se recomienda cambiar 777 por permisos específicos del usuario.

Acceder a los servicios

MLflow: http://localhost:5000

Jupyter: http://localhost:8888 (token: admin)

API: http://localhost:8000

Uso
MLflow

Guardar parámetros, métricas y artefactos de modelos:

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("fraud_detection")

with mlflow.start_run(run_name="LR_Scorecard"):
    mlflow.log_params(best_params)
    mlflow.log_metric("AUC", auc)
    mlflow.log_metric("KS", ks)
    mlflow.log_metric("Fbeta", fb)
    mlflow.sklearn.log_model(pipeline, name="LR_Scorecard")

Jupyter

Los notebooks se encuentran en /home/jovyan/work/notebooks dentro del contenedor.

Puedes usar los datos de ./data como volumen montado.
Modelo Base

1. Carga de datos

Se importa el dataset fraud_train.csv desde la carpeta data/.

Se seleccionan únicamente las columnas relevantes según el análisis exploratorio previo:

'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'MonthClaimed',
'WeekOfMonthClaimed', 'MaritalStatus', 'Fault', 'PolicyType',
'VehicleCategory', 'VehiclePrice', 'Deductible', 'PastNumberOfClaims',
'AgeOfVehicle', 'AgeOfPolicyHolder', 'AgentType', 'NumberOfSuppliments',
'AddressChange_Claim', 'BasePolicy', 'FraudFound_P'

2. Preprocesamiento de datos

Se reemplazan ciertas categorías en la variable Make para agrupar autos de lujo:

'Porche', 'Ferrari', 'Mecedes' → 'Luxury'

Se separan las variables predictoras (X) del target (y):

X = df_final.drop('FraudFound_P', axis=1)
y = df_final['FraudFound_P']


División en conjunto de entrenamiento y validación:

Validación del 10% del dataset.

Estratificación por la variable objetivo para mantener proporciones de fraude.

3. Optimización de hiperparámetros con Optuna

Se define una función objetivo que:

Crea un pipeline con WOEEncoder y LogisticRegression.

Realiza validación cruzada (StratifiedKFold) de 5 folds.

Predice probabilidades y aplica un threshold de 0.5.

Calcula F-beta score (β=2) como métrica a maximizar.

Hiperparámetros optimizados:

C: regularización de la regresión logística.

max_iter: número máximo de iteraciones.

Ventaja: F-beta con β>1 prioriza recall, útil para detectar todos los casos de fraude, aunque genere algunos falsos positivos.

4. Entrenamiento del modelo final

Se entrena un pipeline con los mejores hiperparámetros encontrados.

Pipeline:

WOEEncoder: transforma variables categóricas en valores WOE.

LogisticRegression: modelo principal con class_weight="balanced" para manejar desbalanceo.

5. Generación de Scorecard

Se calculan los puntos por variable y categoría según:

points_per_category = {k: -coef[i] * factor * v for k, v in woe_map.items()}


Fórmula de score individual a partir de la probabilidad:

odds = prob / (1 - prob)
score = offset - factor * np.log(odds)


Permite interpretar la contribución de cada variable al riesgo de fraude.

6. Evaluación del modelo

Se generan las siguientes métricas en el conjunto de validación:

AUC: Área bajo la curva ROC.

KS: Kolmogorov-Smirnov.

F-beta (β=2).

Matriz de confusión visualizada usando ConfusionMatrixDisplay.

Probabilidades del modelo convertidas a predicciones binarias con threshold de 0.5.

7. Registro con MLflow

Configuración de MLflow Tracking:

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("fraud_detection")


Se registran:

Parámetros del modelo (best_params).

Métricas (AUC, KS, F-beta).

Artifacts:

Modelo serializado (lr_model.pkl)

Scorecard (lr_base_scorecard.csv)

Matriz de confusión (lr_base_confusion_matrix.png)

8. Visualizaciones

Matriz de confusión para inspección visual del desempeño.

Posibilidad de graficar ROC y KS si se desea en futuras versiones.

Uso API

Request

Content-Type: application/json

Body:
Lista de diccionarios (features) con las columnas requeridas por el modelo.

{
  "features": [
    {
      "ID": "CL00007646",
      "Month": "Aug",
      "WeekOfMonth": 4,
      "DayOfWeek": "Friday",
      "Make": "Honda",
      "AccidentArea": "Urban",
      "DayOfWeekClaimed": "Monday",
      "MonthClaimed": "Aug",
      "WeekOfMonthClaimed": 5,
      "Sex": "Male",
      "MaritalStatus": "Married",
      "Fault": "Policy Holder",
      "PolicyType": "Sedan - All Perils",
      "VehicleCategory": "Sedan",
      "VehiclePrice": "30000 to 39000",
      "FraudFound_P": 0,
      "Deductible": 400,
      "Days_Policy_Accident": "more than 30",
      "Days_Policy_Claim": "more than 30",
      "PastNumberOfClaims": 1,
      "AgeOfVehicle": "7 years",
      "AgeOfPolicyHolder": "36 to 40",
      "PoliceReportFiled": "No",
      "WitnessPresent": "No",
      "AgentType": "External",
      "NumberOfSuppliments": "1 to 2",
      "AddressChange_Claim": "4 to 8 years",
      "NumberOfCars": "2 vehicles",
      "Year": 1994,
      "BasePolicy": "All Perils"
    }
  ]
}


Nota: puedes enviar varias filas en la lista "features" para predicciones batch.

Ejemplo con PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "features": [
      {
        "ID": "CL00007646",
        "Month": "Aug",
        "DayOfWeek": "Friday",
        "Make": "Honda",
        "AccidentArea": "Urban",
        "MonthClaimed": "Aug",
        "WeekOfMonthClaimed": 5,
        "MaritalStatus": "Married",
        "Fault": "Policy Holder",
        "PolicyType": "Sedan - All Perils",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "30000 to 39000",
        "Deductible": 400,
        "PastNumberOfClaims": 1,
        "AgeOfVehicle": "7 years",
        "AgeOfPolicyHolder": "36 to 40",
        "AgentType": "External",
        "NumberOfSuppliments": "1 to 2",
        "AddressChange_Claim": "4 to 8 years",
        "BasePolicy": "All Perils",
        "FraudFound_P": 0
      }
    ]
  }'

Respuesta
{
  "predictions": [0]
}


"predictions" devuelve la lista de predicciones para cada fila enviada.

Cada elemento corresponde al resultado del modelo (por ejemplo, 0 = no fraude, 1 = fraude).