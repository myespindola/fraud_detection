Aquí tienes un README.md en formato Markdown listo para tu proyecto de Detección de Fraude:

# Proyecto: Detección de Fraude

Este proyecto implementa un pipeline de **Machine Learning** para la detección de fraude. Utiliza **MLflow** para el tracking de experimentos, **Jupyter** para desarrollo y análisis, y una **API** para servir modelos entrenados. Todo se ejecuta en **contenedores Docker** para portabilidad y reproducibilidad.

---

## Estructura del Proyecto



.
├── mlflow/ # Dockerfile y configuración de MLflow
│ └── requirements.txt
├── jupyter/ # Dockerfile y notebooks de análisis
│ ├── requirements.txt
│ └── notebooks/
├── api/ # Dockerfile para API de modelos
│ └── ...
├── data/ # Datos de entrenamiento y prueba
├── docker-compose.yml
└── README.md


---

## Requisitos

- Docker >= 24
- Docker Compose >= 3.9

---

## Configuración y Despliegue

Construir y levantar los contenedores:

```bash
docker-compose up --build


Esto creará tres servicios:

mlflow: Servidor MLflow en http://localhost:5000

jupyter: Jupyter Notebook en http://localhost:8888 (token: admin)

api: API para servir modelos ML en el puerto 8000

Verificar directorios y permisos

Asegúrate que la carpeta ./mlflow tenga permisos de lectura/escritura:

mkdir -p ./mlflow/artifacts
chmod -R 777 ./mlflow


Nota: Para producción se recomienda cambiar 777 por permisos específicos del usuario.

Acceso a los Servicios

MLflow: http://localhost:5000

Jupyter: http://localhost:8888 (token: admin)

API: http://localhost:8000

Uso de MLflow

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

Jupyter Notebooks

Los notebooks se encuentran en /home/jovyan/work/notebooks dentro del contenedor.
Se pueden usar los datos en ./data como volumen montado.

Modelo Base
Carga de datos

Dataset: fraud_train.csv en data/

Columnas relevantes:

'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'MonthClaimed',
'WeekOfMonthClaimed', 'MaritalStatus', 'Fault', 'PolicyType',
'VehicleCategory', 'VehiclePrice', 'Deductible', 'PastNumberOfClaims',
'AgeOfVehicle', 'AgeOfPolicyHolder', 'AgentType', 'NumberOfSuppliments',
'AddressChange_Claim', 'BasePolicy', 'FraudFound_P'

Preprocesamiento

Agrupación de autos de lujo: 'Porche', 'Ferrari', 'Mecedes' → 'Luxury'

Separación de variables predictoras y target:

X = df_final.drop('FraudFound_P', axis=1)
y = df_final['FraudFound_P']


Validación del 10% del dataset, estratificada por la variable objetivo.

Optimización de Hiperparámetros con Optuna

Pipeline con WOEEncoder y LogisticRegression

Validación cruzada StratifiedKFold 5 folds

Métrica: F-beta (β=2)

Hiperparámetros: C y max_iter

F-beta con β>1 prioriza recall, útil para detectar todos los casos de fraude.

Entrenamiento del Modelo Final

Pipeline:

WOEEncoder: transforma variables categóricas en valores WOE

LogisticRegression: class_weight="balanced"

Generación de Scorecard

Fórmula de score individual:

odds = prob / (1 - prob)
score = offset - factor * np.log(odds)


Permite interpretar la contribución de cada variable al riesgo de fraude.

Evaluación del Modelo

Métricas en validación:

AUC

KS

F-beta (β=2)

Matriz de confusión con ConfusionMatrixDisplay

Probabilidades convertidas a predicciones binarias con threshold 0.5

Registro con MLflow

Parámetros del modelo (best_params)

Métricas (AUC, KS, F-beta)

Artefactos:

Modelo serializado (lr_model.pkl)

Scorecard (lr_base_scorecard.csv)

Matriz de confusión (lr_base_confusion_matrix.png)

Uso de la API
Request

Content-Type: application/json

Body: lista de diccionarios con columnas requeridas

{
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
}


Puedes enviar varias filas para predicciones batch.

Ejemplo PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" `
-Body '{ "features": [ { "ID": "CL00007646", "Month": "Aug", "DayOfWeek": "Friday", "Make": "Honda", "AccidentArea": "Urban", "MonthClaimed": "Aug", "WeekOfMonthClaimed": 5, "MaritalStatus": "Married", "Fault": "Policy Holder", "PolicyType": "Sedan - All Perils", "VehicleCategory": "Sedan", "VehiclePrice": "30000 to 39000", "Deductible": 400, "PastNumberOfClaims": 1, "AgeOfVehicle": "7 years", "AgeOfPolicyHolder": "36 to 40", "AgentType": "External", "NumberOfSuppliments": "1 to 2", "AddressChange_Claim": "4 to 8 years", "BasePolicy": "All Perils", "FraudFound_P": 0 } ] }'

Respuesta
{
  "predictions": [0]
}


Cada elemento corresponde al resultado del modelo (0 = no fraude, 1 = fraude).
