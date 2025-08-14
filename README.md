🕵️‍♂️ Proyecto: Detección de Fraude
Este proyecto implementa un pipeline de Machine Learning para la detección de fraude. El flujo de trabajo completo se gestiona con herramientas y tecnologías modernas para garantizar la portabilidad y reproducibilidad.

MLflow: Para el seguimiento de experimentos, modelos y métricas.

Jupyter: Como entorno de desarrollo y análisis exploratorio.

API: Para servir los modelos entrenados y realizar predicciones en tiempo real.

Docker: Todo se ejecuta en contenedores para una configuración uniforme y reproducible.

📂 Estructura del Proyecto
.
├── mlflow/              # Configuración y Dockerfile de MLflow
│ └── requirements.txt
├── jupyter/             # Notebooks y Dockerfile de Jupyter
│ ├── requirements.txt
│ └── notebooks/
├── api/                 # API para servir el modelo
│ └── ...
├── data/                # Datos de entrenamiento y prueba
├── docker-compose.yml   # Archivo para orquestar los contenedores
└── README.md            # Este archivo
⚙️ Requisitos
Asegúrate de tener instalados los siguientes componentes:

Docker: v24 o superior.

Docker Compose: v3.9 o superior.

🚀 Configuración y Despliegue
Levantar los Contenedores
Para construir y levantar todos los servicios, ejecuta el siguiente comando en la raíz del proyecto:

Bash

docker-compose up --build
Esto iniciará tres servicios principales:

mlflow: Servidor de MLflow en http://localhost:5000

jupyter: Jupyter Notebook en http://localhost:8888 (con token: admin)

api: API para el modelo ML en http://localhost:8000

Permisos
Antes de iniciar, es crucial asegurarse de que la carpeta de artefactos de MLflow tenga los permisos correctos.

Bash

mkdir -p ./mlflow/artifacts
chmod -R 777 ./mlflow
Nota: Para entornos de producción, se recomienda cambiar 777 por permisos más específicos para el usuario.

💻 Uso de los Servicios
Acceso a los servicios
Servicio	URL de Acceso
MLflow	http://localhost:5000
Jupyter	http://localhost:8888 (token: admin)
API	http://localhost:8000

Exportar a Hojas de cálculo
Uso de MLflow
Puedes registrar tus experimentos y modelos directamente desde tus scripts o notebooks de Jupyter:

Python

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
Los notebooks se encuentran en la ruta /home/jovyan/work/notebooks dentro del contenedor. El directorio data/ está montado como un volumen, permitiendo el acceso a los datos de entrenamiento y prueba.

📊 Modelo Base de Detección de Fraude
Carga de Datos y Preprocesamiento
El modelo utiliza el dataset fraud_train.csv ubicado en la carpeta data/.

Columnas relevantes:

'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'MonthClaimed', 'WeekOfMonthClaimed', 'MaritalStatus', 'Fault', 'PolicyType', 'VehicleCategory', 'VehiclePrice', 'Deductible', 'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder', 'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim', 'BasePolicy', 'FraudFound_P'

Preprocesamiento:

Agrupación de autos de lujo ('Porche', 'Ferrari', 'Mercedes') en la categoría 'Luxury'.

Separación de variables predictoras (X) y la variable objetivo (y): FraudFound_P.

Optimización de Hiperparámetros
Se usa Optuna para optimizar el pipeline de WOEEncoder y LogisticRegression.

Validación Cruzada: Se usa StratifiedKFold con 5 folds.

Métrica de Optimización: F-beta (
beta=2), que prioriza el recall para una mejor detección de casos de fraude.

Hiperparámetros a optimizar: C y max_iter.

Entrenamiento del Modelo Final
El modelo final es un pipeline con:

WOEEncoder: Transforma variables categóricas en valores numéricos.

LogisticRegression: Con el parámetro class_weight="balanced" para manejar el desbalanceo de clases.

Generación de Scorecard
La puntuación individual de cada caso se calcula con la siguiente fórmula, lo que permite una interpretación clara del riesgo de fraude.

score=offset−factor∗np.log(odds)
Evaluación del Modelo
Las métricas de validación registradas son:

AUC

KS

F-beta (
beta=2)

Se genera una Matriz de Confusión y se registran los siguientes artefactos con MLflow:

Modelo serializado (lr_model.pkl)

Scorecard (lr_base_scorecard.csv)

Matriz de confusión (lr_base_confusion_matrix.png)

🌐 Uso de la API para Predicciones
La API está disponible en http://localhost:8000 y permite enviar datos en formato JSON para obtener predicciones.

Request
JSON

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
Ejemplo con PowerShell
PowerShell

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" `
-Body '{ "features": [ { "ID": "CL00007646", "Month": "Aug", "DayOfWeek": "Friday", "Make": "Honda", "AccidentArea": "Urban", "MonthClaimed": "Aug", "WeekOfMonthClaimed": 5, "MaritalStatus": "Married", "Fault": "Policy Holder", "PolicyType": "Sedan - All Perils", "VehicleCategory": "Sedan", "VehiclePrice": "30000 to 39000", "Deductible": 400, "PastNumberOfClaims": 1, "AgeOfVehicle": "7 years", "AgeOfPolicyHolder": "36 to 40", "AgentType": "External", "NumberOfSuppliments": "1 to 2", "AddressChange_Claim": "4 to 8 years", "BasePolicy": "All Perils", "FraudFound_P": 0 } ] }'
Respuesta
La API retorna una lista de predicciones. Un 0 indica "no fraude" y un 1 indica "fraude".

JSON

{
  "predictions": [0]
}
