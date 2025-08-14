from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# ----- Cargar modelo al iniciar -----
MODEL_PATH = "/mlflow/artifacts/1/5324911c0fa649798464866bb60214d1/artifacts/lr_baselr_model.pkl"
model = joblib.load(MODEL_PATH)

# ----- Columnas que vamos a usar para predecir -----
SELECTED_COLUMNS = [
    'Month',
    'DayOfWeek',
    'Make',
    'AccidentArea',
    'MonthClaimed',
    'WeekOfMonthClaimed',
    'MaritalStatus',
    'Fault',
    'PolicyType',
    'VehicleCategory',
    'VehiclePrice',
    'Deductible',
    'PastNumberOfClaims',
    'AgeOfVehicle',
    'AgeOfPolicyHolder',
    'AgentType',
    'NumberOfSuppliments',
    'AddressChange_Claim',
    'BasePolicy',
    'FraudFound_P'
]

# ----- Diccionario de reemplazo para Make -----
REPLACE_MAP = {
    'Porche': 'Luxyry',
    'Ferrari': 'Luxyry',
    'Mecedes': 'Luxyry'
}

# ----- Definir app y modelo de datos -----
app = FastAPI(title="MLflow Prediction API")

def score_from_prob(prob, factor, offset):
    odds = prob / (1 - prob)
    return offset - factor * np.log(odds)
class InputData(BaseModel):
    features: list[dict]  # Lista de diccionarios con nombres de columnas

# ----- Endpoint de predicción -----
@app.post("/predict")
def predict(data: InputData):
    # Convertir a DataFrame
    df = pd.DataFrame(data.features)
    
    # Reemplazar valores en 'Make'
    df['Make'] = df['Make'].replace(REPLACE_MAP)
    
    # Filtrar solo las columnas seleccionadas
    df = df[SELECTED_COLUMNS]
    
    # Separar features de target si es necesario
    X = df.drop(columns=["FraudFound_P"], errors="ignore")
    
    # Predecir
    preds = model.predict(X).tolist()
    return {"predictions": preds}
