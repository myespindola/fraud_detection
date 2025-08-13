import mlflow
from mlflow.exceptions import RestException
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_NAME = "iris-classifier"
MODEL_STAGE = "Production"
model = None

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        print(f"Modelo: {MODEL_NAME} cargado desde stage: {MODEL_STAGE}")
    except RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            print(f"No existe el modelo: {MODEL_NAME} en stage: {MODEL_STAGE}")
            model = None
        else:
            print(f"Error cargando modelo: {e}")
            model = None
    except Exception as e:
        print(f"Error inesperado cargando modelo: {e}")
        model = None

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "No hay modelo en produccion disponible"}
    prediction = model.predict([data.features])
    return {"prediction": prediction.tolist()}
