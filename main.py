from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
from predecir import predecir_credito  # o importar local si está en el mismo archivo

app = FastAPI()

# 🔐 CORS: permitir acceso desde tu frontend en Netlify
origins = [
    "https://mellow-bavarois-164ff2.netlify.app",  # Tu sitio en producción
    "http://localhost:5173"                        # Opcional para pruebas locales
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧾 Modelo del JSON que espera el endpoint
class InputCliente(BaseModel):
    edad: int
    sexo: str
    region: str
    estado_civil: str
    nivel_educacional: str
    personas_a_cargo: str
    trabaja_actualmente: str
    tipo_contrato: str
    antiguedad_empleo: str
    tramo_ingresos: str
    producto_financiero: Union[str, List[str]]
    instituciones_financieras: str
    pago_mensual: str
    solicito_credito_ult_6m: str

# 🎯 Endpoint para predicción
@app.post("/predict")
def predict_credito(data: InputCliente):
    return predecir_credito(data.dict())
