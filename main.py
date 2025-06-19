from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union
from predecir import predecir_credito  # o simplemente importa directo si está en el mismo archivo

app = FastAPI()

# 🚨 Cambia esta URL por la de tu frontend
origins = [
    "https://tu-frontend.web.app",     # frontend en Firebase
    "https://tusitio.netlify.app",     # ejemplo Netlify
    "http://localhost:3000",           # desarrollo local
    "*"                                # ⚠️ usar solo en pruebas, no en producción
]

# Agregar middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # puedes usar ["*"] para permitir todo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Entrada esperada
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

# Endpoint
@app.post("/predict")
def predict_credito(data: InputCliente):
    return predecir_credito(data.dict())
