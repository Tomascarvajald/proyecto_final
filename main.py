from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle

# Cargar modelo, encoder y columnas
with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("columnas.pkl", "rb") as f:
    columnas_finales = pickle.load(f)

with open("productos.pkl", "rb") as f:
    cols_productos = pickle.load(f)

# Inicializar app
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir input JSON
class SolicitudCredito(BaseModel):
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
    producto_financiero: list[str]
    instituciones_financieras: str
    pago_mensual: str
    solicito_credito_ult_6m: str

# Ruta raíz de prueba
@app.get("/")
def home():
    return {"mensaje": "API de evaluación crediticia activa"}

# Endpoint de predicción
@app.post("/predict")
def predict(data: SolicitudCredito):
    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])

    # Procesar productos financieros
    productos = df_input["producto_financiero"].apply(lambda x: ', '.join(x))
    dummies_productos = productos.str.get_dummies(sep=', ')
    for col in cols_productos:
        if col not in dummies_productos:
            dummies_productos[col] = 0
    dummies_productos = dummies_productos[cols_productos]
    df_input = df_input.drop(columns=["producto_financiero"])

    # Codificar variables categóricas
    cat_cols = encoder.feature_names_in_
    encoded = encoder.transform(df_input[cat_cols])
    df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    # Armar dataset final
    df_final = pd.concat([
        df_input.drop(columns=cat_cols).reset_index(drop=True),
        df_encoded.reset_index(drop=True),
        dummies_productos.reset_index(drop=True)
    ], axis=1)

    # Predicción
    pred = modelo.predict(df_final)[0]
    return {"otorgar_credito": int(pred)}
