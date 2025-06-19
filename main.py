from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

modelo = joblib.load("modelo_rf.pkl")
encoder = joblib.load("onehot_encoder.pkl")
cols_productos = joblib.load("columnas_productos.pkl")

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

app = FastAPI()

@app.get("/")
def home():
    return {"mensaje": "API de evaluación crediticia activa"}

@app.post("/predict")
def predict(data: SolicitudCredito):
    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])
    productos = df_input["producto_financiero"].apply(lambda x: ', '.join(x))
    dummies_productos = productos.str.get_dummies(sep=', ')
    for col in cols_productos:
        if col not in dummies_productos:
            dummies_productos[col] = 0
    dummies_productos = dummies_productos[cols_productos]
    df_input = df_input.drop(columns=["producto_financiero"])
    cat_cols = encoder.feature_names_in_
    encoded = encoder.transform(df_input[cat_cols])
    df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    df_final = pd.concat([
        df_input.drop(columns=cat_cols).reset_index(drop=True),
        df_encoded.reset_index(drop=True),
        dummies_productos.reset_index(drop=True)
    ], axis=1)
    pred = modelo.predict(df_final)[0]
    return {"otorgar_credito": int(pred)}
