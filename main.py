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

def predecir_credito(input_json):
    import pickle
    import pandas as pd

    # Cargar modelo y objetos auxiliares
    modelo = pickle.load(open("modelo.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    cols_productos = pickle.load(open("productos.pkl", "rb"))

    # Convertir JSON a DataFrame
    df_input = pd.DataFrame([input_json])

    # Procesar productos financieros
    productos_cliente = df_input["producto_financiero"]
    productos_separados = productos_cliente.apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    dummies_productos = productos_separados.str.get_dummies(sep=', ')

    # Asegurar columnas de productos
    for col in cols_productos:
        if col not in dummies_productos.columns:
            dummies_productos[col] = 0
    dummies_productos = dummies_productos[cols_productos]

    df_input = df_input.drop(columns=["producto_financiero"])

    # Codificar variables categóricas
    cat_cols = encoder.feature_names_in_
    cat_encoded = encoder.transform(df_input[cat_cols])
    df_cat_encoded = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))

    # Ensamblar input final
    df_input_final = pd.concat([
        df_input.drop(columns=cat_cols).reset_index(drop=True),
        df_cat_encoded.reset_index(drop=True),
        dummies_productos.reset_index(drop=True)
    ], axis=1)

    # Asegurar columnas esperadas por el modelo
    for col in modelo.feature_names_in_:
        if col not in df_input_final.columns:
            df_input_final[col] = 0

    df_input_final = df_input_final.loc[:, ~df_input_final.columns.duplicated()]
    df_input_final = df_input_final[list(modelo.feature_names_in_)]

    # Predicción binaria y probabilidad
    prob_default = modelo.predict_proba(df_input_final)[0][1]
    return {
        "otorgar_credito": int(prob_default < 0.5),  # se otorga si probabilidad de default < 0.5
        "probabilidad_default": round(prob_default, 3)
    }



# 🎯 Endpoint para predicción
@app.post("/predict")
def predict_credito(data: InputCliente):
    return predecir_credito(data.dict())
