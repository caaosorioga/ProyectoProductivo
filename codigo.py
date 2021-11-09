import streamlit as st
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler

st.write("### Esta aplicación simula si a ud le podrían aprobar o no una tarjeta de crédito")

datos=pd.read_csv('creditapproval.csv')

datos_drop=datos.drop(["CodigoZip","GrupoPobl", "Genero", "Edad", "Deuda", "Casado", "ClienteBanco", "NivelEstudios", "Empleado", "LicenciaConduccion", "Ciudadania"],axis=1)

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for col in datos_drop:
    if datos_drop[col].dtypes=='object':
        datos_drop[col]=LE.fit_transform(datos_drop[col])

datos_drop.fillna(datos_drop.mean(), inplace=True)

atributos = ['TiempoEmpleado', 'IncumplimientosPrevios','PuntajeCredito','Ingresos']
columnas_x = datos_drop[atributos]
x = columnas_x.values

scaler = StandardScaler().fit(x)
st.image('tarjeta.jpg')

col1, col2, col3 = st.columns((2,1,2))

x_usuario = np.zeros(len(atributos))

with col1:

    for i in range(len(atributos)):
        x_usuario[i] = st.number_input(atributos[i],step=1)

with col3:
    Red_neuronal = joblib.load('modelo_arbol.joblib')
    x = x_usuario.reshape(1,-1)
    x2 = scaler.transform(x)

    y_pred = Red_neuronal.predict(x2)

if y_pred == 0:
    Resultado="Negada"
else:
    Resultado="Aprobada"


st.write("## Resultado del ejercicio = ", Resultado)

#C:\Users\Carlos\Documents\Pascual Bravo\Examen Modulo 2\Proyecto_Productivo
