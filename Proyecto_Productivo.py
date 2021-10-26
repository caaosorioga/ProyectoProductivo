import streamlit as st
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler

st.write("### Esta aplicación simula si ud le podrían arobar o no una tarjeta de crédito")

datos=pd.read_csv('C:\\Users\\Carlos\\Downloads\\credit-approval.csv')
atributos = ['Genero', 'Edad', 'Deuda', 'Casado', 'ClienteBanco', 'NivelEstudios', 'TiempoEmpleado', 'IncumplimientosPrevios', 'Empleado', 'PuntajeCredito', 'LicenciaConduccion', 'Ciudadania', 'Ingresos']
columnas_x = datos[atributos]
x = columnas_x.values
scaler = StandardScaler().fit(x)
col1, col2, col3 = st.columns((2,1,2))

x_usuario = np.zeros(len(atributos))

with col1:

    for i in range(len(atributos)):
        x_usuario[i] = st.number_input(atributos[i],step=1)

with col3:
    Red_neuronal = keras.models.load_model('modelo_arbol.joblib')
    x = x_usuario.reshape(1,-1)
    x2 = scaler.transform(x)

    y_pred = Red_neuronal.predict(x2)

st.write("Resultado del ejercicio = ", y_pred)

#C:\Users\Carlos\Documents\Pascual Bravo\Examen Modulo 2\Proyecto_Productivo
