import pandas as pd
import streamlit as st
import numpy as np

#### Model ####
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('Iris.csv')

X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']

logreg = LogisticRegression()
logreg.fit(X, y)

#### Model ####

st.header("Iris Veriseti Tahminleme Uygulaması")
st.dataframe(df)  # Same as st.write(df)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.text("SepalLengthCm")
    SepalLengthCm = st.text_input("",value=1, key="SepalLengthCm")

with col2:
    st.text("SepalWidthCm")
    SepalWidthCm = st.text_input("",value=1, key="SepalWidthCm")

with col3:
    st.text("PetalLengthCm")
    PetalLengthCm = st.text_input("",value=1, key="PetalLengthCm")

with col4:
    st.text("PetalWidthCm")
    PetalWidthCm = st.text_input("",value=1, key="PetalWidthCm")
    
def predictionFunction(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):

    test = pd.DataFrame()
    test['SepalLengthCm'] = [(SepalLengthCm)]
    test['SepalWidthCm'] = [SepalWidthCm]
    test['PetalLengthCm'] = [PetalLengthCm]
    test['PetalWidthCm'] = [PetalWidthCm]

    st.header("Tahmin Sonucu : "+logreg.predict(test))

st.button("Tahmin Et", on_click=predictionFunction,args= (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, ))
