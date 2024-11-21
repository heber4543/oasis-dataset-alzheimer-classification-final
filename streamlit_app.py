# importar librerias
import streamlit as st
import joblib
import pandas as pd
# cargar el pipeline
preprocessor = joblib.load('modelo/archivos/pipeline_373.pkl')
# cargar los modelos
lr_150 = joblib.load('modelo/archivos/best_lr_model_150.pkl')
lr_373 = joblib.load('modelo/archivos/best_lr_model_373.pkl')
svm_150 = joblib.load('modelo/archivos/best_svm_model_150.pkl')
svm_373 = joblib.load('modelo/archivos/best_svm_model_373.pkl')
rf_150 = joblib.load('modelo/archivos/best_rf_model_150.pkl')
rf_373 = joblib.load('modelo/archivos/best_rf_model_373.pkl')
# titulo
st.title("OASIS DATASET - ALZHEIMER PREDICTION")
# descripcion del dataset
descripcion = """
The Open Access Series of Imaging Studies (OASIS) is a series of
neuroimaging data sets that are publicly available for study and
analysis. The present MRI data set consists of a longitudinal collection of 150 subjects aged 60 to 96 years, all acquired on the same
scanner using identical sequences. Each subject was scanned
on two or more visits, separated by at least 1 year, for a total of
373 imaging sessions (Marcus et al., 2009).
"""
st.markdown(descripcion)
# descripcion de la implementacion
st.subheader("IMPLEMENTATION")
app = """
This app implements two classification models: Support Vector Machine (SVM) and Logistic Regression. 
The same dataset is used in two variations: the first variation includes only the first visit of each patient, 
as reported in the literature (150 patients). The second variation treats each visit as a distinct patient, 
resulting in a total of 373 patients. Consequently, there are four models: a) Logistic Regression trained 
with 150 patients, b) Logistic Regression trained with 373 patients, c) SVM trained with 150 patients, 
and d) SVM trained with 373 patients.
"""
st.markdown(app)
# entradas del usuario
st.subheader("ENTER YOUR DATA")
mr_delay = st.number_input("MR Delay", value = 576)
age = st.number_input("Age", value = 69)
educ = st.number_input("Years of Education", value = 12)
ses = st.number_input("Socioeconomic Status", value = 2)
mmse = st.number_input("Mini-Mental State Examination Score", value = 24)
cdr = st.number_input("Clinical Dementia Rating", value = 0.5)
etiv = st.number_input("Estimated Total Intracranial Volume", value = 1480)
nwbv = st.number_input("Normalized Whole-Brain Volume", value = 0.791)
sex = st.selectbox("Sex", ['M', 'F'], index=0)
asf = st.number_input("Atlas Scaling Factor", value = 1.186)
# crear dataFrame con las entradas del usuario
input_df = pd.DataFrame({
    'MR Delay': [mr_delay],
    'Age': [age],
    'EDUC': [educ],
    'SES': [ses],
    'MMSE': [mmse],
    'CDR': [cdr],
    'eTIV': [etiv],
    'nWBV': [nwbv],
    'ASF': [asf],
    'M/F': [sex]  
})
# seleccion del modelo
models = {
    "Linear Regression with 150 dataset": lr_150,
    "Linear Regression with 373 dataset": lr_373,
    "Support Vector Machine with 150 dataset": svm_150,
    "Support Vector Machine with 373 dataset": svm_373,
    "Random Forest with 150 dataset": rf_150,
    "Random Forest with 373 dataset": rf_373
}
selected_model = st.selectbox("Select the model you prefer", list(models.keys()))
# predicción
if st.button("Predict"):
    try:
        # aplicar el preprocesador a los datos de entrada
        input_processed = preprocessor.transform(input_df) 
        # seleccionar el modelo
        model = models[selected_model]
        # predicción
        prediction = model.predict(input_processed)
        # mostrar la predicción
        if prediction[0] == 0:
            st.write("The prediction of {} is: Non Demented".format(selected_model))
        else:
            st.write("The prediction of {} is: Demented".format(selected_model))
    except Exception as e:
        st.error(f"Prediction error: {e}")