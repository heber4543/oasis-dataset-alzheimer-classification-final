# 

### This is a project for the Machine Learning course in the Master's program in Computer Engineering

### DATASET
The Open Access Series of Imaging Studies (OASIS) is a series of neuroimaging data sets that are publicly available for study and analysis. The present MRI data set consists of a longitudinal collection of 150 subjects aged 60 to 96 years, all acquired on the same scanner using identical sequences. Each subject was scanned on two or more visits, separated by at least 1 year, for a total of 373 imaging sessions (Marcus et al., 2009).

### IMPLEMENTATION
This app implements six classification models: Support Vector Machine (SVM), Logistic Regression, and Random Forest.
The same dataset is used in two variations: the first variation includes only the first visit of each patient, as reported in the literature (150 patients). The second variation treats each visit as a distinct patient, resulting in a total of 373 patients.

Consequently, the six models are:
a) Logistic Regression trained with 150 patients,
b) Logistic Regression trained with 373 patients,
c) SVM trained with 150 patients,
d) SVM trained with 373 patients,
e) Random Forest trained with 150 patients, and
f) Random Forest trained with 373 patients.

### Acces
Click [here](https://oasis-dataset-alzheimer-classification-final.streamlit.app/) to access the app

### Dataset
To load the OASIS dataset, click [here](https://sites.wustl.edu/oasisbrains/home/oasis-2/).
If you prefer, you can download it from this repository, but for ethical reasons, it’s better to do it from the official page.

### Model training 
The source code where the data was prepared and the model was developed can be found in the `modelo` directory, in the file named `proyecto_final.py`. This file was used to train the best models; additionally, a pipeline for data processing was implemented.

### Best model
In the folder `archivos`, you will find 7 `.pkl` files corresponding to the models and the pipeline used in the app.

### APP
You will find two files: `requirements.txt` and `streamlit_app.py`. The first contains the libraries required to run the app. The second is the app code (model implementation and interface).
