# Importing libraries
from sklearn.intelex import patch_sklearn
patch_sklearn()
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
DATA_PATH = "Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24
)

def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

models = {
    "SVC": SVC(),
}

svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)

final_svm_model = SVC()
final_svm_model.fit(X, y)

test_data = pd.read_csv("Testing.csv").dropna(axis=1)
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])
svm_preds = final_svm_model.predict(test_X)

symptoms = X.columns.values
symptom_index = {}

for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1   
    
    input_data = np.array(input_data).reshape(1, -1)
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    predictions =svm_prediction
    
    return predictions

predictDisease("Itching,Skin Rash,Nodal Skin Eruptions")