import argparse
import subprocess
import time
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import spacy
import en_core_web_sm

def argumentos():
    parser = argparse.ArgumentParser(description="Modelo con MLflow")
    parser.add_argument('--nombre_job', type=str, help='Nombre del trabajo en MLflow')
    parser.add_argument('--n_estimators_list', nargs='+', type=int, help='Lista de n_estimators')
    return parser.parse_args()

def load_dataset():
    # Cargamos el dataset desde el archivo JSON
    df = pd.read_json(r'C:\Users\Jonay\Desktop\ENTREGA\Despliegue Algoritmos\Movies_and_TV_5.json', lines=True)
    return df

def data_treatment(df):
    # Filtramos reseñas positivas y negativas
    positive_reviews = df[df['overall'] > 3]
    negative_reviews = df[df['overall'] <= 3]

    # Tomamos 5000 muestras aleatorias de cada grupo
    positive_sample = positive_reviews.sample(n=5000, random_state=42)
    negative_sample = negative_reviews.sample(n=5000, random_state=42)
    df_balanced = pd.concat([positive_sample, negative_sample]).sample(frac=1, random_state=42)

    # Separamos en train, val y test
    train_data, temp_data = train_test_split(df_balanced, test_size=0.3, stratify=df_balanced['overall'], random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['overall'], random_state=42)

    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data

def extract_features(train_data, val_data, test_data):
    cv = TfidfVectorizer(
        max_df=0.9,
        min_df=4,
        max_features=2000,
        strip_accents='ascii',
        ngram_range=(1, 1)
    )
    
    x_train_cv = cv.fit_transform(train_data['reviewText'])
    x_val_cv = cv.transform(val_data['reviewText'])
    x_test_cv = cv.transform(test_data['reviewText'])
    
    return x_train_cv, x_val_cv, x_test_cv

def mlflow_tracking(nombre_job, x_train_cv, x_val_cv, x_test_cv, y_train, y_test, n_estimators):
    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui', '--port', '5000'])
    print(mlflow_ui_process)
    time.sleep(5)
    mlflow.set_experiment(nombre_job)

    for i in n_estimators:
        with mlflow.start_run() as run:
            # Crear y entrenar el modelo de regresión logística
            model = LogisticRegression(
                max_iter=1000, 
                solver='lbfgs', 
                penalty='l2', 
                C=1.0, 
                class_weight=None
            )
            model.fit(x_train_cv, y_train)
            y_test_pred = model.predict(x_test_cv)
            
            accuracy_test = accuracy_score(y_test, y_test_pred)
            mlflow.log_metric('accuracy_test', accuracy_test)
            mlflow.log_param('n_estimators', i)
            mlflow.log_param('model_type', 'LogisticRegression')
            mlflow.sklearn.log_model(model, 'clf-model')
            print(f"Modelo con {i} estimadores - Accuracy en test: {accuracy_test}")
            print("\nReporte de clasificación:\n", classification_report(y_test, y_test_pred))
    print("Se ha acabado el entrenamiento del modelo correctamente")


