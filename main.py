from funciones import argumentos, load_dataset, data_treatment, extract_features, mlflow_tracking

def main():
    print("Ejecutamos el main")
    args_values = argumentos() 
    df = load_dataset()
    train_data, val_data, test_data = data_treatment(df)
    x_train_cv, x_val_cv, x_test_cv = extract_features(train_data, val_data, test_data)
    mlflow_tracking(args_values.nombre_job, x_train_cv, x_val_cv, x_test_cv, train_data['overall'], test_data['overall'], args_values.n_estimators_list)  # Ejecutar el tracking con MLflow

if __name__ == "__main__":
    main()
