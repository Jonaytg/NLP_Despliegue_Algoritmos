# Proyecto de Reseñas - Despliegue de Algoritmos

Este proyecto aplica técnicas de procesamiento de lenguaje natural (NLP) para analizar reseñas de usuarios. El objetivo es desarrollar, evaluar y desplegar modelos de clasificación binaria (positiva/negativa) sobre textos reales.

## Contenido del proyecto

1. **Exploración de Datos**
   - Selección de 10,000 muestras equilibradas entre reseñas positivas y negativas.
   - División del dataset en conjuntos de entrenamiento y prueba.
   - Tokenización y análisis inicial del texto.

2. **Preprocesamiento**
   - Limpieza del texto.
   - Conversión a minúsculas, eliminación de puntuación y stopwords.
   - Vectorización con técnicas como TF-IDF.

3. **Entrenamiento de Modelos**
   - Modelos probados: Naive Bayes, Regresión Logística y Random Forest.
   - Comparación de métricas: accuracy, F1-score, matriz de confusión.

4. **Despliegue**
   - Entrenamiento de un modelo final.
   - Guardado del modelo con `joblib` o `pickle`.
   - Ejemplo de carga del modelo y predicción para producción.
