# Proyecto Integrador: Clasificación de Sonidos Cardiacos y Pulmonares

Este proyecto utiliza técnicas de procesamiento de señales de audio y aprendizaje automático para clasificar sonidos grabados de pulmones y corazón en diversas patologías.

## Características
- Extracción de características mediante **MFCCs** (Mel-Frequency Cepstral Coefficients).
- Clasificación utilizando el algoritmo **Random Forest**.
- Análisis exploratorio de datos (EDA) completo.
- Visualizaciones avanzadas de desempeño del modelo.

## Visualizaciones Añadidas
1.  **Histograma de Duración**: Distribución de la longitud de las muestras.
2.  **Importancia de Características**: Identificación de los coeficientes MFCC más relevantes.
3.  **Matriz de Confusión Normalizada**: Análisis de precisión por clase en porcentaje.
4.  **Curva Precision-Recall**: Evaluación del balance entre precisión y sensibilidad.

## Requisitos
- Python 3.x
- Jupyter Notebook / Lab
- Librerías: `librosa`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

## Uso
1. Clona el repositorio.
2. Asegúrate de tener los archivos `.csv` y las carpetas de audio correspondientes.
3. Abre y ejecuta `Proyecto_Integrador_Audio_Final.ipynb`.
