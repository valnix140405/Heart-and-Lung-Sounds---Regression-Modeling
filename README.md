# Proyecto Integrador: Clasificación de Sonidos Cardiacos y Pulmonares

Este proyecto utiliza técnicas de procesamiento de señales de audio y aprendizaje automático para clasificar sonidos grabados de pulmones y corazón en diversas patologías.

## Características
- Extracción de características mediante **MFCCs** (Mel-Frequency Cepstral Coefficients).
- Clasificación utilizando el algoritmo **Random Forest**.
- Análisis exploratorio de datos (EDA) completo.
- Visualizaciones avanzadas de desempeño del modelo.

## Requisitos
- Python 3.x
- Jupyter Notebook / Lab
- Librerías: `librosa`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

## Uso

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/valnix140405/Heart-and-Lung-Sounds---Regression-Modeling.git
   ```
2. **Configura los datos**:
   - El repositorio incluye los archivos `.csv` con los metadatos.
   - Debido a su tamaño, las carpetas de audio (`HS/`, `LS/`, `Mix/`) no están en GitHub.
   - **Debes crear estas carpetas** en el directorio raíz y colocar los archivos `.wav` correspondientes dentro de ellas para que el código funcione.
3. **Instala las dependencias**:
   ```bash
   pip install librosa pandas numpy matplotlib seaborn scikit-learn
   ```
4. **Ejecuta el notebook**: Abre `Proyecto_Integrador_Audio_Final.ipynb` en Jupyter.

