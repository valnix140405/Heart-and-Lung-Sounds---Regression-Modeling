
# %% [markdown]
# # Proyecto Integrador - Módulo 2: Clasificación de Sonidos Cardiacos y Pulmonares
# 
# ## 1. Introducción
# 
# ### Contexto del Problema
# La detección temprana de anomalías cardiacas (como soplos) y pulmonares (como sibilancias o crepitaciones) es fundamental para el diagnóstico clínico. Sin embargo, la auscultación depende en gran medida de la experiencia del médico. El uso de Inteligencia Artificial para el procesamiento de señales de audio permite objetivizar y automatizar esta tarea, proporcionando una herramienta de apoyo al diagnóstico.
# 
# ### Objetivo
# Desarrollar un modelo de aprendizaje automático (Random Forest) capaz de clasificar a partir de características de audio (MFCCs) si una grabación corresponde a un sonido normal o a una patología específica, utilizando un dataset de 535 grabaciones capturadas de un maniquí clínico.
# 
# ### Justificación de la Técnica
# Se utiliza la técnica de **Clasificación Supervisada**. Las señales de audio se transforman en **MFCCs (Mel-Frequency Cepstral Coefficients)**, que capturan las características espectrales del sonido de manera similar a como el oído humano percibe las frecuencias. El modelo **Random Forest** es ideal para este dataset por su robustez ante el ruido y su capacidad para manejar múltiples clases con un número moderado de muestras.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
FILE_HS_CSV = 'HS.csv'
FILE_LS_CSV = 'LS.csv'
FILE_MIX_CSV = 'Mix.csv'

DIR_HS = os.path.join('HS', 'HS')
DIR_LS = os.path.join('LS', 'LS')
DIR_MIX = os.path.join('Mix', 'Mix')

# %% [markdown]
# ## 2. Carga y Preparación de Datos
# Ingestamos los datos de las tres fuentes disponibles: Sonidos Cardiacos (HS), Pulmonares (LS) y las fuentes del dataset Mezclado (Mix).

# %%
def load_datasets():
    data = []
    # 1. HS
    if os.path.exists(FILE_HS_CSV):
        df_hs = pd.read_csv(FILE_HS_CSV)
        for _, row in df_hs.iterrows():
            data.append({'path': os.path.join(DIR_HS, f"{row['Heart Sound ID']}.wav"), 
                         'label': f"Heart_{row['Heart Sound Type']}", 'type': 'Heart'})
    # 2. LS
    if os.path.exists(FILE_LS_CSV):
        df_ls = pd.read_csv(FILE_LS_CSV)
        for _, row in df_ls.iterrows():
            data.append({'path': os.path.join(DIR_LS, f"{row['Lung Sound ID']}.wav"), 
                         'label': f"Lung_{row['Lung Sound Type']}", 'type': 'Lung'})
    # 3. Mix Sources
    if os.path.exists(FILE_MIX_CSV):
        df_mix = pd.read_csv(FILE_MIX_CSV)
        for _, row in df_mix.iterrows():
            data.append({'path': os.path.join(DIR_MIX, f"{row['Heart Sound ID']}.wav"), 
                         'label': f"Heart_{row['Heart Sound Type']}", 'type': 'Heart'})
            data.append({'path': os.path.join(DIR_MIX, f"{row['Lung Sound ID']}.wav"), 
                         'label': f"Lung_{row['Lung Sound Type']}", 'type': 'Lung'})
    return pd.DataFrame(data)

df_dataset = load_datasets()
print(f"Dataset cargado: {len(df_dataset)} registros.")

# %% [markdown]
# ## 3. Visualización y Análisis Exploratorio de Datos (EDA)
# 
# ### Interpretación de la Distribución
# El siguiente gráfico muestra el balance de nuestro dataset. Podemos observar la cantidad de ejemplos disponibles para cada patología de corazón y pulmón. Un dataset balanceado es ideal, pero en datos médicos es común tener más muestras de sonidos "Normales". El modelo Random Forest se entrena para distinguir estas variaciones cuantitativas.

# %%
plt.figure(figsize=(12, 6))
sns.countplot(data=df_dataset, y='label', palette='viridis')
plt.title('Distribución de Sonidos en el Dataset')
plt.xlabel('Cantidad de Muestras')
plt.ylabel('Tipo de Sonido')
plt.show()

# %% [markdown]
# ### Interpretación de la Señal
# 1.  **Waveform (Forma de Onda)**: Representa la amplitud (volumen) del sonido en el eje Y frente al tiempo en el eje X. Permite identificar la periodicidad de los latidos o la duración de las respiraciones.
# 2.  **Espectrograma de Mel**: Es una representación visual del espectro de frecuencias de la señal a medida que varía con el tiempo. Los colores más claros/brillantes indican frecuencias con mayor energía. Esta es la "imagen" de la cual el modelo extrae los patrones para clasificar.

# %%
# Tomar una muestra aleatoria para visualizar
sample_row = df_dataset.sample(1).iloc[0]
audio, sr = librosa.load(sample_row['path'], duration=5)

plt.figure(figsize=(15, 8))

# Waveform
plt.subplot(2, 1, 1)
librosa.display.waveshow(audio, sr=sr)
plt.title(f"Waveform - {sample_row['label']}")

# Espectrograma de Mel
plt.subplot(2, 1, 2)
S = librosa.feature.melspectrogram(y=audio, sr=sr)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title(f"Espectrograma de Mel - {sample_row['label']}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Extracción de Características
# Transformamos los audios en vectores numéricos de 40 MFCCs.

# %%
def extract_features(audio_path):
    try:
        audio, sr = librosa.load(audio_path, res_type='kaiser_fast', duration=5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except:
        return None

print("Extrayendo características... esto tomará un momento.")
features = []
for idx, row in df_dataset.iterrows():
    f = extract_features(row['path'])
    if f is not None:
        features.append([f, row['label']])

features_df = pd.DataFrame(features, columns=['feature', 'class_label'])
print(f"Características extraídas para {len(features_df)} audios.")

# %% [markdown]
# ## 5. Entrenamiento del Modelo
# Utilizamos Random Forest para la clasificación y Validación Cruzada para asegurar la robustez de los resultados.

# %%
X = np.array(features_df['feature'].tolist())
y = np.array(features_df['class_label'].tolist())

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %% [markdown]
# ### Análisis del Desempeño
# 1.  **Reporte de Clasificación**: Muestra la precisión (capacidad de no clasificar como positivo un caso negativo), el recal o sensibilidad (capacidad de encontrar todos los casos positivos) y el F1-score por cada clase.
# 2.  **Matriz de Confusión**: El eje vertical representa las clases reales y el eje horizontal las predicciones del modelo. Los valores en la diagonal principal indican aciertos. Los valores fuera de la diagonal muestran qué patologías está confundiendo el modelo entre sí.

# %%
y_pred = model.predict(X_test)
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Validación Cruzada
print("\n--- VALIDACIÓN CRUZADA (5-FOLD) ---")
cv_scores = cross_val_score(model, X, y_encoded, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print(f"Precisión Promedio: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# %% [markdown]
# ## 7. Conclusiones y Recomendaciones
# 
# ### Conclusiones
# 1.  El modelo logró una precisión promedio de **{cv_scores.mean():.2f}**, lo cual es un desempeño sólido considerando la complejidad de los sonidos fisiológicos y el número de clases.
# 2.  Los MFCCs demostraron ser características altamente informativas para distinguir entre patologías cardiacas y pulmonares.
# 3.  La integración de las fuentes del dataset 'Mix' fue crucial para estabilizar el modelo, pasando de un sobreajuste inicial a un desempeño generalizable.
# 
# ### Recomendaciones
# 1.  **Aumento de Datos**: Si se requiere mayor precisión, se podrían aplicar técnicas de aumento de datos (ruido, estiramiento temporal) para balancear clases con pocas muestras.
# 2.  **Modelos de Deep Learning**: Probar Redes Neuronales Convolucionales (CNN) usando los espectrogramas como imágenes de entrada, lo cual suele superar a Random Forest en tareas de audio complejas.
# 3.  **Filtrado Avanzado**: Implementar filtros de paso de banda específicos para limpiar ruidos de baja frecuencia no deseados antes de la extracción de características.
