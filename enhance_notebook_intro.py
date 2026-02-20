import json
import os
import sys

print("Starting Notebook Introduction and EDA Update...")

notebook_path = os.path.join("sleep quality", "Proyecto_Integrador_Sleep.ipynb")

if not os.path.exists(notebook_path):
    print(f"Error: Notebook not found at {notebook_path}")
    sys.exit(1)

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print("Notebook loaded.")

    # --- 1. Modify Introduction (Section 1) ---
    intro_cell_index = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and '## 1. Carga y Exploración de Datos' in "".join(cell['source']):
            intro_cell_index = i
            break
            
    if intro_cell_index != -1:
        # Create detailed introduction content
        detailed_intro = [
            "# Proyecto Integrador: Análisis de Calidad de Sueño\n",
            "\n",
            "## 1. Introducción\n",
            "\n",
            "### 1.1 Origen de los Datos\n",
            "El conjunto de datos utilizado es el **Sleep Health and Lifestyle Dataset** obtenido de Kaggle. Este dataset comprende 400 filas y 13 columnas, cubriendo una amplia gama de variables relacionadas con el sueño y los hábitos diarios.\n",
            "\n",
            "### 1.2 Descripción del Dataset\n",
            "El dataset incluye detalles sobre hábitos de sueño, estilo de vida y salud cardiovascular. \n",
            "- **Total de Registros:** ~400 filas.\n",
            "- **Variables Predictoras (Inputs):**\n",
            "    1. `Gender`: Género de la persona (Male/Female).\n",
            "    2. `Age`: Edad de la persona en años.\n",
            "    3. `Occupation`: Ocupación o profesión.\n",
            "    4. `Physical Activity Level`: Minutos de actividad física diaria.\n",
            "    5. `Stress Level`: Nivel subjetivo de estrés (escala 1-10).\n",
            "    6. `BMI Category`: Categoría de Índice de Masa Corporal (e.g., Normal, Overweight).\n",
            "    7. `Blood Pressure`: Presión arterial (sistólica/diastólica).\n",
            "    8. `Heart Rate`: Frecuencia cardíaca en reposo (bpm).\n",
            "    9. `Daily Steps`: Número de pasos diarios.\n",
            "    10. `Sleep Disorder`: Trastorno del sueño diagnosticado (None, Insomnia, Sleep Apnea).\n",
            "- **Variable Objetivo (Target):**\n",
            "    1. `Sleep Duration`: Duración del sueño en horas por día.\n",
            "\n",
            "### 1.3 Definición del Problema\n",
            "El objetivo es construir un modelo de regresión que pueda estimar la **duración del sueño** de una persona en función de sus características biológicas y de estilo de vida. Entender estos factores es crucial para identificar hábitos que promuevan una mejor higiene del sueño.\n",
            "\n",
            "### 1.4 Justificación de la Técnica\n",
            "Dado que `Sleep Duration` es una variable **numérica continua**, la **Regresión Lineal** es una técnica adecuada para modelar la relación entre los predictores y el objetivo, permitiéndonos cuantificar el impacto de cada variable (e.g., cuánto disminuye el sueño por cada punto extra de estrés).\n",
            "\n",
            "---\n",
            "## 2. Carga y Exploración de Datos\n"
        ]
        
        # Replace the simple header with the detailed introduction
        nb['cells'][intro_cell_index]['source'] = detailed_intro
        print("Updated Section 1: Introduction.")

    # --- 2. Add EDA Interpretations (Section 2) ---
    # Find Distribution plot cell to insert explanation after it
    dist_cell_index = -1
    for i, cell in enumerate(nb['cells']):
        source = "".join(cell.get('source', []))
        if cell['cell_type'] == 'code' and "sns.histplot" in source and "Sleep Duration" in source:
            dist_cell_index = i
            break
    
    if dist_cell_index != -1:
        # Check if explanation already exists to avoid duplication
        next_cell = nb['cells'][dist_cell_index + 1] if dist_cell_index + 1 < len(nb['cells']) else None
        if next_cell and next_cell['cell_type'] == 'markdown' and "Interpretación de la Distribución" in "".join(next_cell['source']):
            print("Distribution explanation already exists.")
        else:
            interpretation_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Interpretación de la Distribución\n",
                    "El histograma muestra cómo se distribuye la duración del sueño en nuestra muestra. \n",
                    "- Si la distribución se asemeja a una **campana de Gauss (Normal)**, los modelos lineales suelen funcionar mejor.\n",
                    "- Observamos si hay **sesgo** (colas hacia la derecha o izquierda) o valores atípicos que puedan distorsionar el modelo."
                ]
            }
            nb['cells'].insert(dist_cell_index + 1, interpretation_cell)
            print("Inserted Distribution interpretation.")

    # Find Correlation Heatmap cell to insert explanation after it
    # Note: Heatmap cell index might have shifted due to previous insertion
    corr_cell_index = -1
    for i, cell in enumerate(nb['cells']):
        source = "".join(cell.get('source', []))
        if cell['cell_type'] == 'code' and "sns.heatmap" in source:
            corr_cell_index = i
            break
            
    if corr_cell_index != -1:
        next_cell = nb['cells'][corr_cell_index + 1] if corr_cell_index + 1 < len(nb['cells']) else None
        if next_cell and next_cell['cell_type'] == 'markdown' and "Interpretación de la Correlación" in "".join(next_cell['source']):
             print("Correlation explanation already exists.")
        else:
            interpretation_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Interpretación de la Correlación\n",
                    "El mapa de calor nos revela las relaciones lineales entre variables numéricas:\n",
                    "- **Correlación Positiva (Rojo/Naranja)**: Cuando una variable aumenta, la otra también (ej. `Quality of Sleep` y `Sleep Duration` suelen estar correlacionados).\n",
                    "- **Correlación Negativa (Azul)**: Cuando una variable aumenta, la otra disminuye (ej. `Stress Level` suele tener correlación negativa con `Sleep Duration`).\n",
                    "- **Multicolinealidad**: Si dos variables predictoras tienen una correlación muy alta (cercana a 1 o -1), podrían confundir al modelo. (ej. `Systolic` y `Diastolic` BP suelen estar muy correlacionadas)."
                ]
            }
            nb['cells'].insert(corr_cell_index + 1, interpretation_cell)
            print("Inserted Correlation interpretation.")

    # Save changes
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("Notebook updated successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
