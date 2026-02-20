import json
import os
import sys

print("Starting Notebook Model Interpretation Update...")

notebook_path = os.path.join("sleep quality", "Proyecto_Integrador_Sleep.ipynb")

if not os.path.exists(notebook_path):
    print(f"Error: Notebook not found at {notebook_path}")
    sys.exit(1)

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print("Notebook loaded.")

    # --- 3. Add Interpretation for Model Results (Section 6.1) ---
    
    # We look for the cell containing the SCATTER PLOT code we added earlier
    target_code_snippet = "SCATTER PLOT: REAL VS PREDICHO"
    scatter_cell_index = -1
    
    for i, cell in enumerate(nb['cells']):
        source = "".join(cell.get('source', []))
        if cell['cell_type'] == 'code' and target_code_snippet in source:
            scatter_cell_index = i
            break
            
    if scatter_cell_index != -1:
        # Check if interpretation already exists
        next_cell = nb['cells'][scatter_cell_index + 1] if scatter_cell_index + 1 < len(nb['cells']) else None
        if next_cell and next_cell['cell_type'] == 'markdown' and "Interpretación de los Resultados Gráficos" in "".join(next_cell['source']):
             print("Model interpretation already exists.")
        else:
            interpretation_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Interpretación de los Resultados Gráficos\n",
                    "\n",
                    "1.  **Gráfica Real vs Predicho**:\n",
                    "    *   **Ideal**: Todos los puntos deberían caer sobre la línea roja punteada (diagonal).\n",
                    "    *   **Observación**: Si los puntos están muy dispersos alrededor de la línea, el modelo tiene un error considerable. Si siguen la tendencia, el modelo ha capturado bien el patrón general.\n",
                    "\n",
                    "2.  **Gráfica de Residuos**:\n",
                    "    *   **Ideal**: Los puntos (residuos) deben estar distribuidos aleatoriamente alrededor de la línea roja horizontal (0), sin formar patrones claros (como una \"U\" o un embudo).\n",
                    "    *   **Observación**: Si vemos una distribución aleatoria, asumimos que el modelo es válido (**Homocedasticidad**). Si hay patrones, podría indicar que faltan variables o que la relación no es totalmente lineal.\n",
                    "\n",
                    "---"
                ]
            }
            nb['cells'].insert(scatter_cell_index + 1, interpretation_cell)
            print("Inserted Scatter/Residuals interpretation.")

    # We look for the cell containing FEATURE IMPORTANCE code
    feature_code_snippet = "FEATURE IMPORTANCE"
    feature_cell_index = -1
    
    # Re-scan indices as they might have shifted
    for i, cell in enumerate(nb['cells']):
        source = "".join(cell.get('source', []))
        if cell['cell_type'] == 'code' and feature_code_snippet in source:
            feature_cell_index = i
            break
            
    if feature_cell_index != -1:
         # Check if interpretation already exists
        next_cell = nb['cells'][feature_cell_index + 1] if feature_cell_index + 1 < len(nb['cells']) else None
        if next_cell and next_cell['cell_type'] == 'markdown' and "Interpretación de Coeficientes" in "".join(next_cell['source']):
             print("Coefficients interpretation already exists.")
        else:
            interpretation_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Interpretación de Coeficientes\n",
                    "Esta gráfica nos muestra **qué variables influyen más** en la duración del sueño según el modelo:\n",
                    "*   **Barras Grandes (Positivas)**: Factores que *aumentan* el sueño. (e.g., Si `Physical Activity` es alta y positiva, hacer ejercicio ayuda a dormir más).\n",
                    "*   **Barras Grandes (Negativas)**: Factores que *disminuyen* el sueño. (e.g., Si `Stress Level` es muy negativo, más estrés significa menos sueño).\n",
                    "*   **Cercanas a Cero**: Variables que no aportan mucha información al modelo para predecir el sueño."
                ]
            }
            nb['cells'].insert(feature_cell_index + 1, interpretation_cell)
            print("Inserted Feature Importance interpretation.")

    # Save changes
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("Notebook model interpretations updated successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
