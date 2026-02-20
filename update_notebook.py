import json
import os
import sys

print("Starting Notebook Update...")

notebook_path = os.path.join("sleep quality", "Proyecto_Integrador_Sleep.ipynb")

if not os.path.exists(notebook_path):
    print(f"Error: Notebook not found at {notebook_path}")
    sys.exit(1)

# New cells to insert
new_cells_content = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 6.1 Análisis Gráfico de Resultados\n",
            "\n",
            "Visualizaremos el desempeño del mejor modelo para entender mejor sus predicciones."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# SCATTER PLOT: REAL VS PREDICHO\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Seleccionamos el mejor modelo (Ridge)\n",
            "best_model_name = 'Ridge'\n",
            "if best_model_name in models:\n",
            "    best_model = models[best_model_name]\n",
            "    y_pred_best = best_model.predict(X_test_scaled)\n",
            "\n",
            "    plt.figure(figsize=(12, 5))\n",
            "\n",
            "    # 1. Real vs Predicho\n",
            "    plt.subplot(1, 2, 1)\n",
            "    plt.scatter(y_test, y_pred_best, alpha=0.6, color='purple')\n",
            "    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n",
            "    plt.xlabel('Valor Real')\n",
            "    plt.ylabel('Valor Predicho')\n",
            "    plt.title(f'Real vs Predicho ({best_model_name})')\n",
            "\n",
            "    # 2. Residuos\n",
            "    residuals = y_test - y_pred_best\n",
            "    plt.subplot(1, 2, 2)\n",
            "    sns.scatterplot(x=y_pred_best, y=residuals, color='orange', alpha=0.6)\n",
            "    plt.axhline(0, color='red', linestyle='--')\n",
            "    plt.xlabel('Valor Predicho')\n",
            "    plt.ylabel('Residuos')\n",
            "    plt.title('Gráfica de Residuos')\n",
            "\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "else:\n",
            "    print(f\"Modelo {best_model_name} no encontrado en 'models'.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# FEATURE IMPORTANCE\n",
            "if 'best_model' in locals():\n",
            "    feature_names = X.columns\n",
            "    coef_df = pd.DataFrame({\n",
            "        'Feature': feature_names,\n",
            "        'Coefficient': best_model.coef_\n",
            "    })\n",
            "    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()\n",
            "    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)\n",
            "\n",
            "    plt.figure(figsize=(10, 6))\n",
            "    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')\n",
            "    plt.title(f'Importancia de las Características ({best_model_name})')\n",
            "    plt.xlabel('Coeficiente')\n",
            "    plt.show()"
        ]
    }
]

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print("Notebook loaded.")

    # Find index of "Conclusiones"
    insert_index = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source_content = "".join(cell.get('source', []))
            if '## 7. Conclusiones' in source_content:
                insert_index = i
                break

    if insert_index != -1:
        print(f"Inserting {len(new_cells_content)} cells before index {insert_index}")
        # Insert in reverse order to maintain sequence
        for cell in reversed(new_cells_content):
            nb['cells'].insert(insert_index, cell)
    else:
        print("Could not find Conclusiones section, appending to end.")
        nb['cells'].extend(new_cells_content)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("Notebook updated successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
