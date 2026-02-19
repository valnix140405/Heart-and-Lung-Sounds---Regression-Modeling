import json
import os

nb_path = 'OnlineNewsPopularity/Proyecto_Integrador_News.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the index of the heatmap cell
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if 'source' in cell:
        source_text = "".join(cell['source'])
        if "sns.heatmap(corr_matrix" in source_text:
            insert_idx = i + 1
            break

if insert_idx != -1:
    # New Markdown Cell
    new_md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Correlación con la Variable Objetivo (Bar Chart)\n",
            "A continuación, visualizamos explícitamente cómo se correlaciona cada variable numérica con el **Log de Shares**. \n",
            "Esto permite identificar rápidamente qué características tienen mayor impacto positivo o negativo."
        ]
    }

    # New Code Cell
    new_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calcular correlación de todas las variables con log_shares\n",
            "correlations = df_clean.select_dtypes(include=[np.number]).corrwith(df_clean['log_shares']).sort_values(ascending=False)\n",
            "\n",
            "# Excluir la propia variable objetivo y 'shares' original\n",
            "correlations = correlations.drop(['log_shares', 'shares'], errors='ignore')\n",
            "\n",
            "# Seleccionar las 10 más positivas y 10 más negativas\n",
            "top_positive = correlations.head(10)\n",
            "top_negative = correlations.tail(10)\n",
            "top_corr = pd.concat([top_positive, top_negative])\n",
            "\n",
            "plt.figure(figsize=(12, 8))\n",
            "sns.barplot(x=top_corr.values, y=top_corr.index, palette='coolwarm')\n",
            "plt.title('Top 20 Variables con Mayor Correlación con Log(Shares)')\n",
            "plt.xlabel('Coeficiente de Correlación')\n",
            "plt.axvline(x=0, color='black', linestyle='-')\n",
            "plt.grid(axis='x', linestyle='--', alpha=0.7)\n",
            "plt.show()"
        ]
    }

    # Insert cells
    nb['cells'].insert(insert_idx, new_code_cell)
    nb['cells'].insert(insert_idx, new_md_cell)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Successfully inserted cells at index {insert_idx}")
else:
    print("Heatmap cell not found!")
