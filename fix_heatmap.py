import json

nb_path = 'OnlineNewsPopularity/Proyecto_Integrador_News.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell 9 (the heatmap cell) and update the sns.heatmap line
cell = nb['cells'][9]
new_source = []
for line in cell['source']:
    if 'sns.heatmap(corr_matrix' in line:
        # Replace the heatmap call to include vmin=-1, vmax=1, center=0
        line = line.replace(
            "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)",
            "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, center=0)"
        )
    new_source.append(line)

cell['source'] = new_source
nb['cells'][9] = cell

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully updated heatmap with vmin=-1, vmax=1, center=0")
