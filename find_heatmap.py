import json

nb_path = 'OnlineNewsPopularity/Proyecto_Integrador_News.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][9]
with open('cell9_source.txt', 'w', encoding='utf-8') as f:
    for line in cell['source']:
        f.write(line)
