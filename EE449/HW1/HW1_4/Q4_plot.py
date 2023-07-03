from utils import part4Plots

import json

models = ['mlp_1', 'mlp_2', 'cnn_3', 'cnn_4',
          'cnn_5']  # Models that will be printed
results = list()
for model in models:
    results = []
    f = open('Q4_JSON\Q4_'+model+'.json')

    # returns JSON object as
    # a dictionary
    results.append(json.load(f))
    f.close()
    part4Plots(results, save_dir=r'Q3_IMAGES', filename=model+"_plot")
