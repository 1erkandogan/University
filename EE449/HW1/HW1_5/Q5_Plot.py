from utils import part5Plots

import json

f = open('Q5_JSON\Q5_learning.json')
data1 = json.load(f)
f.close()

part5Plots(data1, save_dir=r'Q5_IMAGES', filename="learning")

f = open('Q5_JSON\Q5_learning2.json')
data2 = json.load(f)
f.close()

part5Plots(data2, save_dir=r'Q5_IMAGES', filename="learning2")

f = open('Q5_JSON\Q5_learning3.json')
data2 = json.load(f)
f.close()

part5Plots(data2, save_dir=r'Q5_IMAGES', filename="learning3")
