from mainModel import mainModel
from pathlib import Path
import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

if not "root" in locals():
        current_path = Path(os.getcwd())
        root = current_path.parent.absolute()
os.chdir(root.joinpath('home'))

model = mainModel()

dataPath = 'jovyan/activities_data/hi__paris_2022_hackathon/final_challenge/datasets/datasets_test/test'
model2carbPath = 'jovyan/activities_data/hi__paris_2022_hackathon/final_challenge/datasets/car_models_footprint.csv'
model2Carb = pd.read_csv(model2carbPath, sep=';')

imgsPath = [el for el in os.listdir(dataPath) if el.endswith('.jpg')]
predictions = {'im_name':[], 'x_min':[], 'y_min':[], 'x_max':[], 'y_max':[], 'e':[]}
#imgsPath = imgsPath[:100]
for imgPath in tqdm(imgsPath): 
    output = model.predict([dataPath+'/'+imgPath])
    predictions['im_name'].append(imgPath)
    predictions['x_min'].append(output[0][0])
    predictions['y_min'].append(output[0][1])
    predictions['x_max'].append(output[0][2])
    predictions['y_max'].append(output[0][3])
    if  (output[1] in model2Carb['models'].to_list()):
        predictions['e'].append(model2Carb[model2Carb['models']==output[1]]['Average of CO2 (g per km)'].values[0])
    else : predictions['e'].append(0)

results = pd.DataFrame(predictions).to_csv('jovyan/my_work/C02Work_predictions_team12.csv')
