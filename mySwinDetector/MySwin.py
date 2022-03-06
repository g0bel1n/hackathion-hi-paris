from mySwinDetector import swin
import numpy as np
from tqdm import tqdm

from pathlib import Path
import os
import numpy as np

if not "root" in locals():
        current_path = Path(os.getcwd())
        root = current_path.parent.absolute()
os.chdir(root.joinpath('home/jovyan/my_work'))
from mmdetection.mmdet.apis import inference_detector, show_result_pyplot
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

def surface(coords):
    return (coords[2]-coords[0])*(coords[3]-coords[1])

class MySwin:

    def __init__(self, threshold=0.8):
        self.model = swin.load_swin()
        self.threshold = threshold

    def predictOne(self,imgPath: str)->list:

        result = inference_detector(self.model,imgPath)
        carsResults = result[0][2]
        trucksResults = result[0][7]

        try : 
            bestCarResultIndex = np.argmax([surface(carResult[0:4]) if len(carResult)>0 else 0 for carResult in carsResults])
            carBestScore = carsResults[bestCarResultIndex][-1]
        except:
            carBestScore = 0
        
        try : 
            bestTruckResultIndex = np.argmax([surface(trucksResult[0:4])  if len(trucksResult)>0 else 0 for trucksResult in trucksResults])
            truckBestScore = trucksResults[bestTruckResultIndex][-1]
        except:
            truckBestScore = 0

        if (not carBestScore) and (not truckBestScore):
            output = [0,0,0,0,0]
        elif (not truckBestScore) or carBestScore> truckBestScore:
            output = carsResults[bestCarResultIndex]
        elif (not carBestScore) or truckBestScore> carBestScore: 
            output = trucksResults[bestTruckResultIndex]
        

        #return [output[0:4], plt.imread(imgPath)[int(output[1]):int(output[3]),int(output[0]):int(output[2])]] if ((output[4]>=self.threshold) or (output[2]-output[0]>output[3]-output[1])) else ([0,0,0,0],0)
        return [output[0:4], plt.imread(imgPath)[int(output[1]):int(output[3]),int(output[0]):int(output[2])]] if ((output[4]>=self.threshold) and (output[2]-output[0]>output[3]-output[1])) else ([0,0,0,0],0)


    def predictBatch(self, dirPath:str)->np.ndarray:
        
        imgsPath = os.listdir(dirPath)
        predictions = list()
        for img in imgsPath:
            predictions.append(self.predictOne(img))
        return np.array(predictions)

if __name__=='__main__':
    MySwin()
    

