from mySwinDetector.MySwin import MySwin
from myClassifier.classifierModel import ClassifierModel
from PIL import Image

from pathlib import Path
import os
import numpy as np

if not "root" in locals():
        current_path = Path(os.getcwd())
        root = current_path.parent.absolute()
os.chdir(root)

class mainModel:

    def __init__(self):
        self.detector = MySwin(threshold=0.8)
        self.classifier = ClassifierModel()

    def predict(self,listOfImgsPath):

        for imgPath in listOfImgsPath:
            intermediateOutput = self.detector.predictOne(imgPath)
            if not np.any(intermediateOutput[0]) : return [[0,0,0,0],0]
            else:
                predicted_class = self.classifier.predict_one_image(Image.fromarray(intermediateOutput[1]))
                return([intermediateOutput[0],predicted_class])


if __name__=='__name':
    mainModel()
    
