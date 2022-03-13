from mySwinDetector.MySwin import MySwin
from myClassifier.classifierModel import ClassifierModel
from PIL import Image

from pathlib import Path
import os
import numpy as np

if "root" not in locals():
        current_path = Path(os.getcwd())
        root = current_path.parent.absolute()
os.chdir(root)

class mainModel:

    def __init__(self):
        self.detector = MySwin(threshold=0.8)
        self.classifier = ClassifierModel()

    def predict(self,listOfImgsPath):
        """
        This function takes in a list of image paths and returns a list of bounding boxes and the
        predicted class for each image
        
        :param listOfImgsPath: a list of image paths that you want to predict
        :return: The return is a list of two elements. The first element is a list of the bounding boxes
        of the detected objects. The second element is the predicted class of the image.
        """

        for imgPath in listOfImgsPath:
                intermediateOutput = self.detector.predictOne(imgPath)
                if not np.any(intermediateOutput[0]) and not np.any(
                    intermediateOutput[0]):
                        return [[0,0,0,0],0]
                predicted_class = self.classifier.predict_one_image(Image.fromarray(intermediateOutput[1]))
                return([intermediateOutput[0],predicted_class])


if __name__=='__name':
    mainModel()
    
