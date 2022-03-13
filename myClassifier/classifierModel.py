import torch
import torch.nn as nn
import os
from pathlib import Path
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import albumentations 
from albumentations import pytorch as AT
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.io import loadmat
from PIL import Image


if not "root" in locals():
        current_path = Path(os.getcwd())
        root = current_path.parent.absolute()
os.chdir(root)

data_transforms_test = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])




def load_model(model_file):
    """
    Loads a pretrained model and sets the last layer to have 30 outputs
    
    :param model_file: The path to the saved model file
    :return: The model is being returned.
    """
    model_ft = models.resnext101_32x8d(pretrained=True) 
    model_ft.fc = torch.nn.Linear(model_ft.fc.in_features, 30)
    checkpoint = torch.load(model_file)
    model_ft.load_state_dict(checkpoint)
    return model_ft

def prediction_bar(output):
    """
    This function takes in the output of the model and plots the confidence score of the top 5 classes
    
    :param output: The output of the model
    :return: the top 5 predictions and their respective confidence scores.
    """
    
    output = output.detach().numpy()
    pred_labels = output.argsort()[0]
    pred_labels = np.flip(pred_labels[-1*len(pred_labels):])
    
    prediction, label = [], []
    
    for i in pred_labels[:5]:
        prediction.append(float(output[:,i]*100))
        label.append(str(i))
    
    labels = loadmat('jovyan/my_work/myClassifier/cars_meta.mat')

    #Write all the labels as Cars models in separate list
    labels = labels['class_names'][0]
        
    for i in pred_labels[:5]:
        print('Class: {} , confidence: {:.2f}%'.format(labels[int(i)][0],float(output[:,i])*100))
        
    plt.bar(label,prediction, color='green')
    plt.title("Confidence Score Plot")
    plt.xlabel("Confidence Score")
    plt.ylabel("Class number")
    
    return None

# This class is a wrapper for the above functions. 
class ClassifierModel:
    
    def __init__(self):
        self.model=load_model('jovyan/my_work/myClassifier/res152-3.pth')
        self.model.eval()
        labels = loadmat('jovyan/my_work/myClassifier/cars_meta.mat')
        self.keep_id = [0,3,6,7,8,9,10,12,13,15,16,17,20,21,23,24,25,26,27,29,32,34,42,46,47,51,55,58,59,65,67,69,70,72,73,78,79,80,82,83,91,94,95,97,99,102,103,104,105,106,107,108,110,111,113,114,115,116,119,123,125,127,132,133,136,137,139,140,141,143,145,146,147,149,150,151,154,157,160,162,165,166,167,171,172,174,175,176,177,179,180,181,183,185,186,187,191,192,193,194]
        self.labels = labels['class_names'][0]


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict_one_image(self,image, transforms = data_transforms_test):
        """
        Given an image, the function returns the predicted label of the image
        
        :param image: the image to be predicted
        :param transforms: The transforms to be applied to the image
        :return: the predicted label for the image.
        """
        image= image.convert('RGB')

        image_tr= data_transforms_test(image)
        data = image_tr.expand(1,-1,-1,-1)

        probs = nn.Softmax(dim = 1)
        output = self.model(data)
        output = probs(output)
        _, predicted = torch.max(output.data, 1)
        
        output = output.detach().numpy()
        pred_labels = output.argsort()[0]
        pred_labels = np.flip(pred_labels[-1*len(pred_labels):])
        
        return self.labels[self.keep_id[int(pred_labels[0])]][0]



if __name__=='__main__':
    ClassifierModel()
