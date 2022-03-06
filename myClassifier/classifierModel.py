import torch
import torch.nn as nn
import os
from pathlib import Path
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
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
    model_ft = models.resnext101_32x8d(pretrained=True) 
    model_ft.fc = torch.nn.Linear(model_ft.fc.in_features, 30)
    checkpoint = torch.load(model_file)
    model_ft.load_state_dict(checkpoint)
    return model_ft

def prediction_bar(output):
    
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

class ClassifierModel:
    

    def __init__(self):

        self.model=load_model('jovyan/my_work/myClassifier/res152.pth')
        self.model.eval()
        labels = loadmat('jovyan/my_work/myClassifier/cars_meta.mat')
        self.keep_id = [0, 6, 12, 20, 21, 24, 26, 29, 42, 51, 55, 58, 69, 73, 78, 79, 104, 108, 114, 125, 132, 137, 146, 149, 157, 166, 167, 177, 186, 192]


    #Write all the labels as Cars models in separate list
        self.labels = labels['class_names'][0]


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict_one_image(self,image, transforms = data_transforms_test):
        image= image.convert('RGB')

        image_tr= data_transforms_test(image)
        data = image_tr.expand(1,-1,-1,-1)

        probs = nn.Softmax(dim = 1)
        output = self.model(data)
        output = probs(output)
        _, predicted = torch.max(output.data, 1)
        
        #prediction_bar(output)
        output = output.detach().numpy()
        pred_labels = output.argsort()[0]
        pred_labels = np.flip(pred_labels[-1*len(pred_labels):])
        
        return self.labels[self.keep_id[int(pred_labels[0])]][0]



if __name__=='__main__':
    
    ClassifierModel()