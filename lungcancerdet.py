import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import PIL as pl
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras import utils  # type: ignore
from tensorflow.keras.layers import Conv2D, AvgPool2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
import os
import warnings
warnings.filterwarnings('ignore')
dataset_path = r"C:\Users\Shubhrojit Panda\Desktop\New folder\LungcancerDataSet"
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))
img = cv2.imread(r'C:\Users\Shubhrojit Panda\Desktop\New folder\LungcancerDataSet\Data\test\BenginCases\Bengin case (101).jpg')
plt.title('Bengin Case')
plt.imshow(img, label = 'Bengin Case')
plt.axis()
plt.show()
img = cv2.imread(r'C:\Users\Shubhrojit Panda\Desktop\New folder\LungcancerDataSet\Data\test\MalignantCases\Malignant case (464).jpg')
plt.title('Mailgnant Case')
plt.imshow(img, label = 'Malignant Case')
plt.axis()
plt.show()
img = cv2.imread(r'C:\Users\Shubhrojit Panda\Desktop\New folder\LungcancerDataSet\Data\test\normal\24.png')
plt.title('Normal Case')
plt.imshow(img,label = 'Normal Case')
plt.axis()
plt.show()
print(img.shape)
dir = r'C:\Users\Shubhrojit Panda\Desktop\New folder\The IQ-OTHNCCD lung cancer dataset'
img_width= 256
img_height = 256
categories = ['Bengin cases, Malignant cases', 'Normal cases']
img_data=[]
for cata in categories:
    folder = os.path.join(dir, cata)
    label = categories.index(cata)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        try:
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (img_height,img_width))