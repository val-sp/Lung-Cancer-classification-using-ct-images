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
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import Conv2D, AvgPool2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
import os
import warnings
warnings.filterwarnings('ignore')
dataset_path = r"C:\Users\Shubhrojit Panda\Desktop\New folder\The IQ-OTHNCCD lung cancer dataset"
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))
img = cv2.imread(r'C:\Users\Shubhrojit Panda\Desktop\New folder\The IQ-OTHNCCD lung cancer dataset\Bengin cases\Bengin case (10).jpg')
plt.title('Bengin Case')
plt.imshow(img, label = 'Bengin Case')
plt.axis()
plt.show()
img = cv2.imread(r'C:\Users\Shubhrojit Panda\Desktop\New folder\The IQ-OTHNCCD lung cancer dataset\Malignant cases\Malignant case (10).jpg')
plt.title('Mailgnant Case')
plt.imshow(img, label = 'Malignant Case')
plt.axis()
plt.show()
img = cv2.imread(r'C:\Users\Shubhrojit Panda\Desktop\New folder\The IQ-OTHNCCD lung cancer dataset\Normal cases\Normal case (10).jpg')
plt.title('Normal Case')
plt.imshow(img,label = 'Normal Case')
plt.axis()
plt.show()
print(img.shape)
dir = r"C:\Users\Shubhrojit Panda\Desktop\New folder\The IQ-OTHNCCD lung cancer dataset"
img_width= 256
img_height = 256
categories = ['Bengin cases', 'Malignant cases', 'Normal cases']
img_data=[]
for cata in categories:
    folder = os.path.join(dir, cata)
    label = categories.index(cata)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        try:
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (img_height,img_width))
            if img_array is not None and not img_array.size == 0:
                img_data.append([img_array, label])
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            continue
random.shuffle(img_data)
x=[]
y=[]
for features , labels in img_data:
    x.append(features)
    y.append(labels)
X=np.array(x,dtype=float)
Y=np.array(y,dtype=float)
print(X[19])
X = X/255.0
print(X.shape)
x, x_test, y, y_test = train_test_split(X,Y, test_size = 0.2)
x_train, x_val, y_train, y_val = train_test_split(X,Y, test_size=0.2)
model = Sequential()
datagen = ImageDataGenerator(rotation_range=15,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             fill_mode='nearest')
datagen.fit(x_train)
augmented_images_per_original = 6
augmented_x_train=[]
augmented_y_train=[]
for i in range(len(x_train)):
    for _ in range(augmented_images_per_original):
        augmented_image = datagen.flow(np.expand_dims(x_train[i], axis=0), batch_size=1)[0][0]
        augmented_x_train.append(augmented_image)
        augmented_y_train.append(y_train[i])
augmented_x_train=np.array(augmented_x_train)
augmented_y_train=np.array(augmented_y_train)
x_train_augmented = np.concatenate((x_train, augmented_x_train), axis=0)
y_train_augmented = np.concatenate((y_train, augmented_y_train), axis=0)
model.add(Conv2D(128, (3,3), padding = 'same', input_shape = X.shape[1:], activation = 'relu'))
model.add(AvgPool2D(2,2))
model.add(Conv2D(128, (3,3), padding = 'same', activation = 'relu'))
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dropout(0.2, seed = 12))
model.add(Dense(3000, activation = 'relu'))
model.add(Dense(1500, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.save("lung_cancer_detection_model.h5")