# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:28:28 2020

@author: lenovo
"""
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
# imageio library will be used for image input/output
import imageio
import pandas as pd
# this library is needed to read XML files for converting it into CSV
import xml.etree.ElementTree as ET
import re
from preprocessing0 import *

LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']

IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
#OBJ_THRESHOLD    = 0.3
#NMS_THRESHOLD    = 0.65
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 3
#WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50

train_image_folder ="C:/Users/lenovo/pyprog/yolov2/jpeg_new/"   #image folder path
train_annot_folder= "C:/Users/lenovo/pyprog/yolov2/ann_new/"    #annotation folder path

c=0
for ann in sorted(os.listdir(train_image_folder)):
    if "jpg" in ann:
        c=c+1
print(c) #number of image files. expected to be 17125
c=0
for ann in sorted(os.listdir(train_annot_folder)):
    if "xml" in ann:
        c=c+1
print(c) #number of annotation files. expected to be 17125

## xml to csv
xml_df = xml_to_csv(train_annot_folder,train_image_folder, labels=LABELS)
#print('xml dataframe consists all objects details')
#print(xml_df)

## Parse annotations 
train_image, seen_train_labels = parse_annotation(train_annot_folder,train_image_folder, labels=LABELS)
print("N train = {}".format(len(train_image)))
print(train_image[2])
y_pos = np.arange(len(seen_train_labels))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.barh(y_pos,list(seen_train_labels.values()))
ax.set_yticks(y_pos)
ax.set_yticklabels(list(seen_train_labels.keys()))
ax.set_title("The total number of objects = {} in {} images".format(
    np.sum(list(seen_train_labels.values())),len(train_image)
))
plt.show()

# Apply augmentation to our images and save files into 'aug_images/' folder with 'aug1_' prefix.
# 20% cropping
crop_images_df = image_aug(xml_df, train_image_folder,train_image_folder, 'crop_', augmentor=iaa.Crop(percent=(0.20)))
#-60 to +60 degree random ratate
rotate_images_df = image_aug(xml_df, train_image_folder, train_image_folder, 'rotate_', augmentor=iaa.Affine(rotate=(-60, 60)))
#-50 to +50 random hue
hue_images_df = image_aug(xml_df, train_image_folder,train_image_folder, 'hue_', augmentor=iaa.AddToHue((-50,50)))
#-50 to +50 random saturation
sat_images_df = image_aug(xml_df, train_image_folder, train_image_folder, 'sat_', augmentor=iaa.AddToSaturation((-50,50)))
#horizontal flip ie mirror image
flip_images_df = image_aug(xml_df, train_image_folder,train_image_folder, 'flip_', augmentor=iaa.Fliplr(1.0))

crop_images_df.dropna(axis='index',inplace=True)
crop_images_df.reset_index(inplace=True)
crop_images_df.drop(['index'], axis=1,inplace=True)

rotate_images_df.dropna(axis='index',inplace=True)
rotate_images_df.reset_index(inplace=True)
rotate_images_df.drop(['index'], axis=1,inplace=True)

hue_images_df.dropna(axis='index',inplace=True)
hue_images_df.reset_index(inplace=True)
hue_images_df.drop(['index'], axis=1,inplace=True)

sat_images_df.dropna(axis='index',inplace=True)
sat_images_df.reset_index(inplace=True)
sat_images_df.drop(['index'], axis=1,inplace=True)

flip_images_df.dropna(axis='index',inplace=True)
flip_images_df.reset_index(inplace=True)
flip_images_df.drop(['index'], axis=1,inplace=True)

#print("lets print crop_images_df")
#print(crop_images_df)

train_image= add_in_list(folder_name=train_image_folder ,augimg_df=crop_images_df ,grouped= crop_images_df.groupby('filename'),lis=train_image)
train_image= add_in_list(folder_name=train_image_folder ,augimg_df=rotate_images_df ,grouped= rotate_images_df.groupby('filename'),lis=train_image)
train_image= add_in_list(folder_name=train_image_folder ,augimg_df=hue_images_df ,grouped= hue_images_df.groupby('filename') ,lis=train_image)
train_image= add_in_list(folder_name=train_image_folder ,augimg_df=sat_images_df ,grouped= sat_images_df.groupby('filename') ,lis=train_image)
train_image= add_in_list(folder_name=train_image_folder ,augimg_df=flip_images_df ,grouped= flip_images_df.groupby('filename') ,lis=train_image)

print(len(train_image))#expected to be102566
import pickle
pic_out=open("train_image.pickle","wb")
pickle.dump(train_image,pic_out)
pic_out.close()
