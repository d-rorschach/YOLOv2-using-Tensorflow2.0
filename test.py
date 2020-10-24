# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 18:07:13 2020

@author: lenovo
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
from preprocessing1 import *
from model import *
from imp_funcs import *
import os
import pandas as pd
from preprocessing0 import *

_path=r'C:\Users\lenovo\Pictures\7e610292-59cf-11e8-9d43-4957c6efbcb7.jpg' #insert image path

iou_threshold = 0.5
obj_threshold = 0.3

LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']

IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
TRUE_BOX_BUFFER  = 50

model, true_boxes = define_YOLOv2(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS, trainable=False)
model.load_weights("C:/Users/lenovo/pyprog/yolov2/weights_yolo_on_voc2012final.h5")

figsize = (15,15)
outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
imageReader    = ImageReader(IMAGE_H,IMAGE_W=IMAGE_W, norm=lambda image : image / 255.)

out      = imageReader.fit(_path)

X_test = np.expand_dims(out,0)
## model
dummy_array    = np.zeros((len(X_test),1,1,1,TRUE_BOX_BUFFER,4))
y_pred         = model.predict([X_test,dummy_array])

for iframe in range(len(y_pred)):    
    netout         = y_pred[iframe] 
    netout_scale   = outputRescaler.fit(netout)
    boxes          = find_high_class_probability_bbox(netout_scale,obj_threshold)
    #if len(boxes) > 0:
    final_boxes    = nonmax_suppression(boxes,iou_threshold=iou_threshold,obj_threshold=obj_threshold)
    ima = draw_boxes(X_test[iframe],final_boxes,LABELS,verbose=True)
    plt.figure(figsize=figsize)
    plt.imshow(ima)
    plt.title("predicted image")
    plt.show()