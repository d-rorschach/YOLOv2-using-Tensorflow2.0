# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:30:24 2020

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

######initialization
test_annot_folder="C:/Users/lenovo/pyprog/yolov2/ann_new/"
test_image_folder="C:/Users/lenovo/pyprog/yolov2/jpeg_new/"


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

NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 3
#WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50

###########parsing
test_image, seen_test_labels = parse_annotation(test_annot_folder,test_image_folder, labels=LABELS)
print("N test = {}".format(len(test_image)))
y_pos = np.arange(len(seen_test_labels))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.barh(y_pos,list(seen_test_labels.values()))
ax.set_yticks(y_pos)
ax.set_yticklabels(list(seen_test_labels.keys()))
ax.set_title("The total number of objects = {} in {} images".format(
    np.sum(list(seen_test_labels.values())),len(test_image)
))
plt.show()

############moddel and weight
model, true_boxes = define_YOLOv2(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS, trainable=False)
model.load_weights("C:/Users/lenovo/pyprog/yolov2/weights_yolo_on_voc2012final.h5")

#####mAP
n=len(test_image)
total_object=0
for i in range(n):
    total_object=total_object+len(test_image[i]['object'])
print('total no of object=' + str(total_object))

df=pd.DataFrame({'image_name':[] , 'detection':[] , 'confidence':[] , 'TP':[] , 'FP':[],})

def adjust_minmax(c,_max):
        if c < 0:
            c = 0   
        if c > _max:
            c = _max
        return c
def changexy(xmin,ymin,xmax,ymax):
    image_h, image_w = IMAGE_H,IMAGE_W
    
    xmin = adjust_minmax(int(xmin*image_w),image_w)
    ymin = adjust_minmax(int(ymin*image_h),image_h)
    xmax = adjust_minmax(int(xmax*image_w),image_w)
    ymax = adjust_minmax(int(ymax*image_h),image_h)
    return xmin,ymin,xmax,ymax


goodAnchorBoxFinder    = BestAnchorBoxFinder([])
#bbox_iou = bestAnchorBoxFinder.bbox_iou(new_boxes[i], new_boxes[j])



for i in range(n):
    if(i%300==0):
        print(i)
    _path = test_image[i]['filename']
    #print(_path)
    ground_box_list=[]
    outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
    imageReader    = ImageReader(IMAGE_H,IMAGE_W=IMAGE_W, norm=lambda image : image / 255.)
    
    out      = imageReader.fit(_path)

    X_test = np.expand_dims(out,0)
    #predict model
    dummy_array    = np.zeros((len(X_test),1,1,1,TRUE_BOX_BUFFER,4))
    y_pred         = model.predict([X_test,dummy_array])
    
    #for iframe in range(len(y_pred)):
    netout         = y_pred[0] 
    netout_scale   = outputRescaler.fit(netout)
    boxes          = find_high_class_probability_bbox(netout_scale,obj_threshold)
    final_boxes    = nonmax_suppression(boxes,iou_threshold=iou_threshold,obj_threshold=obj_threshold)
    if len(final_boxes) > 0:
        #print('some object detected of length '+str(len(final_boxes)))
        inputEncoder = ImageReader(IMAGE_H=416,IMAGE_W=416, norm=normalize)
        image, all_objs = inputEncoder.fit(test_image[i])
        
        #print('no of real obj= {}'.format(len(ground_box_list)))
        for p in range(len(final_boxes)):
            max_iou=0.0
            max_ind=0
            for t in range(len(all_objs)):
                if (LABELS[final_boxes[p].label]==all_objs[t]['name']):
                    xmin_pred,ymin_pred,xmax_pred,ymax_pred = changexy(final_boxes[p].xmin,final_boxes[p].ymin,final_boxes[p].xmax,final_boxes[p].ymax)
                    #print('rescaled value of pred {} {} {} {}'.format(pred_xmin,pred_ymin,pred_xmax,pred_ymax))
                    ########finding iou
                    # determining the minimum and maximum -coordinates of the intersection rectangle
                    xmin_inter = max(all_objs[t]['xmin'], xmin_pred)
                    ymin_inter = max(all_objs[t]['ymin'], ymin_pred)
                    xmax_inter = min(all_objs[t]['xmax'], xmax_pred)
                    ymax_inter = min(all_objs[t]['ymax'], ymax_pred)

                    # calculate area of intersection rectangle
                    inter_area = max(0, xmax_inter - xmin_inter ) * max(0, ymax_inter - ymin_inter )

                    # calculate area of actual and predicted boxes
                    actual_area = (all_objs[t]['xmax'] - all_objs[t]['xmin'] ) * (all_objs[t]['ymax'] - all_objs[t]['ymin'] )
                    pred_area = (xmax_pred - xmin_pred ) * (ymax_pred - ymin_pred)

                    # computing intersection over union
                    bbox_iou = inter_area / float(actual_area + pred_area - inter_area)

                    # return the intersection over union value
                    if(bbox_iou > max_iou):
                        max_iou=bbox_iou
                        max_ind=t
            #print('lets find max iou of the prediction {}'.format(max_iou))
            if(max_iou > 0.4 ):
                all_objs.remove(all_objs[max_ind])
                df=df.append({'image_name': int(i) ,'detection':LABELS[int(final_boxes[p].label)],
                           'confidence':final_boxes[p].get_score(),'TP':1,'FP':0}, ignore_index=True)
            else:
                df=df.append({'image_name': i ,'detection':LABELS[int(final_boxes[p].label)],
                           'confidence':final_boxes[p].get_score(),'TP':0,'FP':1}, ignore_index=True)
            
    
        #print('model didnt detect any object')
    
    # creating column 'TP/FP' which will store TP for True positive and FP for False positive
    # if IOU is greater than 0.5 then TP else FP
   
    # calculating Precision and recall

df.sort_values(by='confidence',ascending=False,inplace=True)
df.reset_index(inplace=True)

tp_list=[]
fp_list=[]
TP=0
FP=0
precision=[]
recall=[]

for i in range(len(df)):
    TP=TP+df.loc[i]['TP']
    FP=FP+df.loc[i]['FP']
    tp_list.append(TP)
    fp_list.append(FP)
    precision.append(TP/(TP+FP))
    recall.append(TP/total_object)

df['Acc_TP']=tp_list
df['Acc_FP']=fp_list
df['Precision']=precision
df['Recall']=recall

#calculating Interpolated Precision
df['IP'] = df.groupby('Recall')['Precision'].transform('max')




prec_at_rec = []

for recall_level in np.linspace(0.0, 1.0, 11):
    try:
        x = df[df['Recall'] >= recall_level]['Precision']
        prec = max(x)
    except:
        prec = 0.0
    prec_at_rec.append(prec)
avg_prec = np.mean(prec_at_rec)
print('11 point precision is ', prec_at_rec)
print('\nmap is ', avg_prec)