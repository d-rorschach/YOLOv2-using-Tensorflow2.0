# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:37:31 2020

@author: lenovo
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from preprocessing1 import *
from model import *
from lossfunc import *
from imp_funcs import *
import os

pic_in=open("train_image.pickle","rb")
train_image=pickle.load(pic_in)
print(len(train_image))#expected to be102750

#########initialization
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

train_image_folder ="C:/Users/lenovo/pyprog/yolov2/jpeg_new/"
train_annot_folder= "C:/Users/lenovo/pyprog/yolov2/ann_new/"



print("*"*30)
print("Input")
for i in range(1):
    timage = train_image[i]

    for key, v in timage.items():
        print("  {}: {}".format(key,v))
    print("*"*30)
    print("Output")
    inputEncoder = ImageReader(IMAGE_H=416,IMAGE_W=416, norm=normalize)
    image, all_objs = inputEncoder.fit(timage)
    print("          {}".format(all_objs))
    plt.imshow(image)
    plt.title("image.shape={}".format(image.shape))
    plt.show()
    
    
generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : 50,
}



train_valid_split = int(0.9*len(train_image))

np.random.shuffle(train_image)
valid_image = train_image[train_valid_split:]
train_image = train_image[:train_valid_split]
train_batch_generator = SimpleBatchGenerator(train_image, generator_config, norm=normalize,shuffle=True)
valid_batch_generator = SimpleBatchGenerator(valid_image, generator_config, norm=normalize,shuffle=False)


[x_batch,b_batch],y_batch = train_batch_generator.__getitem__(idx=3)


print(len(train_image))
print(len(valid_image))
print("x_batch: (BATCH_SIZE, IMAGE_H, IMAGE_W, N channels)           = {}".format(x_batch.shape))
print("y_batch: (BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + N classes) = {}".format(y_batch.shape))
print("b_batch: (BATCH_SIZE, 1, 1, 1, TRUE_BOX_BUFFER, 4)            = {}".format(b_batch.shape))


iframe= 0
check_object_in_grid_anchor_pair(iframe , generator_config , y_batch)

plot_image_with_grid_cell_partition(iframe , generator_config , x_batch)
plot_grid(iframe , generator_config , y_batch)
plt.show()


## true_boxes is the tensor that takes "b_batch"
model, true_boxes = define_YOLOv2(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS, trainable=False)
model.summary()

#####do it for the first time
path_to_weight = "C:/Users/lenovo/pyprog/yolov2/yolov2.weights"
weight_reader = WeightReader(path_to_weight)
print("all_weights.shape = {}".format(weight_reader.all_weights.shape))
nb_conv = 23
model = set_pretrained_weight(model,nb_conv, path_to_weight)

###########for rest of the times
'''model.load_weights("weights_yolo_on_voc2012final.h5")
print(model.get_weights())'''


########loss function
def custom_loss(y_true, y_pred):
    true_boxes=b_batch
    return custom_core_loss(
                     y_true,
                     y_pred,
                     true_boxes,
                     GRID_W,
                     GRID_H,
                     BATCH_SIZE,
                     ANCHORS,
                     LAMBDA_COORD,
                     LAMBDA_CLASS,
                     LAMBDA_NO_OBJECT, 
                     LAMBDA_OBJECT)

##########training
dir_log = "logs/"
try:
    os.makedirs(dir_log)
except:
    print('already there')


early_stop = EarlyStopping(monitor='loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint('weights_yolo_on_voc2012.h5', 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)


optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0) 0.5e-4
model.compile(loss=custom_loss, optimizer=optimizer)

print(len(train_batch_generator))

model.fit_generator(generator        = train_batch_generator, 
                    steps_per_epoch  = len(train_batch_generator), 
                    epochs           = 2, 
                    verbose          = 1,
                    validation_data  = valid_batch_generator,
                    validation_steps = len(valid_batch_generator),
                    callbacks        = [early_stop, checkpoint],
                    max_queue_size   = 3
                   )




imageReader = ImageReader(IMAGE_H,IMAGE_W=IMAGE_W, norm=normalize)
out = imageReader.fit(train_image_folder + "2007_000032.jpg")

print(out.shape)
X_test = np.expand_dims(out,0)
print(X_test.shape)
# handle the hack input
dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
y_pred = model.predict([X_test,dummy_array])
print(y_pred.shape)


plt.imshow(out)

netout         = y_pred[0]
outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
netout_scale   = outputRescaler.fit(netout)

obj_threshold = 0.3
boxes = find_high_class_probability_bbox(netout_scale,obj_threshold)
print("\nobj_threshold={}".format(obj_threshold))
print("In total, YOLO can produce GRID_H * GRID_W * BOX = {} bounding boxes ".format( GRID_H * GRID_W * BOX))
print("I found {} bounding boxes with top class probability > {}".format(len(boxes),obj_threshold))


print("Plot with high object threshold")
ima = draw_boxes(X_test[0],boxes,LABELS,verbose=True)
figsize = (15,15)
plt.figure(figsize=figsize)
plt.imshow(ima); 
plt.title("Plot with high object threshold")
plt.show()


iou_threshold = 0.5
print(len(boxes))
print(obj_threshold)
final_boxes = nonmax_suppression(boxes,iou_threshold=iou_threshold,obj_threshold=obj_threshold)
print("{} final number of boxes".format(len(final_boxes)))
print(type(final_boxes))
                   

ima = draw_boxes(X_test[0],final_boxes,LABELS,verbose=True)
figsize = (15,15)
plt.figure(figsize=figsize)
plt.imshow(ima); 
plt.show()

########testing
#np.random.seed(1)
Nsample   = 20
image_nms = list(np.random.choice(os.listdir(train_image_folder),Nsample))
#file_list=os.listdir(train_image_folder)
#image_nms=[]
#for i in range(Nsample):
#    image_nms.append(file_list[i])
#iou_threshold=0.6
#obj_threshold=0.4
#image_nms=['2007_000027.jpg','crop_2007_000027.jpg','rotate_2007_000027.jpg','hue_2007_000027.jpg','sat_2007_000027.jpg','flip_2007_000027.jpg']
print(image_nms)
print(iou_threshold)
print(obj_threshold)

outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
imageReader    = ImageReader(IMAGE_H,IMAGE_W=IMAGE_W, norm=lambda image : image / 255.)
X_test         = []
fpath=[]
for img_nm in image_nms:
    _path    = os.path.join(train_image_folder,img_nm)
    fpath.append(_path)
    out      = imageReader.fit(_path)
    X_test.append(out)

X_test = np.array(X_test)

## model
dummy_array    = np.zeros((len(X_test),1,1,1,TRUE_BOX_BUFFER,4))
y_pred         = model.predict([X_test,dummy_array])

for iframe in range(len(y_pred)):
    ground_box_list=[]
    for i in range(len(train_image)):
        if train_image[i]['filename']==fpath[iframe]:
            inputEncoder = ImageReader(IMAGE_H=416,IMAGE_W=416, norm=normalize)
            image, all_objs = inputEncoder.fit(train_image[i])
            print('index in train_image list = '+str(i))
            for j in range(len(all_objs)):
                classes=np.zeros(shape=len(LABELS),dtype=float)
                for l in range(len(LABELS)):
                    if LABELS[l]==all_objs[j]['name']:
                        classes[l]=1.0
                        break
                ground_box = BoundBox(all_objs[j]['xmin'], all_objs[j]['ymin'], all_objs[j]['xmax'], all_objs[j]['ymax'], 1, classes)
                ground_box_list.append(ground_box)
            break
        
    true_ima = draw_boxes(X_test[iframe],ground_box_list,LABELS,verbose=True,extra=True)
    figsize = (15,15)
    plt.figure(figsize=figsize)
    plt.imshow(true_ima)
    plt.title("true image")
    plt.show()
    ground_box_list.clear()
        
        
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
fpath.clear()