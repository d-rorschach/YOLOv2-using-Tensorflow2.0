# YOLOv2-using-Tensorflow2.0
# real time object detection project using Yolov2 algorithm
Code is written in Python3. The model has been trained on PASCAL VOC 2012 dataset.
preprocessing0.py file contains image augmentation and parsing functions.
main0.py do parsing and augmentation by importing preprocessing0 and save them in .pickle file. Therefore, you need to run main0 code only for once.
preprocessing1.py contains image resizing, anchor box finding, bounding box, batch generation and some other important functions.
model.py contains model and weight file reader.
loss.py contains loss function.
imp_funcs.py contains code for non max suppression, boxing localized part by seaborn etc additional functions.
training.py do the training.
test.py file can do object detection of any image. you just need to add a path of that image.
find_map.py finds accuracy score of your model in terms of mAP value.
You can also train this model in MS COCO dataset. All you need is to change the LABELS list.
