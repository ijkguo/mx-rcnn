# Fast R-CNN in MXNet

Fast R-CNN treats general object detection as a classification problem and
bounding box prediction as a regression problem. Classifying cropped region
feature maps and predicting bounding box displacements together yields
detection results. Cropping feature maps instead of image input accelerates
computation utilizing shared convolution maps. Bounding box displacements
are simultaneously learned in the training process.

## Getting Started

* MXNet with `ROIPooling` and `smooth_l1` operators are required
* Download data and place them to `data` folder according to `Data Folder Structure`.
  You might want to create a symbolic link to VOCdevkit folder
```
Pascal VOCdevkit
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
Ross's precomputed object proposals
http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz
```
* Data Folder Structure (suppose root is `data`)
```
demo
selective_search_data
cache (created by imdb)
-- name + source + roidb.pkl (create by imdb)
-- name (created by detection and evaluation)
VOCdevkit
-- VOC + year (JPEG images and annotations)
-- results (created by evaluation)
---- VOC + year
------ main
-------- comp4_det_val_aeroplane.txt
```
* Download VGG16 pretrained model, use `mxnet/tools/caffe_converter` to convert it,
  rename to `vgg16-symbol.json` and `vgg16-0001.params` and place it in `model` folder
* Download 'demo' data and put it in `data/demo` from
```
https://github.com/rbgirshick/fast-rcnn/tree/master/data/demo
```
