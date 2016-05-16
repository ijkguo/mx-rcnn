# Fast R-CNN in MXNet

Fast R-CNN treats general object detection as a classification problem and
bounding box prediction as a regression problem. Classifying cropped region
feature maps and predicting bounding box displacements together yields
detection results. Cropping feature maps instead of image input accelerates
computation utilizing shared convolution maps. Bounding box displacements
are simultaneously learned in the training process.

## Getting Started

* MXNet with `ROIPooling` and `smooth_l1` operators are required
* Download data and place theme according to `Data Folder Summary`
```
Pascal VOCdevkit
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
Ross's precomputed object proposals
http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz
```
* Data Folder Structure
```
VOCdevkit
-- selective_search_data
-- cache (created by imdb)
---- name + source + roidb.pkl (create by imdb)
---- name (created by detection and evaluation)
------ {detector name}_detections.pkl
------ annotations.pkl
-- results (created by evaluation)
---- VOC + year
------ main
-------- comp4_det_val_aeroplane.txt
-- VOC + year # original VOC data
```
* Download pretrained models
```
ImageNet pretrained model
http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
Ross's reference model (trained on PASCAL VOC07 trainval)
http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/fast_rcnn_models.tgz
```
* Use Caffe Converter to convert these models. Use `symbol.get_symbol_vgg_test()`
for model construction.
