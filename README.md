# Faster R-CNN in MXNet

![example detections](https://cloud.githubusercontent.com/assets/13162287/22101032/92085dc0-de6c-11e6-9228-67e72606ddbc.png)

Region Proposal Network solves object detection as a regression problem 
from the objectness perspective. Bounding boxes are predicted by applying 
learned bounding box deltas to base boxes, namely anchor boxes across 
different positions in feature maps. Training process directly learns a 
mapping from raw image intensities to bounding box transformation targets.

Fast R-CNN treats general object detection as a classification problem and
bounding box prediction as a regression problem. Classifying cropped region
feature maps and predicting bounding box displacements together yields
detection results. Cropping feature maps instead of image input accelerates
computation utilizing shared convolution maps. Bounding box displacements
are simultaneously learned in the training process.

Faster R-CNN utilize an alternate optimization training process between RPN 
and Fast R-CNN. Fast R-CNN weights are used to initiate RPN for training.
The approximate joint training scheme does not backpropagate rcnn training
error to rpn training.

## Experiments

| Indicator | py-faster-rcnn (caffe resp.) | mx-rcnn (this reproduction) |
| :-------- | :--------------------------- | :-------------------------- |
| Training speed [1] | 2.5 img/s training, 5 img/s testing | 3.8 img/s in training, 12.5 img/s testing |
| Valset performance [2] | mAP 73.2               | mAP 75.97                   |
| Memory usage [3]  | 11G for Fast R-CNN     | 4.6G for Fast R-CNN         |
| Parallelization  [4] | None              | 3.8 img/s to 6 img/s for 2 GPUs |
| Extensibility [5] | Old framework and base networks | ResNet           |

[1] On Ubuntu 14.04.5 with device Titan X, cuDNN enabled.
    The experiment is VGG-16 end-to-end training.  
[2] VGG network. Trained end-to-end on VOC07trainval+12trainval, tested on VOC07 test.  
[3] VGG network. Fast R-CNN is the most memory expensive process.  
[4] VGG network (parallelization limited by bandwidth).
    ResNet-101 speeds up from 2 img/s to 3.5 img/s.  
[5] py-faster-rcnn does not support ResNet or recent caffe version.

| Method | Network | Training Data | Testing Data | Reference | Result | Link  |
| :----- | :------ | :------------ | :----------- | :-------: | :----: | :---: |
| Fast R-CNN | VGG16 | VOC07 | VOC07test | 66.9 | 66.50 | [Dropbox](https://www.dropbox.com/s/xmxjitv0kl96h7v/vgg_fast_rcnn-0008.params?dl=0) |
| Faster R-CNN alternate | VGG16 | VOC07 | VOC07test | 69.9 | 69.62 | [Dropbox](https://www.dropbox.com/s/fgj71uzxz8h6ajj/vgg_voc_alter-0008.params?dl=0) |
| Faster R-CNN end-to-end | VGG16 | VOC07 | VOC07test | 69.9 | 70.23 | [Dropbox](https://www.dropbox.com/s/gfxnf1qzzc0lzw2/vgg_voc07-0010.params?dl=0) |
| Faster R-CNN end-to-end | VGG16 | VOC07+12 | VOC07test | 73.2 | 75.97 | [Dropbox](https://www.dropbox.com/s/rvktx65s48cuyb9/vgg_voc0712-0010.params?dl=0) |
| Faster R-CNN end-to-end | ResNet-101 | VOC07+12 | VOC07test | 76.4 | 79.35 | [Dropbox](https://www.dropbox.com/s/ge2wl0tn47xezdf/resnet_voc0712-0010.params?dl=0) |
| Faster R-CNN end-to-end | VGG16 | COCO train | COCO val | 21.2 | 22.8 | [Dropbox](https://www.dropbox.com/s/e0ivvrc4pku3vj7/vgg_coco-0010.params?dl=0) |
| Faster R-CNN end-to-end | ResNet-101 | COCO train | COCO val | 27.2 | 26.1 | [Dropbox](https://www.dropbox.com/s/bfuy2uo1q1nwqjr/resnet_coco-0010.params?dl=0) |

The above experiments were conducted on [this version of repository](https://github.com/precedenceguo/mx-rcnn/tree/6a1ab0eec5035a10a1efb5fc8c9d6c54e101b4d0)
with [a modified MXNet based on 0.9.1 nnvm pre-release](https://github.com/precedenceguo/mxnet/tree/simple).

## Set up environment
* Install Python package `pip install numpy matplotlib cython easydict`.
* Install MXNet `pip install mxnet-cu90`. Type `import mxnet` in Python console to confirm.

## Run prediction with existing model
* Download a example model trained on VOC07 trainval from
  [Baidu Yun](http://pan.baidu.com/s/1boRhGvH) (ixiw) or [Dropbox](https://www.dropbox.com/s/jrr83q0ai2ckltq/final-0000.params.tar.gz?dl=0) and put the model `final-0000.params` in `HOME`.
* Run predition on `myimage.jpg` with `python demo.py --prefix final --epoch 0 --image myimage.jpg --gpu 0 --vis`.
  Drop the `--vis` if you do not have a display or want to save as a new file.
  `--prefix final --epoch 0` tells the program to load model `final-0000.params`.

## Train new model
The following tutorial is based on VOC data, VGG network.
Before proceed, look at sidenotes on preparing data and pretrained models.
Command line arguments `--network resnet` and `--dataset coco` are used for other networks and datasets.
Scripts like `script/vgg_voc07.sh` are examples to run experiments with different settings.

### Scheme 1: alternate between RPN and Fast RPN
`bash script/vgg_alter_voc07.sh 0` starts alternate training with gpu 0.
* Start training by running `python train_alternate.py`. This will train the VGG network on the VOC07 trainval.
  More control of training process can be found in the argparse help.
* Start testing by running `python test.py --prefix model/final --epoch 0` after completing the training process.
  This will test the VGG network on the VOC07 test with the model in `HOME/model/final-0000.params`.
  Adding a `--vis` will turn on visualization and `-h` will show help as in the training process.

### Scheme 2: end-to-end but no backpropagation from Fast R-CNN to RPN
`bash script/vgg_voc07.sh 0` starts end-to-end training with gpu 0.
* Start training by running `python train_end2end.py`. This will train the VGG network on VOC07 trainval.
* Start testing by running `python test.py`. This will test the VGG network on the VOC07 test.

### Scheme 3: Fast R-CNN with existing object proposals
`bash script/get_selective.sh` downloads precomputed object proposals. `bash script/vgg_fast_rcnn.sh 0` starts training Fast R-CNN with aformentioned proposals with gpu 0.
* To reproduce Fast R-CNN, `scipy` is used to load selective search proposals, thus an additional dependency.
* Download [precomputed selective search data](https://github.com/rbgirshick/fast-rcnn/tree/master/data) and place them to `data` folder.
* Start training by running `python -m rcnn.tools.train_rcnn --proposal selective_search` to use the selective search proposal.
* Start testing by running `python -m rcnn.tools.test_rcnn --proposal selective_search`.
* `script/vgg_fast_rcnn.sh` will train Fast R-CNN on VOC07 and test on VOC07test.

## Sidenotes

### Download data and label
`bash script/get_voc.sh` and `bash script/get_coco.sh` downloads data from official repos and organizes them.
* Make a folder `data` in `HOME`. `data` folder will be used to place the training data folder `VOCdevkit` and `coco`.
* Download and extract [Pascal VOC data](http://host.robots.ox.ac.uk/pascal/VOC/), place the `VOCdevkit` folder in `HOME/data`.
* Download and extract [coco dataset](http://mscoco.org/dataset/), place all images to `coco/images` and annotation jsons to `data/annotations`.

### Download ImageNet models
`bash script/get_pretrained_model.sh` downloads pretrained ImageNet models and organizes them.
* Make a folder `model` in `HOME`. `model` folder will be used to place model checkpoints along the training process. 
  It is recommended to set `model` as a symbolic link to somewhere else in hard disk.
* Download VGG16 pretrained model `vgg16-0000.params` from [MXNet model gallery](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-vgg.md) to `model` folder.
* Download ResNet pretrained model `resnet-101-0000.params` from [ResNet](https://github.com/tornadomeet/ResNet) to `model` folder.

### Dataset attributes
All dataset have three attributes, `image_set`, `root_path` and `dataset_path`.  
* `image_set` could be `2007_trainval` or something like `2007trainval+2012trainval`.  
* `root_path` is usually `data`, where `cache`, `selective_search_data`, `rpn_data` will be stored.  
* `dataset_path` could be something like `data/VOCdevkit`, where images, annotations and results can be put so that many copies of datasets can be linked to the same actual place.    

### Format of command line arguments
Command line arguments follow the format of mxnet/example/image-classification.
* `prefix` refers to the first part of a saved model file name and `epoch` refers to a number in this file name.
  In `model/vgg-0000.params`, `prefix` is `"model/vgg"` and `epoch` is `0`.
* `begin_epoch` means the start of your training process, which will apply to all saved checkpoints.
* Remember to turn off cudnn auto tune. `export MXNET_CUDNN_AUTOTUNE_DEFAULT=0`.

### Code Structure
This repository provides Faster R-CNN as a package named `rcnn`.
  * `rcnn.core`: core routines in Faster R-CNN training and testing.
  * `rcnn.cython`: cython speedup from py-faster-rcnn.
  * `rcnn.dataset`: dataset library. Base class is `rcnn.dataset.imdb.IMDB`.
  * `rcnn.io`: prepare training data.
  * `rcnn.processing`: data and label processing library.
  * `rcnn.pycocotools`: python api from coco dataset.
  * `rcnn.symbol`: symbol and operator.
  * `rcnn.tools`: training and testing wrapper.
  * `rcnn.utils`: utilities in training and testing, usually overloads mxnet functions.

### History
* Fast R-CNN (v1)
* Faster R-CNN (v2)
* Faster R-CNN with module training (v3)
* Faster R-CNN with end-to-end training (v3.5, tornadomeet/mx-rcnn)
* Faster R-CNN with end-to-end training and module testing (v4)
* Faster R-CNN with accelerated training and resnet (v5, current)  

### Disclaimer
This repository used code from [MXNet](https://github.com/dmlc/mxnet),
[Fast R-CNN](https://github.com/rbgirshick/fast-rcnn),
[Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn),
[caffe](https://github.com/BVLC/caffe),
[tornadomeet/mx-rcnn](https://github.com/tornadomeet/mx-rcnn),
[MS COCO API](https://github.com/pdollar/coco).  
Training data are from
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/),
[ImageNet](http://image-net.org/),
[COCO](http://mscoco.org/).  
Model comes from
[VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/),
[ResNet](https://github.com/tornadomeet/ResNet).  
Thanks to tornadomeet for end-to-end experiments and MXNet contributers for helpful discussions.

### References
1. Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang. MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems. In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015
2. Ross Girshick. "Fast R-CNN." In Proceedings of the IEEE International Conference on Computer Vision, 2015.
3. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In IEEE Transactions on Pattern Analysis and Machine Intelligence, 2016.
4. Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. "Caffe: Convolutional architecture for fast feature embedding." In Proceedings of the ACM International Conference on Multimedia, 2014.
5. Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. "The pascal visual object classes (voc) challenge." International journal of computer vision 88, no. 2 (2010): 303-338.
6. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "ImageNet: A large-scale hierarchical image database." In Computer Vision and Pattern Recognition, IEEE Conference on, 2009.
7. Karen Simonyan, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
8. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition". In Computer Vision and Pattern Recognition, IEEE Conference on, 2016.
9. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. "Microsoft COCO: Common Objects in Context" In European Conference on Computer Vision, pp. 740-755. Springer International Publishing, 2014.
