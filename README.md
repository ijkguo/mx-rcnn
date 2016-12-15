# Faster R-CNN in MXNet with distributed implementation and data parallelization

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

## For MXNet starters
* Suppose `HOME` represents where this file is located. All commands, unless stated otherwise, should be started from `HOME`.
* Make a folder `model` in `HOME`. `model` folder will be used to place model checkpoints along the training process. 
  It is recommended to make `model` as a symbolic link to some place in hard disk.
* `prefix` refers to the first part of a saved model file name and `epoch` refers to a number in this file name.
  In `e2e-0001.params`, `prefix` is `"e2e"` and `epoch` is `1`.
* `begin_epoch` means the start of your training process, which will apply to all saved checkpoints.

## Getting Started
* Install python package `cython`, `cv2`, `easydict`, `matplotlib`, `numpy`.
* Install [MXNet](https://github.com/precedenceguo/mxnet/tree/simple) and [Python interface](http://mxnet.io/get_started/ubuntu_setup.html).
* Run `make` in `HOME`

## Demo
* Try out detection result by running `python demo.py --prefix final --epoch 0 --image myimage.jpg --gpu 0`.
  Suppose you have downloaded pretrained network and place the extracted file `final-0000.params` in this folder and there is an image named `myimage.jpg`.

## Prepare Pascal VOC data
* Download Pascal VOC data and place the `VOCdevkit` folder in `HOME/data`.

  ```
  Pascal VOCdevkit Download Link
  http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
  ```
* You might want to create a symbolic link to VOCdevkit folder by `ln -s /path/to/your/VOCdevkit data/VOCdevkit`.

## Prepare VGG pretrained model
* Download VGG16 pretrained model from [MXNet model gallery](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-vgg.md),
  rename `vgg16-0000.params` to `vgg16-0001.params` and place it in `model` folder.

## Alternate Training
* Start training by running `python train_alternate.py` after VOCdevkit is ready.
  A typical command would be `python train_alternate.py --gpus 0`. This will train the network on the VOC07 trainval.
  More control of training process can be found in the argparse help accessed by `python train_alternate.py -h`.
* Start testing by running `python test.py --prefix model/final --epoch 0` after completing the training process.
  This will test the network on the VOC07 test with the model `final-0000.params` in `HOME/model`.
  Adding a `--vis` will turn on visualization and `-h` will show help as in the training process.
* An experiment yields 69.54 mAP, close to 69.9 as reported.

## End-to-end Training
* End-to-end training is the same as [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), which is an approximate training process.
  It is first transplanted by [tornadomeet/mx-rcnn](https://github.com/tornadomeet/mx-rcnn), which is a fork of this repository.
* Start training by running `python train_end2end.py`. A typical command would be `python train_end2end.py`. This will train the network on VOC07 trainval.
  As usual control of training process can be found in argparse help.
* Start testing by running `python test.py`. This will test the network on the VOC07 test.
* An experiment yields 69.65 mAP.

## Fast R-CNN
* To reproduce Fast R-CNN, `scipy` is used to load selective search proposals.
* Download precomputed selective search data and place them to `data` folder according to `Data Folder Structure`.
* Start training by running `python -m rcnn.tools.train_rcnn --proposal ss` to use the selective search proposal.
* Start testing by running `python -m rcnn.tools.test_rcnn --proposal ss`.
* An experiment yields 66.5 mAP, close to 66.9 as reported.

## Structure
* This repository provides Faster R-CNN as a package named `rcnn`.
    * `rcnn.core`: core routines in Faster R-CNN training and testing.
    * `rcnn.cython`: cython speedup from py-faster-rcnn.
    * `rcnn.dataset`: dataset library. Base class is `rcnn.dataset.imdb.IMDB`.
    * `rcnn.io`: prepare training data.
    * `rcnn.processing`: data and label processing library.
    * `rcnn.tools`: training and testing wrapper.
    * `rcnn.utils`: utilities in training and testing, usually overloads mxnet functions.

## Information
* Download link to trained model
  Baidu Yun: http://pan.baidu.com/s/1boRhGvH (ixiw) or Dropbox: https://www.dropbox.com/s/jrr83q0ai2ckltq/final-0000.params.tar.gz?dl=0
* Download link to Pascal VOC and precomputed selective search proposals
  ```
  selective_search_data (from py-faster-rcnn)
  Download link accessible at https://github.com/rbgirshick/fast-rcnn/blob/master/data/scripts/fetch_selective_search_data.sh
  ```

* Data Folder Structure

  ```
  VOCdevkit
  -- VOC + year (JPEG images and annotations)
  -- results (will be created by evaluation)
  ---- VOC + year
  ------ main
  -------- comp4_det_val_aeroplane.txt
  selective_search_data
  rpn_data (will be created by rpn)
  cache (will be created by imdb)
  ```

## Disclaimer
This repository used code from [MXNet](https://github.com/dmlc/mxnet),
[Fast R-CNN](https://github.com/rbgirshick/fast-rcnn),
[Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn),
[caffe](https://github.com/BVLC/caffe),
[tornadomeet/mx-rcnn](https://github.com/tornadomeet/mx-rcnn).

Training data are from
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/),
[ImageNet](http://image-net.org/).

Model comes from
[VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).
Thanks to tornadomeet for end-to-end experiments and MXNet contributers for helpful discussions.

## References
1. Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang. MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems. In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015
2. Ross Girshick. "Fast R-CNN." In Proceedings of the IEEE International Conference on Computer Vision, 2015.
3. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In Advances in Neural Information Processing Systems, 2015.
4. Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. "Caffe: Convolutional architecture for fast feature embedding." In Proceedings of the ACM International Conference on Multimedia, 2014.
5. Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. "The pascal visual object classes (voc) challenge." International journal of computer vision 88, no. 2 (2010): 303-338.
6. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "ImageNet: A large-scale hierarchical image database." In Computer Vision and Pattern Recognition, IEEE Conference on, 2009.
7. Karen Simonyan, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
