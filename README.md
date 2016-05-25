# Fast R-CNN in MXNet

Fast R-CNN treats general object detection as a classification problem and
bounding box prediction as a regression problem. Classifying cropped region
feature maps and predicting bounding box displacements together yields
detection results. Cropping feature maps instead of image input accelerates
computation utilizing shared convolution maps. Bounding box displacements
are simultaneously learned in the training process.

This repository may reflect experimental changes. Refer to `mxnet/example/rcnn/` as a tested example.

## Getting Started

* Install the lastest version of MXNet from DMLC. Follow the instructions at http://mxnet.readthedocs.io/en/latest/how_to/build.html. Install the python interface.
* Download data and place them to `data` folder according to `Data Folder Structure`.
  You might want to create a symbolic link to VOCdevkit folder
```
Pascal VOCdevkit
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
Ross's precomputed object proposals
https://github.com/rbgirshick/fast-rcnn/blob/master/data/scripts/fetch_selective_search_data.sh
Demo data (put in `data/demo` folder)
https://github.com/rbgirshick/fast-rcnn/tree/master/data/demo
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

## Training
* Start training by run `python train.py`. Variable args can be found by run
`python train.py --help`.
* Training can be done in cpu, modify `train.py` accordingly.
```
usage: train.py [-h] [--image_set IMAGE_SET] [--year YEAR]
                [--root_path ROOT_PATH] [--devkit_path DEVKIT_PATH]
                [--pretrained PRETRAINED] [--epoch EPOCH] [--prefix PREFIX]
                [--gpu GPU_ID] [--begin_epoch BEGIN_EPOCH]
                [--end_epoch END_EPOCH] [--frequent FREQUENT]

Train a Fast R-CNN network

optional arguments:
  -h, --help            show this help message and exit
  --image_set IMAGE_SET
                        can be trainval or train
  --year YEAR           can be 2007, 2010, 2012
  --root_path ROOT_PATH
                        output data folder
  --devkit_path DEVKIT_PATH
                        VOCdevkit path
  --pretrained PRETRAINED
                        pretrained model prefix
  --epoch EPOCH         epoch of pretrained model
  --prefix PREFIX       new model prefix
  --gpu GPU_ID          GPU device to train with
  --begin_epoch BEGIN_EPOCH
                        begin epoch of training
  --end_epoch END_EPOCH
                        end epoch of training
  --frequent FREQUENT   frequency of logging
```

## Testing
* Start testing by run `python test.py`. Variable args can be found by run
`python test.py --help`.
* Testing can be done in cpu, modify `test.py` accordingly.
```
usage: test.py [-h] [--image_set IMAGE_SET] [--year YEAR]
               [--root_path ROOT_PATH] [--devkit_path DEVKIT_PATH]
               [--prefix PREFIX] [--epoch EPOCH] [--gpu GPU_ID]

Test a Fast R-CNN network

optional arguments:
  -h, --help            show this help message and exit
  --image_set IMAGE_SET
                        can be test
  --year YEAR           can be 2007, 2010, 2012
  --root_path ROOT_PATH
                        output data folder
  --devkit_path DEVKIT_PATH
                        VOCdevkit path
  --prefix PREFIX       new model prefix
  --epoch EPOCH         epoch of pretrained model
  --gpu GPU_ID          GPU device to test with
```

## Demonstration
* Run demo by `demo.py --gpu 0 --prefix path-to-model --epoch 0`, in which
`path-to-model + '%4d' % epoch.params` will be the params file and
`path-to-model + '-symbol.json'` will be the symbol json.
* If no training has been done, download reference model from Ross Girshick and use
`mxnet/caffe/caffe_converter` to convert it to MXNet. A script in `tools` is also 
provided for your convienience. This script need to be modified in your environment.
```
https://github.com/rbgirshick/fast-rcnn/blob/master/data/scripts/fetch_fast_rcnn_models.sh
```
* Demo can be run in cpu, modify `demo.py` accordingly.
```
usage: demo.py [-h] [--prefix PREFIX] [--epoch EPOCH] [--gpu GPU_ID]

Demonstrate a Fast R-CNN network

optional arguments:
  -h, --help       show this help message and exit
  --prefix PREFIX  new model prefix
  --epoch EPOCH    epoch of pretrained model
  --gpu GPU_ID     GPU device to test with
```

## Disclaimer
This repository used code from [MXNet](https://github.com/dmlc/mxnet),
[Fast R-CNN](https://github.com/rbgirshick/fast-rcnn),
[Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn),
[caffe](https://github.com/BVLC/caffe). Training data are from
[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/),
[ImageNet](http://image-net.org/). Model comes from
[VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

## References
1. Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang. MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems. In Neural Information Processing Systems, Workshop on Machine Learning Systems, 2015
2. Ross Girshick. "Fast R-CNN." In Proceedings of the IEEE International Conference on Computer Vision, 2015.
3. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards real-time object detection with region proposal networks." In Advances in Neural Information Processing Systems, 2015.
4. Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. "Caffe: Convolutional architecture for fast feature embedding." In Proceedings of the ACM International Conference on Multimedia, 2014.
5. Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. "The pascal visual object classes (voc) challenge." International journal of computer vision 88, no. 2 (2010): 303-338.
6. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "ImageNet: A large-scale hierarchical image database." In Computer Vision and Pattern Recognition, IEEE Conference on, 2009.
7. Karen Simonyan, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

