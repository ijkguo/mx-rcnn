from mxnet.gluon import nn
from ndnet.net_rcnn import FRCNN
from ndnet.net_resnet_v1a import ResNetV1a
from ndnet.net_resnet_v2a import ResNetV2a

__all__ = ['get_net',
           'resnet50_v1a_voc',
           'resnet50_v1a_coco',
           'resnet50_v2a_voc',
           'resnet50_v2a_coco']


def get_net(network, pretrained_base, args):
    networks = {'resnet50_v1a_voc': resnet50_v1a_voc,
                'resnet50_v1a_coco': resnet50_v1a_coco,
                'resnet50_v2a_voc': resnet50_v2a_voc,
                'resnet50_v2a_coco': resnet50_v2a_coco}
    try:
        net = networks[network](pretrained_base, args)
    except KeyError:
        raise NotImplementedError('Network {} not implemented'.format(network))
    return net


def resnet50_v1a_voc(pretrained_base, args):
    backbone = ResNetV1a(layers=(3, 4, 6, 3), prefix='')
    if pretrained_base:
        backbone.load_parameters(args.pretrained, allow_missing=True, ignore_extra=True)
    features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(backbone, layer))
    top_features = nn.HybridSequential()
    for layer in ['layer4']:
        top_features.add(getattr(backbone, layer))
    return FRCNN(
        features, top_features, batch_images=args.batch_images, train_patterns='|'.join(['.*rpn', '.*dense', '.*stage(2|3|4)_conv']),
        img_short=600, img_max_size=1000, img_means=(122.7717, 115.9465, 102.9801), img_stds=(1.0, 1.0, 1.0), clip=None,
        rpn_feature_stride=16, rpn_anchor_scales=(8, 16, 32), rpn_anchor_ratios=(0.5, 1, 2),
        rpn_channels=1024, rpn_nms_thresh=0.7, rpn_min_size=16,
        rpn_train_pre_topk=12000, rpn_train_post_topk=2000, rpn_test_pre_topk=6000, rpn_test_post_topk=300,
        rcnn_feature_stride=16, rcnn_pooled_size=(14, 14), rcnn_roi_mode='align',
        rcnn_num_classes=21, rcnn_batch_rois=128, rcnn_bbox_stds=(0.1, 0.1, 0.2, 0.2),
        rcnn_fg_fraction=0.25, rcnn_fg_overlap=0.5, rcnn_nms_thresh=0.3, rcnn_nms_topk=-1)


def resnet50_v1a_coco(pretrained_base, args):
    backbone = ResNetV1a(layers=(3, 4, 6, 3), prefix='')
    if pretrained_base:
        backbone.load_parameters(args.pretrained, allow_missing=True, ignore_extra=True)
    features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(backbone, layer))
    top_features = nn.HybridSequential()
    for layer in ['layer4']:
        top_features.add(getattr(backbone, layer))
    return FRCNN(
        features, top_features, batch_images=args.batch_images, train_patterns='|'.join(['.*rpn', '.*dense', '.*stage(2|3|4)_conv']),
        img_short=800, img_max_size=1333, img_means=(122.7717, 115.9465, 102.9801), img_stds=(1.0, 1.0, 1.0), clip=4.42,
        rpn_feature_stride=16, rpn_anchor_scales=(2, 4, 8, 16, 32), rpn_anchor_ratios=(0.5, 1, 2),
        rpn_channels=1024, rpn_nms_thresh=0.7, rpn_min_size=16,
        rpn_train_pre_topk=12000, rpn_train_post_topk=2000, rpn_test_pre_topk=6000, rpn_test_post_topk=1000,
        rcnn_feature_stride=16, rcnn_pooled_size=(14, 14), rcnn_roi_mode='align',
        rcnn_num_classes=81, rcnn_batch_rois=512, rcnn_bbox_stds=(0.1, 0.1, 0.2, 0.2),
        rcnn_fg_fraction=0.25, rcnn_fg_overlap=0.5, rcnn_nms_thresh=0.5, rcnn_nms_topk=-1)


def resnet50_v2a_voc(pretrained_base, args):
    backbone = ResNetV2a(layers=(3, 4, 6, 3), prefix='')
    if pretrained_base:
        backbone.load_parameters(args.pretrained, allow_missing=True, ignore_extra=True)
    features = nn.HybridSequential()
    for layer in ['layer0', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(backbone, layer))
    top_features = nn.HybridSequential()
    for layer in ['layer4']:
        top_features.add(getattr(backbone, layer))
    return FRCNN(
        features, top_features, batch_images=args.batch_images, train_patterns='|'.join(['.*rpn', '.*dense', '.*stage(2|3|4)_conv']),
        img_short=600, img_max_size=1000, img_means=(0.0, 0.0, 0.0), img_stds=(1.0, 1.0, 1.0), clip=None,
        rpn_feature_stride=16, rpn_anchor_scales=(8, 16, 32), rpn_anchor_ratios=(0.5, 1, 2),
        rpn_channels=1024, rpn_nms_thresh=0.7, rpn_min_size=16,
        rpn_train_pre_topk=12000, rpn_train_post_topk=2000, rpn_test_pre_topk=6000, rpn_test_post_topk=300,
        rcnn_feature_stride=16, rcnn_pooled_size=(14, 14), rcnn_roi_mode='align',
        rcnn_num_classes=21, rcnn_batch_rois=128, rcnn_bbox_stds=(0.1, 0.1, 0.2, 0.2),
        rcnn_fg_fraction=0.25, rcnn_fg_overlap=0.5, rcnn_nms_thresh=0.3, rcnn_nms_topk=-1)


def resnet50_v2a_coco(pretrained_base, args):
    backbone = ResNetV2a(layers=(3, 4, 6, 3), prefix='')
    if pretrained_base:
        backbone.load_parameters(args.pretrained, allow_missing=True, ignore_extra=True)
    features = nn.HybridSequential()
    for layer in ['layer0', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(backbone, layer))
    top_features = nn.HybridSequential()
    for layer in ['layer4']:
        top_features.add(getattr(backbone, layer))
    return FRCNN(
        features, top_features, batch_images=args.batch_images, train_patterns='|'.join(['.*rpn', '.*dense', '.*stage(2|3|4)_conv']),
        img_short=800, img_max_size=1333, img_means=(0.0, 0.0, 0.0), img_stds=(1.0, 1.0, 1.0), clip=4.42,
        rpn_feature_stride=16, rpn_anchor_scales=(2, 4, 8, 16, 32), rpn_anchor_ratios=(0.5, 1, 2),
        rpn_channels=1024, rpn_nms_thresh=0.7, rpn_min_size=16,
        rpn_train_pre_topk=12000, rpn_train_post_topk=2000, rpn_test_pre_topk=6000, rpn_test_post_topk=1000,
        rcnn_feature_stride=16, rcnn_pooled_size=(14, 14), rcnn_roi_mode='align',
        rcnn_num_classes=81, rcnn_batch_rois=512, rcnn_bbox_stds=(0.1, 0.1, 0.2, 0.2),
        rcnn_fg_fraction=0.25, rcnn_fg_overlap=0.5, rcnn_nms_thresh=0.5, rcnn_nms_topk=-1)
