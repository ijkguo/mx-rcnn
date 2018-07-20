from mxnet.gluon import nn
from ndnet.net_rcnn import RPN, RCNN, FRCNN


class ResNetV1a:
    def __init__(self, is_train):
        from ndnet.net_resnet_v1a import ResNetV1a
        self._is_train = is_train
        self._net_cls = ResNetV1a

    def set_args(self, args):
        if self._is_train:
            args.net_train_patterns = '|'.join(['.*rpn', '.*dense', '.*stage(2|3|4)_conv'])
            args.pretrained = args.pretrained if args.pretrained else 'model/resnet50_v1a_0000.params'
        args.img_pixel_means = (122.7717, 115.9465, 102.9801)
        args.img_pixel_stds = (1.0, 1.0, 1.0)
        args.rpn_feat_stride = 16
        args.rcnn_feat_stride = 16
        args.rcnn_pooled_size = (14, 14)

    def get_net(self, args):
        backbone = self._net_cls(layers=(3, 4, 6, 3), prefix='')
        features = nn.HybridSequential()
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add(getattr(backbone, layer))
        top_features = nn.HybridSequential()
        for layer in ['layer4']:
            top_features.add(getattr(backbone, layer))
        rpn = RPN(1024, len(args.rpn_anchor_scales) * len(args.rpn_anchor_ratios))
        rcnn = RCNN(2048, args.rcnn_num_classes)
        net = FRCNN(features, top_features, rpn, rcnn,
            rpn_feature_stride=args.rpn_feat_stride, rpn_anchor_scales=args.rpn_anchor_scales, rpn_anchor_ratios=args.rpn_anchor_ratios,
            rpn_pre_topk=args.rpn_pre_nms_topk, rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh, rpn_min_size=args.rpn_min_size,
            rcnn_feature_stride=args.rcnn_feat_stride, rcnn_pooled_size=args.rcnn_pooled_size, rcnn_roi_mode='align',
            rcnn_num_classes=args.rcnn_num_classes, rcnn_batch_size=args.rcnn_batch_size, rcnn_batch_rois=args.rcnn_batch_rois, rcnn_bbox_stds=args.rcnn_bbox_stds,
            rcnn_fg_fraction=args.rcnn_fg_fraction, rcnn_fg_overlap=args.rcnn_fg_overlap,
            rcnn_nms_thresh=args.rcnn_nms_thresh, rcnn_nms_topk=args.rcnn_nms_topk)
        return net


class ResNetV2a:
    def __init__(self, is_train):
        from ndnet.net_resnet_v2a import ResNetV2a
        self._is_train = is_train
        self._net_cls = ResNetV2a

    def set_args(self, args):
        if self._is_train:
            args.net_train_patterns = '|'.join(['.*rpn', '.*dense', '.*stage(2|3|4)_conv'])
            args.pretrained = args.pretrained if args.pretrained else 'model/resnet50_0000.params'
        args.img_pixel_means = (0.0, 0.0, 0.0)
        args.img_pixel_stds = (1.0, 1.0, 1.0)
        args.rpn_feat_stride = 16
        args.rcnn_feat_stride = 16
        args.rcnn_pooled_size = (14, 14)

    def get_net(self, args):
        backbone = self._net_cls(layers=(3, 4, 6, 3), prefix='')
        features = nn.HybridSequential()
        for layer in ['layer0', 'layer1', 'layer2', 'layer3']:
            features.add(getattr(backbone, layer))
        top_features = nn.HybridSequential()
        for layer in ['layer4']:
            top_features.add(getattr(backbone, layer))
        rpn = RPN(1024, len(args.rpn_anchor_scales) * len(args.rpn_anchor_ratios))
        rcnn = RCNN(2048, args.rcnn_num_classes)
        net = FRCNN(features, top_features, rpn, rcnn,
            rpn_feature_stride=args.rpn_feat_stride, rpn_anchor_scales=args.rpn_anchor_scales, rpn_anchor_ratios=args.rpn_anchor_ratios,
            rpn_pre_topk=args.rpn_pre_nms_topk, rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh, rpn_min_size=args.rpn_min_size,
            rcnn_feature_stride=args.rcnn_feat_stride, rcnn_pooled_size=args.rcnn_pooled_size, rcnn_roi_mode='align',
            rcnn_num_classes=args.rcnn_num_classes, rcnn_batch_size=args.rcnn_batch_size, rcnn_batch_rois=args.rcnn_batch_rois, rcnn_bbox_stds=args.rcnn_bbox_stds,
            rcnn_fg_fraction=args.rcnn_fg_fraction, rcnn_fg_overlap=args.rcnn_fg_overlap,
            rcnn_nms_thresh=args.rcnn_nms_thresh, rcnn_nms_topk=args.rcnn_nms_topk)
        return net


class NetworkFactory:
    NETWORKS = {
        'resnet50_v1a': ResNetV1a,
        'resnet50_v2a': ResNetV2a
    }
    def __init__(self, network):
        if network not in self.NETWORKS:
            raise ValueError("network {} not supported".format(network))
        self._nt_cls = self.NETWORKS[network]

    def get_train(self, args):
        nt = self._nt_cls(is_train=True)
        nt.set_args(args)
        net = nt.get_net(args)
        return net

    def get_test(self, args):
        nt = self._nt_cls(is_train=False)
        nt.set_args(args)
        net = nt.get_net(args)
        return net
