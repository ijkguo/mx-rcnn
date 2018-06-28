class ResNet50:
    def __init__(self, is_train):
        from .net_resnet import FRCNNResNet, get_feat_size
        self._is_train = is_train
        self._net_cls = FRCNNResNet
        self._as_fn = get_feat_size

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
        if self._is_train:
            net = self._net_cls(
                anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                rpn_min_size=args.rpn_min_size,
                num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=args.rcnn_fg_fraction,
                rcnn_fg_overlap=args.rcnn_fg_overlap, rcnn_bbox_stds=args.rcnn_bbox_stds,
                rcnn_roi_mode='align')
        else:
            net = self._net_cls(
                anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                rpn_min_size=args.rpn_min_size,
                num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                rcnn_batch_rois=args.rcnn_batch_rois,
                rcnn_roi_mode='align')
        return net

    def get_as_fn(self):
        return self._as_fn


NETWORKS = {
    'resnet50': ResNet50
}


def get_network_train(network, args):
    if network not in NETWORKS:
        raise ValueError("network {} not supported".format(network))
    nt_cls = NETWORKS[network](is_train=True)
    nt_cls.set_args(args)
    net = nt_cls.get_net(args)
    as_fn = nt_cls.get_as_fn()
    return net, as_fn


def get_network_test(network, args):
    if network not in NETWORKS:
        raise ValueError("network {} not supported".format(network))
    nt_cls = NETWORKS[network](is_train=False)
    nt_cls.set_args(args)
    net = nt_cls.get_net(args)
    return net
