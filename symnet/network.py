class VGG16:
    def __init__(self, is_train):
        from .symbol_vgg import get_vgg_train, get_vgg_test
        self._is_train = is_train
        self._sym_fn = get_vgg_train if is_train else get_vgg_test

    def set_args(self, args):
        if not args.pretrained:
            args.pretrained = 'model/vgg16-0000.params'
        args.img_pixel_means = (123.68, 116.779, 103.939)
        args.img_pixel_stds = (1.0, 1.0, 1.0)
        args.net_fixed_params = ['conv1', 'conv2']
        args.rpn_feat_stride = 16
        args.rcnn_feat_stride = 16
        args.rcnn_pooled_size = (7, 7)

    def get_net(self, args):
        if self._is_train:
            return self._sym_fn(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                                rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                                rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                                rpn_min_size=args.rpn_min_size, rpn_batch_rois=args.rpn_batch_rois,
                                num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                                rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                                rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=args.rcnn_fg_fraction,
                                rcnn_fg_overlap=args.rcnn_fg_overlap, rcnn_bbox_stds=args.rcnn_bbox_stds)
        else:
            return self._sym_fn(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                                rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                                rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                                rpn_min_size=args.rpn_min_size,
                                num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                                rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size)


class ResNet50:
    def __init__(self, is_train):
        from .symbol_resnet import get_resnet_train, get_resnet_test
        self._is_train = is_train
        self._sym_fn = get_resnet_train if is_train else get_resnet_test

    def set_args(self, args):
        if not args.pretrained:
            args.pretrained = 'model/resnet-50-0000.params'
        args.img_pixel_means = (0.0, 0.0, 0.0)
        args.img_pixel_stds = (1.0, 1.0, 1.0)
        args.net_train_patterns = '|'.join(['.*rpn', '.*dense', '.*stage(2|3|4)_conv'])
        args.rpn_feat_stride = 16
        args.rcnn_feat_stride = 16
        args.rcnn_pooled_size = (14, 14)

    def get_net(self, args):
        if self._is_train:
            return self._sym_fn(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                                rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                                rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                                rpn_min_size=args.rpn_min_size, rpn_batch_rois=args.rpn_batch_rois,
                                num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                                rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                                rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=args.rcnn_fg_fraction,
                                rcnn_fg_overlap=args.rcnn_fg_overlap, rcnn_bbox_stds=args.rcnn_bbox_stds,
                                units=(3, 4, 6, 3), filter_list=(256, 512, 1024, 2048))
        else:
            return self._sym_fn(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                                rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                                rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                                rpn_min_size=args.rpn_min_size,
                                num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                                rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                                units=(3, 4, 6, 3), filter_list=(256, 512, 1024, 2048))


class ResNet101:
    def __init__(self, is_train):
        from .symbol_resnet import get_resnet_train, get_resnet_test
        self._is_train = is_train
        self._sym_fn = get_resnet_train if is_train else get_resnet_test

    def set_args(self, args):
        if self._is_train and not args.pretrained:
            args.pretrained = 'model/resnet-101-0000.params'
        args.img_pixel_means = (0.0, 0.0, 0.0)
        args.img_pixel_stds = (1.0, 1.0, 1.0)
        args.net_train_patterns = '|'.join(['.*rpn', '.*dense', '.*stage(2|3|4)_conv'])
        args.rpn_feat_stride = 16
        args.rcnn_feat_stride = 16
        args.rcnn_pooled_size = (14, 14)

    def get_net(self, args):
        if self._is_train:
            return self._sym_fn(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                                rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                                rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                                rpn_min_size=args.rpn_min_size, rpn_batch_rois=args.rpn_batch_rois,
                                num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                                rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                                rcnn_batch_rois=args.rcnn_batch_rois, rcnn_fg_fraction=args.rcnn_fg_fraction,
                                rcnn_fg_overlap=args.rcnn_fg_overlap, rcnn_bbox_stds=args.rcnn_bbox_stds,
                                units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048))
        else:
            return self._sym_fn(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                                rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                                rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                                rpn_min_size=args.rpn_min_size,
                                num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                                rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                                units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048))


NETWORKS = {
    'vgg16': VGG16,
    'resnet50': ResNet50,
    'resnet101': ResNet101
}


def get_feat_shape_fn(sym):
    feat_sym = sym.get_internals()['rpn_cls_score_output']
    def _feat_shape(im_height, im_width):
        _, out_shape, _ = feat_sym.infer_shape(data=(1, 3, im_height, im_width))
        return out_shape[0][-2:]
    return _feat_shape


def get_network_train(network, args):
    if network not in NETWORKS:
        raise ValueError("network {} not supported".format(network))
    nt_cls = NETWORKS[network](is_train=True)
    nt_cls.set_args(args)
    sym = nt_cls.get_net(args)
    as_fn = get_feat_shape_fn(sym)
    return sym, as_fn


def get_network_test(network, args):
    if network not in NETWORKS:
        raise ValueError("network {} not supported".format(network))
    nt_cls = NETWORKS[network](is_train=False)
    nt_cls.set_args(args)
    sym = nt_cls.get_net(args)
    return sym
