class ResNet50:
    def __init__(self, is_train):
        from ndnet.net_resnet import FRCNNResNet, get_feat_size
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
                rcnn_batch_rois=args.rcnn_batch_rois, rcnn_bbox_stds=args.rcnn_bbox_stds,
                rcnn_nms_thresh=args.rcnn_nms_thresh, rcnn_nms_topk=args.rcnn_nms_topk,
                rcnn_roi_mode='align')
        return net

    def get_as_fn(self):
        return self._as_fn


class NetworkFactory:
    NETWORKS = {
        'resnet50': ResNet50
    }
    def __init__(self, network):
        if network not in self.NETWORKS:
            raise ValueError("network {} not supported".format(network))
        self._nt_cls = self.NETWORKS[network]

    def get_train(self, args):
        nt = self._nt_cls(is_train=True)
        nt.set_args(args)
        net = nt.get_net(args)
        as_fn = nt.get_as_fn()
        return net, as_fn

    def get_test(self, args):
        nt = self._nt_cls(is_train=False)
        nt.set_args(args)
        net = nt.get_net(args)
        return net
