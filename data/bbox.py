import mxnet as mx


def bbox_corner2center(x, split=False):
    xmin, ymin, xmax, ymax = x.split(axis=-1, num_outputs=4)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    x = xmin + (width - 1) / 2
    y = ymin + (height - 1) / 2
    if not split:
        return mx.nd.concat(x, y, width, height, dim=-1)
    else:
        return x, y, width, height


def bbox_center2corner(x, split=False):
    x, y, w, h = x.split(axis=-1, num_outputs=4)
    hw = (w - 1) / 2
    hh = (h - 1) / 2
    xmin = x - hw
    ymin = y - hh
    xmax = x + hw
    ymax = y + hh
    if not split:
        return mx.nd.concat(xmin, ymin, xmax, ymax, dim=-1)
    else:
        return xmin, ymin, xmax, ymax


def bbox_decode(x, anchors, stds=(1.0, 1.0, 1.0, 1.0)):
    ax, ay, aw, ah = anchors.split(axis=-1, num_outputs=4)
    dx, dy, dw, dh = x.split(axis=-1, num_outputs=4)
    ox = dx * stds[0] * aw + ax
    oy = dy * stds[1] * ah + ay
    ow = mx.nd.exp(dw * stds[2]) * aw
    oh = mx.nd.exp(dh * stds[3]) * ah
    return mx.nd.concat(ox, oy, ow, oh, dim=-1)


def bbox_clip(x, height, width):
    xmin, ymin, xmax, ymax = x.split(axis=-1, num_outputs=4)
    xmin = xmin.clip(0, width)
    ymin = ymin.clip(0, height)
    xmax = xmax.clip(0, width)
    ymax = ymax.clip(0, height)
    return mx.nd.concat(xmin, ymin, xmax, ymax, dim=-1)


def pick_deltas(cls, deltas):
    delta0 = deltas.pick(4 * cls, axis=1, keepdims=True)
    delta1 = deltas.pick(4 * cls + 1, axis=1, keepdims=True)
    delta2 = deltas.pick(4 * cls + 2, axis=1, keepdims=True)
    delta3 = deltas.pick(4 * cls + 3, axis=1, keepdims=True)
    return mx.nd.concat(delta0, delta1, delta2, delta3, dim=-1)


def decode_detect(rois, scores, bbox_deltas, im_info, nms_thresh):
    """rois (nroi, 4), scores (nrois, nclasses), bbox_deltas (nrois, 4 * nclasses), im_info (3)"""
    # convert to per class detection results
    cls = scores.argmax(axis=1, keepdims=True)
    conf = scores.max(axis=1, keepdims=True)
    box_deltas = pick_deltas(cls, bbox_deltas)

    # decode bbox regression
    boxes = bbox_corner2center(rois)
    boxes = bbox_decode(box_deltas, boxes)
    pred_boxes = bbox_center2corner(boxes)

    # clip to image boundary
    height, width, scale = im_info
    pred_boxes = bbox_clip(pred_boxes, height, width)

    # revert to original scale
    pred_boxes = pred_boxes / scale

    # non maximum suppression
    nms_in = mx.nd.concat(cls, conf, pred_boxes, dim=1)
    nms_out = mx.nd.contrib.box_nms(nms_in, overlap_thresh=nms_thresh)

    # slice into output
    cls = nms_out.slice_axis(axis=-1, begin=0, end=1)
    conf = nms_out.slice_axis(axis=-1, begin=1, end=2)
    boxes = nms_out.slice_axis(axis=-1, begin=2, end=6)

    # remove background class
    cls -= 1
    return cls, conf, boxes
