import mxnet as mx


def quota_sampler(ious, num_sample, num_input, pos_ratio, pos_thresh):
    max_pos = int(round(num_sample * pos_ratio))

    # ious (N, M)
    # return: indices (num_samples,) row index to ious
    # return: samples (num_samples,) value 1: pos, 0: neg
    # return: matches (num_samples,) value [0, M)
    ious_max = ious.max(axis=-1)
    ious_argmax = ious.argmax(axis=-1)
    # init with 0, which are neg samples
    mask = mx.sym.zeros_like(ious_max)
    # positive samples
    pos_mask = ious_max >= pos_thresh
    mask = mx.sym.where(pos_mask, mx.sym.ones_like(mask), mask)

    # shuffle mask
    rand = mx.sym.random.uniform(0, 1, shape=(num_input,))
    index = mx.sym.argsort(rand)
    mask = mx.sym.take(mask, index)
    ious_argmax = mx.sym.take(ious_argmax, index)

    # sample pos and neg samples
    order = mx.sym.argsort(mask, is_ascend=False)
    topk = mx.sym.slice_axis(order, axis=0, begin=0, end=max_pos)
    bottomk = mx.sym.slice_axis(order, axis=0, begin=-(num_sample - max_pos), end=None)
    selected = mx.sym.concat(topk, bottomk, dim=0)

    # output
    indices = mx.sym.take(index, selected)
    samples = mx.sym.take(mask, selected)
    matches = mx.sym.take(ious_argmax, selected)
    return indices, samples, matches


def cls_encoder(samples, matches, refs, num_sample):
    # samples (B, N, M) (+1, -1, 0: ignore), matches (B, N) [0, M), refs (B, M)
    # reshape refs (B, M) -> (B, 1, M) -> (B, N, M)
    refs = mx.sym.repeat(refs.reshape((0, 1, -1)), axis=1, repeats=num_sample)
    # ids (B, N, M) -> (B, M), note no + 1 here (processed in data pipeline)
    target_ids = mx.sym.pick(refs, matches, axis=2)
    # samples 1/0, mask out neg samples to 0
    targets = mx.sym.where(samples > 0.5, target_ids, mx.sym.zeros_like(target_ids))
    return targets


def corner_to_center(x):
    xmin, ymin, xmax, ymax = x.split(axis=-1, num_outputs=4)
    width = xmax - xmin
    height = ymax - ymin
    x = xmin + width / 2
    y = ymin + height / 2
    return x, y, width, height


def box_encoder(samples, matches, anchors, labels, refs,
                num_class, num_sample, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
    # refs [B, M, 4], anchors [B, N, 4], samples [B, N], matches [B, N]
    # refs [B, M, 4] -> reshape [B, 1, M, 4] -> repeat [B, N, M, 4]
    ref_boxes = mx.sym.repeat(refs.reshape((0, 1, -1, 4)), axis=1, repeats=num_sample)
    # refs [B, N, M, 4] -> 4 * [B, N, M]
    ref_boxes = mx.sym.split(ref_boxes, axis=-1, num_outputs=4, squeeze_axis=True)
    # refs 4 * [B, N, M] -> pick from matches [B, N, 1] -> concat to [B, N, 4]
    ref_boxes = mx.sym.concat(*[mx.sym.pick(ref_boxes[i], matches, axis=2).reshape((0, -1, 1)) \
                         for i in range(4)], dim=2)
    # labels [B, M] -> [B, N, M]
    ref_labels = mx.sym.repeat(labels.reshape((0, 1, -1)), axis=1, repeats=num_sample)
    # labels [B, N, M] -> pick from matches [B, N] -> [B, N, 1]
    ref_labels = mx.sym.pick(ref_labels, matches, axis=2).reshape((0, -1, 1))
    # transform based on x, y, w, h
    # [B, N, 4] -> 4 * [B, N] -> transform -> codecs [B, N, 4]
    g = corner_to_center(ref_boxes)
    a = corner_to_center(anchors)
    t0 = ((g[0] - a[0]) / a[2] - means[0]) / stds[0]
    t1 = ((g[1] - a[1]) / a[3] - means[1]) / stds[1]
    t2 = (mx.sym.log(g[2] / a[2]) - means[2]) / stds[2]
    t3 = (mx.sym.log(g[3] / a[3]) - means[3]) / stds[3]
    codecs = mx.sym.concat(t0, t1, t2, t3, dim=2)
    # samples [B, N] -> [B, N, 1] -> [B, N, 4] -> get +1 class
    # only the positive samples have targets
    # note that iou with padded 0 box is always 0, thus no targets
    temp = mx.sym.tile(samples.reshape((0, -1, 1)), reps=(1, 1, 4)) > 0.5
    targets = mx.sym.where(temp, codecs, mx.sym.zeros_like(codecs))
    masks = mx.sym.where(temp, mx.sym.ones_like(temp), mx.sym.zeros_like(temp))
    # expand class agnostic targets to per class targets
    out_targets = []
    out_masks = []
    for cid in range(num_class):
        # boolean array [B, N, 1]
        same_cid = (ref_labels == cid)
        # keep orig targets
        out_targets.append(targets)
        # but mask out the one not belong to this class [B, N, 1] -> [B, N, 4]
        out_masks.append(masks * same_cid.repeat(axis=-1, repeats=4))
    # targets, masks C * [B, N, 4] -> [C, B, N, 4] -> [B, N, C, 4]
    all_targets = mx.sym.stack(*out_targets, axis=0).transpose((1, 2, 0, 3))
    all_masks = mx.sym.stack(*out_masks, axis=0).transpose((1, 2, 0, 3))
    return all_targets, all_masks


def rcnn_target_sampler(rois, gt_boxes, batch_images, batch_rois, batch_proposals, fg_fraction, fg_overlap):
    # slice into box coordinates
    gt_boxes = mx.sym.slice_axis(gt_boxes, axis=-1, begin=0, end=4)

    # collect results into list
    new_rois = []
    new_samples = []
    new_matches = []
    for i in range(batch_images):
        roi = mx.sym.squeeze(mx.sym.slice_axis(rois, axis=0, begin=i, end=i+1))
        gt_box = mx.sym.squeeze(mx.sym.slice_axis(gt_boxes, axis=0, begin=i, end=i+1), axis=0)
        # concat rpn roi with ground truth
        all_roi = mx.sym.concat(roi, gt_box, dim=0)
        # calculate (N, M) ious between (N, 4) anchors and (M, 4) bbox ground-truths
        # NOTE cannot do batch op, will get (B, N, B, M) ious
        ious = mx.sym.contrib.box_iou(all_roi, gt_box, format='corner')
        # matches (N,) coded to [0, num_classes), padded gt_boxes code to 0
        indices, samples, matches = quota_sampler(ious, num_sample=batch_rois, num_input=batch_proposals, pos_ratio=fg_fraction, pos_thresh=fg_overlap)
        # stack all samples together
        new_rois.append(all_roi.take(indices))
        new_samples.append(samples)
        new_matches.append(matches)
    new_rois = mx.sym.stack(*new_rois, axis=0)
    new_samples = mx.sym.stack(*new_samples, axis=0)
    new_matches = mx.sym.stack(*new_matches, axis=0)
    return new_rois, new_samples, new_matches


def rcnn_target_generator(rois, gt_boxes, samples, matches, batch_rois, num_classes, box_stds):
    # slice into labels and box coordinates
    gt_labels = mx.sym.slice_axis(gt_boxes, axis=-1, begin=4, end=5)
    gt_boxes = mx.sym.slice_axis(gt_boxes, axis=-1, begin=0, end=4)

    cls_target = cls_encoder(samples, matches, gt_labels, num_sample=batch_rois)
    box_target, box_mask = box_encoder(samples, matches, rois, gt_labels, gt_boxes,
                                       num_class=num_classes, num_sample=batch_rois, stds=box_stds,
                                       means=(0., 0., 0., 0.))
    return cls_target, box_target, box_mask
