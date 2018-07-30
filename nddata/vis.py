def vis_detection(im_orig, detections, class_names, thresh=0.7):
    """visualize [cls, conf, x1, y1, x2, y2]"""
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(frameon=False, figsize=(10, 10))
    plt.imshow(im_orig)
    colors = [np.random.random(3) for _ in class_names]
    for [cls, conf, x1, y1, x2, y2] in detections:
        cls = int(cls)
        if cls >= 0 and conf > thresh:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor=colors[cls], linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(x1, y1 - 2, '{:s} {:.3f}'.format(class_names[cls], conf),
                           bbox=dict(facecolor=colors[cls], alpha=0.5), fontsize=12, color='white')


def vis_detection_mono(im_orig, boxes, scores, labels, class_names, thresh=0.7, thresh_hi=None):
    """visualize detection in different color"""
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(frameon=False, figsize=(10, 10))
    ax = plt.axes()
    ax.imshow(im_orig)
    colors = [np.random.random(3) for _ in class_names]

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i]
        score = scores.flat[i]
        cls = int(labels.flat[i])
        if thresh and score < thresh:
            continue
        if thresh_hi and score > thresh_hi:
            continue

        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                             fill=False, edgecolor=colors[cls], linewidth=2, alpha=1)
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 2, '{:s} {:.2f}'.format(class_names[cls], score),
                fontsize=15, family='serif', color='white',
                bbox=dict(facecolor=colors[cls], alpha=0.4, pad=0, edgecolor='none'))


def vis_detection_mask(im_orig, boxes, scores, labels, masks, class_names, thresh=0.7, thresh_hi=None):
    """visualize mask in different color, mask is same shape as im_orig"""
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    plt.figure(frameon=False, figsize=(10, 10))
    ax = plt.axes()
    ax.imshow(im_orig)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i]
        score = scores.flat[i]
        cls = int(labels.flat[i])
        mask = masks[i]
        if thresh and score < thresh:
            continue
        if thresh_hi and score > thresh_hi:
            continue

        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                             fill=False, edgecolor='g', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 2, '{:s} {:.2f}'.format(class_names[cls], score),
                fontsize=15, family='serif', color='white',
                bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'))
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            polygon = plt.Polygon(c.reshape((-1, 2)), fill=True, facecolor=np.random.random(3),
                                  alpha=0.5, edgecolor='w', linewidth=1.2)
            ax.add_patch(polygon)
