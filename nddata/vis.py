def vis_detection(im_orig, detections, class_names, thresh=0.7):
    """visualize [cls, conf, x1, y1, x2, y2]"""
    import matplotlib.pyplot as plt
    import random
    plt.imshow(im_orig)
    colors = [(random.random(), random.random(), random.random()) for _ in class_names]
    for [cls, conf, x1, y1, x2, y2] in detections:
        cls = int(cls)
        if cls >= 0 and conf > thresh:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor=colors[cls], linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(x1, y1 - 2, '{:s} {:.3f}'.format(class_names[cls], conf),
                           bbox=dict(facecolor=colors[cls], alpha=0.5), fontsize=12, color='white')
    plt.show()


def vis_detection_mono(im_orig, boxes, scores, labels, class_names, thresh=0.7):
    import random
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False, figsize=(10, 10))
    ax = plt.axes()
    ax.imshow(im_orig)
    colors = [(random.random(), random.random(), random.random()) for _ in class_names]

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    for i in sorted_inds:
        bbox = boxes[i]
        score = scores.flat[i]
        id = int(labels.flat[i])
        if score < thresh:
            continue

        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                             fill=False, edgecolor=colors[id], linewidth=2, alpha=1)
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1] - 2, '{:s} {:2f}'.format(class_names[id], score),
                fontsize=15, family='serif', color='white',
                bbox=dict(facecolor=colors[id], alpha=0.4, pad=0, edgecolor='none'))
    return ax
