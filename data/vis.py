def vis_detection(im_array, detections, class_names):
    """visualize [cls, conf, x1, y1, x2, y2]"""
    import matplotlib.pyplot as plt
    import random
    plt.imshow(im_array)
    colors = [(random.random(), random.random(), random.random()) for _ in class_names]
    for [cls, conf, x1, y1, x2, y2] in detections:
        if cls > 0:
            cls = int(cls)
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor=colors[cls], linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(x1, y1 - 2, '{:s} {:.3f}'.format(class_names[cls], conf),
                           bbox=dict(facecolor=colors[cls], alpha=0.5), fontsize=12, color='white')
    plt.show()
