import matplotlib.pyplot as plt
import numpy as np
import cv2


def imshow2(img, width=12, height=12):
    plt.figure(figsize=(width, height))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))


def plot_shapes(shape_list):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for shape in shape_list:
        x_list, y_list = zip(*(shape.as_list_of_points()))
        ax.plot((-1 * np.array(x_list)).tolist(), (-1 * np.array(y_list)).tolist())
    plt.show()


def overlay_shapes_on_image(img, shape_list):
    im = img.copy()
    cv2.polylines(im, np.int32([shape.as_numpy_matrix() for shape in shape_list]), True, (0, 255, 255))
    return im


def overlay_points_on_image(img, points):
    im = img.copy()
    for point in points:
        cv2.circle(im, (point[0], point[1]), 1, (0, 255, 255), 1)
    return im
