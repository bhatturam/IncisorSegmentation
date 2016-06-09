import cv2
import numpy as np
import math

from asm.pca import ModedPCAModel


class GreyModel:
    """ A grey level point model based on
        Cootes, Timothy F., and Christopher J. Taylor.
         "Active Shape Model Search using Local Grey-Level Models:
         A Quantitative Evaluation." BMVC. Vol. 93. 1993.

            Attributes:
                _models: A list of ModedPCA models of the grey levels
                         of the landmark points


            Authors: David Torrejon and Bharath Venkatesh

    """

    def size(self):
        return len(self._models)

    def fit_quality(self, point_index, model_factors, test_image, test_shape):
        model_patch = self.generate_grey(point_index, model_factors)
        test_patch = self._get_normal_grey_levels_for_single_point_single_image(test_image, test_shape, point_index)
        return cv2.Mahalanobis(model_patch, test_patch,
                               np.linalg.pinv(cv2.calcCovarMatrix(np.concatenate((model_patch, test_patch)))))

    def _get_point_normal_slope(self, shape, point_index):
        neighborhood = shape.get_neighborhood(point_index, self._normal_neighborhood)
        line = cv2.fitLine(neighborhood, cv2.DIST_L2, 0, 0.01, 0.01)
        return line[0:2] / math.sqrt(np.sum(line[0:2] ** 2))

    def _get_point_normal_pixel_coordinates_train(self, shape, point_index):
        """
        Get the coordinates of pixels lying on the normal of the point
        :param shape:
        :param point_index:
        :return:
        """
        point = shape.get_point(point_index)
        slope = self._get_point_normal_slope(shape, point_index)
        return [[int(point[1] + (incr * slope[0]) + 0.5), int(point[0] - (incr * slope[1]) + 0.5)] for incr in
                range(-self._number_of_pixels, self._number_of_pixels + 1)]

    def _get_normal_grey_levels_for_single_point_single_image(self, image, shape, point_index):
        coordinate_list = self._get_point_normal_pixel_coordinates_train(shape, point_index)
        data = np.zeros((2 * self._number_of_pixels + 1,), dtype=float)
        ctr = 0
        h, w = image.shape
        for coordinates in coordinate_list:
            if 0 <= coordinates[0] < h and 0 <= coordinates[1] < w:
                data[ctr] = image[coordinates[0]][coordinates[1]]
            ctr += 1
        return data

    def getModel(self, point_index):
        return self._models[point_index]

    def generate_grey(self, point_index, factors):
        """
        Generates a grey vector based on a vector of factors of size
        equal to the number of modes of the model, with element
        values between -1 and 1
        :param point_index: The index of the landmark
        :param factors: A vector of size modes() with values
        between -1 and 1
        :return: A vector containing the grey levels of a point
        """
        return self._models[point_index].mean() + self._models[point_index].generate_deviation(factors)

    def mode_greys(self, point_index, m):
        """
        Returns the modal grey levels of the model (Variance limits)
        :param m: A list of vectors containing the modal greys
        """
        if m < 0 or m >= self._models[point_index].modes():
            raise ValueError('Number of modes must be within [0,modes()-1]')
        factors = np.zeros(self._models[point_index].modes())
        mode_greys = []
        for i in range(-1, 2):
            factors[m] = i
            mode_greys.append(self.generate_grey(point_index, factors))
        return mode_greys

    def __init__(self, images, shapes, number_of_pixels=5, pca_variance_captured=0.9, normal_point_neighborhood=4,
                 use_gradient=False, normalize=False):
        self._normal_neighborhood = normal_point_neighborhood
        self._number_of_pixels = number_of_pixels
        self._normalize = normalize
        self._use_gradient = use_gradient
        self._models = []
        for i in range(shapes[0].size()):
            plist = []
            for j in range(len(images)):
                levels = self._get_normal_grey_levels_for_single_point_single_image(images[j], shapes[j], i)
                if self._use_gradient:
                    levels = np.diff(levels)
                if self._normalize:
                    levels = np.divide(levels, np.sqrt(np.sum(levels ** 2)))
                plist.append(levels)
            pdata = np.array(plist)
            self._models.append(ModedPCAModel(pdata, pca_variance_captured))
