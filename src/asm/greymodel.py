import cv2
import numpy as np
import math

from pca import ModedPCAModel


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
        """
        Returns the number of grey point models - i.e the number of landmarks
        :return: Number of point models
        """
        return len(self._models)

    # def fit_quality(self, point_index, model_factors, test_image, test_shape):
    #    model_patch = self.generate_grey(point_index, model_factors)
    #    test_patch = self._get_normal_grey_levels_for_single_point_single_image(test_image, test_shape, point_index)
    #    return cv2.Mahalanobis(model_patch, test_patch,
    #                           np.linalg.pinv(cv2.calcCovarMatrix(np.concatenate((model_patch, test_patch)), 1)))

    def _extract_grey_data(self, image, shape, point_index, number_of_pixels):
        """
        Get the grey level data for the given image and shape for the specified landmark
        :param image: The actual grayscale image
        :param shape: The current shape
        :param point_index: The query point index in shape
        :return: A vector of grey level data number_of_pixels wide
        """
        data = np.zeros((2 * number_of_pixels + 1,), dtype=float)
        ctr = 0
        h, w = image.shape
        generator = shape.get_normal_at_point_generator(point_index, self._normal_neighborhood)
        for increment in range(-number_of_pixels, number_of_pixels + 1):
            coordinates = np.int32(np.round(generator(increment)))
            if 0 <= coordinates[1] < h and 0 <= coordinates[0] < w:
                data[ctr] = image[coordinates[1]][coordinates[0]]  # opencv y x problems
            ctr += 1
        if self._use_gradient:
            data = np.diff(data)
        if self._normalize:
            data = np.divide(data, np.sqrt(np.sum(data ** 2)))
        return data

    def grey_model_point(self, point_index):
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
        :param point_index: The index of the landmark
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

    def __init__(self, images, shape_list, number_of_pixels=5, pca_variance_captured=0.9, normal_point_neighborhood=4,
                 use_gradient=False, normalize=False):
        self._normal_neighborhood = normal_point_neighborhood
        self._number_of_pixels = number_of_pixels
        self._normalize = normalize
        self._use_gradient = use_gradient
        self._models = []
        for i in range(shape_list.tolist()[0].size()):
            plist = []
            for j in range(len(images)):
                levels = self._extract_grey_data(images[j], shape_list[j], i, self._number_of_pixels)
                plist.append(levels)
            pdata = np.array(plist)
            self._models.append(ModedPCAModel(pdata, pca_variance_captured))
