import warnings
import cv2
import numpy as np
import math
from shape import Shape

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
        if self.simple_model:
            return len(self.list_means_points)
        else:
            return len(self._models)

    def fit_quality(self, point_index, test_patch):
        g_diff = test_patch-self.list_means_points[point_index]
        return np.dot(np.dot(g_diff.T, self.list_cov_mat_points[point_index]), g_diff)


    def _get_grey_data(self, image, coordinate_list):
        data = np.array([image[coordinate[1], coordinate[0]] for coordinate in coordinate_list])
        if self._use_gradient:
            data = np.diff(data)
        if self._normalize:
            normval = np.sum(np.abs(data))
            if normval > 0:
                data = np.divide(data, normval)
        return data

    def _get_point_coordinates_along_normal(self, image, shape, point_index, number_of_pixels, break_on_error=True):
        h, w = image.shape
        coordinate_list = []
        generator = shape.normal_at_point_generator(point_index, self._normal_neighborhood)
        increments = range(-number_of_pixels, number_of_pixels + 1)
        for increment in increments:
            coordinates = np.int32(np.round(generator(increment)))
            if 0 <= coordinates[1] < h and 0 <= coordinates[0] < w:
                coordinate_list.append(coordinates)
            elif break_on_error:
                raise ValueError("Index exceeds image dimensions")
        return coordinate_list, generator, increments

    def grey_model_point(self, point_index):
        """
        :param point_index: The index of the landmark
        :return: The modedPCAModel for the landmark
        """
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

    def search(self, test_image, initial_shape, search_number_of_pixels=60):
        """
        Searches for the best positions of the shape points in the test image
        :param test_image: The test image
        :param initial_shape: The initial shape
        :param search_number_of_pixels: The number of pixels to search along normal
        :return: The new shape, and the array of errors - empty if the shape hasnt moved
        """
        point_list = []
        error_list = []
        h, w = test_image.shape
        for point_index in range(self.size()):
            coordinate_list, _, _ = self._get_point_coordinates_along_normal(test_image, initial_shape, point_index,
                                                                             search_number_of_pixels,
                                                                             break_on_error=False)
            #rint coordinate_list
            test_patch = self._get_grey_data(test_image, coordinate_list)
            grey_model = None
            patch_size = 0
            if self.simple_model:
                patch_size = len(self.list_means_points[point_index])
            else:
                patch_size = len(grey_model.mean())
                grey_model = self.grey_model_point(point_index)
            if len(test_patch) < patch_size:
                return initial_shape, np.array([])

            min_index = 0
            select_range = range(min_index, min_index + patch_size)
            if self.simple_model is False:
                min_error, _ = grey_model.fit(test_patch[select_range])
            else:
                min_error = self.fit_quality(test_patch=test_patch[select_range], point_index=point_index)
            for i in range(1 + len(test_patch) - patch_size):
                select_range = range(i, i + patch_size)
                #error, _ = grey_model.fit(test_patch[select_range])

                if self.simple_model is False:
                    error, _ = grey_model.fit(test_patch[select_range])
                else:
                    error = self.fit_quality(test_patch=test_patch[select_range], point_index=point_index)

                if error < min_error:
                    min_index = i
                    min_error = error
            point_list.append(coordinate_list[min_index + self._number_of_pixels])
            error_list.append(min_error)
        return Shape(np.array(point_list)), np.array(error_list)

    def __init__(self, images, shape_list, number_of_pixels_model=60, pca_variance_captured=0.9,
                 normal_point_neighborhood=4,
                 use_gradient=False, normalize=False, simple_model = True):
        self.simple_model = simple_model
        self._normal_neighborhood = normal_point_neighborhood
        self._number_of_pixels = number_of_pixels_model
        self._normalize = normalize
        self._use_gradient = use_gradient
        self._models = []
        self.list_cov_mat_points = []
        self.list_means_points = []
        for i in range(shape_list.tolist()[0].size()):
            plist = []
            for j in range(len(images)):
                coordinate_list, _, _ = self._get_point_coordinates_along_normal(images[j], shape_list[j], i,
                                                                                 self._number_of_pixels)
                levels = self._get_grey_data(images[j], coordinate_list)
                plist.append(levels)
            pdata = np.array(plist)
            if self.simple_model:
                self.list_cov_mat_points.append(np.linalg.pinv(np.cov(pdata.T)))
                self.list_means_points.append(np.mean(pdata, axis=0))

            else:
                self._models.append(ModedPCAModel(pdata, pca_variance_captured))
