import numpy as np
from shapemodel import ShapeModel
from greymodel import GreyModel
from shape import Shape, ShapeList


class ActiveShapeModel:
    def __init__(self, images, shape_list, grey_model_number_of_pixels=15,
                 grey_model_search_number_of_pixels=30,
                 grey_model_normal_point_neighborhood=4,
                 grey_model_use_gradient=True,
                 grey_model_normalize=True,
                 grey_model_pca_variance_captured=0.9,
                 shape_model_pca_variance_captured=0.9,
                 shape_model_gpa_tol=1e-7,
                 shape_model_gpa_max_iters=10000,
                 shape_fit_tol=1e-7, shape_fit_max_iters=10000
                 ):
        self._grey_model = GreyModel(images, shape_list, grey_model_number_of_pixels,
                                     grey_model_pca_variance_captured,
                                     grey_model_normal_point_neighborhood, grey_model_use_gradient,
                                     grey_model_normalize)
        self._shape_model = ShapeModel(shape_list, shape_model_pca_variance_captured, shape_model_gpa_tol,
                                       shape_model_gpa_max_iters)

        self._shape_fit_tol = shape_fit_tol
        self._shape_fit_max_iters = shape_fit_max_iters
        self._grey_model_search_number_of_pixels = grey_model_search_number_of_pixels

    def initial_shape(self):
        return self._shape_model.mean_shape_projected()

    def fit(self, test_image, combined_fit_tol=1e-7, combined_fit_max_iters=10000):
        current_shape = self.initial_shape()
        for num_iter in range(combined_fit_max_iters):
            previous_shape = Shape(current_shape.raw().copy())
            new_shape_grey, error_list = self._grey_model.search(test_image, current_shape,
                                                                 self._grey_model_search_number_of_pixels)

            current_shape, fit_error,num_iters = self._shape_model.fit(new_shape_grey, self._shape_fit_tol,
                                                             self._shape_fit_max_iters)
            if np.linalg.norm(current_shape.raw() - previous_shape.raw()) < combined_fit_tol:
                break
        return current_shape, fit_error, num_iter
