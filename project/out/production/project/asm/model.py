import numpy as np
from shapemodel import ShapeModel
from greymodel import GreyModel


class ActiveShapeModel:
    def __init__(self, images, shapes, grey_model_number_of_pixels=5,
                 grey_model_normal_point_neighborhood=4,
                 grey_model_use_gradient=False,
                 grey_model_normalize=False,
                 grey_model_pca_variance_captured=0.9,
                 shape_model_pca_variance_captured=0.9,
                 shape_model_gpa_tol=1e-7,
                 shape_model_gpa_max_iters=10000):
        self._grey_model = GreyModel(images, shapes, grey_model_number_of_pixels, grey_model_pca_variance_captured,
                                     grey_model_normal_point_neighborhood, grey_model_use_gradient,
                                     grey_model_normalize)
        self._shape_model = ShapeModel(shapes, shape_model_pca_variance_captured, shape_model_gpa_tol,
                                       shape_model_gpa_max_iters)
