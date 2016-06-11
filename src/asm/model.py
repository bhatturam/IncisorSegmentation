import numpy as np
from shapemodel import ShapeModel
from greymodel import GreyModel
from shape import get_bounding_box

class ActiveShapeModel:
    def __init__(self, images, shapes, grey_model_number_of_pixels=5,
                 grey_model_normal_point_neighborhood=4,
                 grey_model_use_gradient=False,
                 grey_model_normalize=False,
                 grey_model_pca_variance_captured=0.9,
                 shape_model_pca_variance_captured=0.9,
                 shape_model_gpa_tol=1e-7,
                 shape_model_gpa_max_iters=10000, max_pyr_down=3):
        self._grey_model = GreyModel(images, shapes, grey_model_number_of_pixels, grey_model_pca_variance_captured,
                                     grey_model_normal_point_neighborhood, grey_model_use_gradient,
                                     grey_model_normalize)
        self._shape_model = ShapeModel(shapes, shape_model_pca_variance_captured, shape_model_gpa_tol,
                                       shape_model_gpa_max_iters)



        def get_origin_point_from_scaled_down(self, test_image):
            """
                Scales down the test_image and then brute forces on the max_pyr_down level
                searching for the best initial position
                params:

                returns:

            """
            aligned_shapes_bb = get_bounding_box(self._shape_model.aligned_shapes())
            original_shapes_bb = get_bounding_box(self._shape_model.get_initial_translation())
            scale_factor = np.diff(original_shapes_bb, axis=0)/np.diff(aligned_shapes_bb, axis=0)
            for i in range(self.max_pyr_down):
                test_image = cv2.pyrDown(test_image)
                point_matrix_level = np.uint32(np.round(point_matrix_level/2))
