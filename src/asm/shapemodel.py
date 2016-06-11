import numpy as np

from asm.pca import ModedPCAModel
from asm.shape import AlignedShapeList, Shape


class ShapeModel:
    """ An Active Shape Model based on
    Cootes, Tim, E. R. Baldock, and J. Graham.
    "An introduction to active shape models."
    Image processing and analysis (2000): 223-248.

    Attributes:
    _aligned_shapes An AlignedShapeList containing
    the training shapes
    _model  The underlying modedPCA Model

    Authors: David Torrejon and Bharath Venkatesh

"""

    def __init__(self, shapes, pca_variance_captured=0.9, gpa_tol=1e-7, gpa_max_iters=10000):
        """
        Constructs the Active Shape Model based on the given list of Shapes.
        :param shapes: A list of Shapes
        :param pca_variance_captured: The fraction of variance to be captured by the shape model
        :param gpa_tol: tol: The convergence threshold for gpa
        (Default: 1e-7)
        :param gpa_max_iters: The maximum number of iterations
        permitted for gpa (Default: 10000)
        """
        self._aligned_shapes = AlignedShapeList(shapes, gpa_tol, gpa_max_iters)
        self._model = ModedPCAModel(self._aligned_shapes.raw(), pca_variance_captured)
        self._initial_translation = self._compute_initial_translation(shapes)

    def aligned_shapes(self):
        """
        Returns the gpa aligned shapes
        :return: A list of Shape objects containing the aligned shapes
        """
        return self._aligned_shapes.shapes()

    def mean_rotation(self):
        """
        Returns the mean rotation matrix
        :return: the mean rotation matrix
        """
        return self._aligned_shapes.mean_rotation()

    def mean_shape(self):
        """
        Returns the mean shape of the model
        :return: A Shape object containing the mean shape
        """
        return self._aligned_shapes.mean_shape()

    def modes(self):
        """
        Returns the number of modes of the model
        :return: the number of modes
        """
        return self._model.modes()

    def generate_shape(self, factors):
        """
        Generates a shape based on a vector of factors of size
        equal to the number of modes of the model, with element
        values between -1 and 1
        :param factors: A vector of size modes() with values
        between -1 and 1
        :return: A Shape object containing the generated shape
        """
        return Shape(
            self.mean_shape().raw() + Shape.from_collapsed_shape(self._model.generate_deviation(factors)).raw())

    def mode_shapes(self, m):
        """
        Returns the modal shapes of the model (Variance limits)
        :param m: A list of Shape objects containing the modal shapes
        """
        if m < 0 or m >= self.modes():
            raise ValueError('Number of modes must be within [0,modes()-1]')
        factors = np.zeros(self.modes())
        mode_shapes = []
        for i in range(-1, 2):
            factors[m] = i
            mode_shapes.append(self.generate_shape(factors))
        return mode_shapes

    def get_initial_translation(self):
        return self._initial_translation

    def _compute_initial_translation(self, shapes):
        """
        params:
            shapes: list of shapes
        Returns:
            the mean of the first x, and first y for the first value of the landmark (topleft corner)
        """
        point_matrix = []
        for shape in shapes:
            point_matrix.append(shape.raw())
        point_matrix=np.array(point_matrix)
        point_matrix = np.uint32(np.round(np.mean(point_matrix,axis=0)))
        return Shape(point_matrix)
