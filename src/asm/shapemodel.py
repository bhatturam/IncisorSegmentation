import numpy as np

from pca import ModedPCAModel
from shape import ShapeList, Shape


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

    def __init__(self, shape_list, pca_variance_captured=0.9, gpa_tol=1e-7, gpa_max_iters=10000):
        """
        Constructs the Active Shape Model based on the given list of Shapes.
        :param shapes: A list of Shapes
        :param pca_variance_captured: The fraction of variance to be captured by the shape model
        :param gpa_tol: tol: The convergence threshold for gpa
        (Default: 1e-7)
        :param gpa_max_iters: The maximum number of iterations
        permitted for gpa (Default: 10000)
        """
        self._shapes = shape_list
        self._aligned_shapes = self._shapes.align(gpa_tol, gpa_max_iters)
        self._model = ModedPCAModel(self._aligned_shapes.collapse(), pca_variance_captured)

    def aligned_shapes(self):
        """
        Returns the gpa aligned shapes
        :return: A list of Shape objects
        """
        return self._aligned_shapes

    def shapes(self):
        """
        Returns the shapes used to make the model
        :return: A list of Shape objects
        """
        return self._shapes

    def mean_shape_unaligned(self):
        """
        Returns the mean shape of the unaligned shapes
        :return: A Shape object
        """
        return self._shapes.mean()

    def mean_shape(self):
        """
        Returns the mean shape of the model
        :return: A Shape object containing the mean shape
        """
        return self._aligned_shapes.mean()

    def fit(self, shape, tol=1e-7, max_iters=10000):
        """
        Refer Protocol 1 - Page 9 of
         An Active Shape Model based on
        Cootes, Tim, E. R. Baldock, and J. Graham.
        "An introduction to active shape models."
        Image processing and analysis (2000): 223-248.
        :param shape: The shape to fit the model to
        :return The fitted Shape and the mean squared error
        """

        factors = np.zeros(self._model.modes())
        current_fit = Shape.from_collapsed_shape(self._model.mean() + self._model.generate_deviation(factors))
        htmatrix = current_fit.homogeneous_transformation_matrix(shape)
        for num_iters in range(max_iters):
            old_factors = factors.copy()
            current_fit = Shape.from_collapsed_shape(self._model.mean() + self._model.generate_deviation(factors))
            #print current_fit.norm()
            htmatrix = current_fit.homogeneous_transformation_matrix(shape)
            inv_tmat = np.linalg.pinv(htmatrix)
            # collapsed_shape = (Shape.from_homogeneous_coordinates(np.dot(shape.raw_homogeneous(), inv_tmat))).collapse()
            collapsed_shape = shape.transform(inv_tmat).collapse()
            collapsed_shape = shape.align(current_fit).collapse()
            #print Shape.from_collapsed_shape(collapsed_shape).norm()
            tangent_factor = np.dot(collapsed_shape, self._model.mean());
            tangent_projection = collapsed_shape / (tangent_factor)
            error, factors = self._model.fit(tangent_projection)
            # error, factors = self._model.fit(collapsed_shape)
            if np.linalg.norm(old_factors - factors) < tol:
                break  # stuff by bharath
        return current_fit.transform(htmatrix), error
        #return current_fit.align(shape), error

    def fit_useless(self, shape):

        """
        USELESS FOR NOW
        One shot fits the model to the iniital shape
        :param shape: The shape to fit the model to
        :return: The fitted Shape and the mean squared error
        """
        _, factors = self._model.fit(shape.center().align(self.mean_shape()).collapse())
        fitted_shape = Shape.from_collapsed_shape(self._model.mean() + self._model.generate_deviation(factors))
        fitted_shape = fitted_shape.align(shape.center()).translate(shape.mean())
        fit_error = np.mean(np.sqrt(np.sum((fitted_shape.raw() - shape.raw()) ** 2)))
        return fitted_shape, fit_error

    def mean_shape_projected(self):
        """
        Returns the mean shape of the aligned shapes scaled and rotated to the
        mean of the unaligned shapes translated by the centroid of the mean of the unaligned shapes.
        This should be a good initial position of the model
        :return: A Shape object
        """
        return self.mean_shape().align(self.mean_shape_unaligned().center()).translate(
            self.mean_shape_unaligned().mean())

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
