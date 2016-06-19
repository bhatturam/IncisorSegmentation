import numpy as np
from pca import PCAModel
from shape import Shape

__author__ = "David Torrejon and Bharath Venkatesh"

"""
Active Shape Models
"""


class ModedPCAModel:
    """ A Moded Principal Component Analysis Model to perform
        dimensionality reduction. Implementation is based on

        Cootes, Tim, E. R. Baldock, and J. Graham.
        "An introduction to active shape models."
        Image processing and analysis (2000): 223-248.

        Attributes:
        _model  The underlying PCA Model
        _modes The number of modes of the model
        _bmax The limits of variation of the shape model

        Authors: David Torrejon and Bharath Venkatesh

    """

    def __init__(self, x, pca_variance_captured=0.9):
        """
        Constructs the Active Shape Model based on the given list of Shapes.
        :param x: The data matrix
        :param pca_variance_captured: The fraction of variance to be captured by the moded model
        """
        self._model = PCAModel(x)
        self._modes = self._model.get_k_cutoff(pca_variance_captured)
        self._b_max = np.multiply(np.sqrt(self._model.get_eigenvalues()[0:self._modes]), 3.0)

    def get_mean(self):
        """
        Returns the mean
        :return: A vector containing the model mean
        """
        return self._model.get_mean()

    def get_number_of_modes(self):
        """
        Returns the number of modes of the model
        :return: the number of modes
        """
        return self._modes

    def _p(self):
        return self._model.get_eigenvectors()[:, 0:self._modes]

    def generate(self, factors):
        """
        Generates a new point based
        on a vector of factors of size equal to the number of
        modes of the model, with element
        values between -1 and 1
        :param factors: A vector of size modes() with values
        between -1 and 1
        :return: A vector containing the generated point
        """
        p = self._p()
        pb = np.dot(p, np.multiply(factors, self._b_max))
        return self.get_mean() + pb

    def fit(self, data):
        """
        Fits the model to a new data point and returns the best factors
        :param data: A data point - array of the size of the mean
        :return: fit error,predicted point,factors array of the size of the mean
        """
        if data.shape == self.get_mean().shape:
            bcand = np.squeeze(np.dot(self._p().T, (data - self.get_mean())))
            factors = np.zeros(bcand.shape)
            for i in range(len(bcand)):
                val = bcand[i] / self._b_max[i]
                if val > 1:
                    val = 1
                elif val < -1:
                    val = -1
                factors[i] = val
            pred = self.generate(factors)
            error = np.sqrt(np.sum((data - pred) ** 2))
            return error, pred, factors
        else:
            raise TypeError(
                "Data has to be of the same size as model - Expected " + str(self.get_mean().shape) + " Got " +
                str(data.shape))


class GaussianModel:
    """ A Gaussian Model to perform
    dimensionality reduction. Implementation is based on
    page 14 of
    Cootes, Tim, E. R. Baldock, and J. Graham.
    "An introduction to active shape models."
    Image processing and analysis (2000): 223-248.

    Attributes:
    _cov  The covariance matrix
    _cov_inv The inverse of the covariance matrix
    _modes The mean

    Authors: David Torrejon and Bharath Venkatesh

    """

    def __init__(self, x):
        """
        Constructs the Active Shape Model based on the given list of Shapes.
        :param x: The data matrix
        """
        self._mean = np.mean(x, axis=0)
        self._cov = np.cov(x.T)
        self._cov_inv = np.linalg.pinv(self._cov)

    def get_mean(self):
        """
        Returns the mean
        :return: A vector containing the model mean
        """
        return self._mean.copy()

    def get_covariance(self):
        """
        Returns the covariance
        :return: A vector containing the model covariance
        """
        return self._cov.copy()

    def generate(self, factors=None):
        """
        Just returns the mean
        :param factors: Unused
        :return: A vector containing the generated point
        """
        return self.get_mean()

    def fit(self, test_point):
        """
        Returns the error (Mahalanobis distance) when test_point is compared to the mean of the model
        and the fit(just the mean) and the factors(again the mean)
        :param test_point: The test point
        :return: error,the mean, empty matrix
        """
        g_diff = test_point - self._mean
        error = np.dot(np.dot(g_diff.T, self._cov_inv), g_diff)
        return error, self.generate(), np.array([])


class PointDistributionModel:
    """
    An Point Distribution Model based on
    Cootes, Tim, E. R. Baldock, and J. Graham.
    "An introduction to active shape models."
    Image processing and analysis (2000): 223-248.

    Attributes:
    _aligned_shapes An AlignedShapeList containing
    the training shapes
    _model  The underlying modedPCA Model

    Authors: David Torrejon and Bharath Venkatesh

    """

    def __init__(self, shape_list, pca_variance_captured=0.9, gpa_tol=1e-7, gpa_max_iters=10000, shape_fit_tol=1e-7,
                 shape_fit_max_iters=10000):
        """
        Constructs the Active Shape Model based on the given list of Shapes.
        :param shape_list: A ShapeList object
        :param pca_variance_captured: The fraction of variance to be captured by the shape model
        :param gpa_tol: tol: The convergence threshold for gpa
        (Default: 1e-7)
        :param gpa_max_iters: The maximum number of iterations
        permitted for gpa (Default: 10000)
        """
        self._shapes = shape_list
        self._aligned_shapes = self._shapes.align(gpa_tol, gpa_max_iters)
        self._model = ModedPCAModel(self._aligned_shapes.as_collapsed_vector(), pca_variance_captured)
        self._shape_fit_tol = shape_fit_tol
        self._shape_fit_max_iters = shape_fit_max_iters

    def get_moded_pca_model(self):
        """
        Returns the underlying moded PCA Model
        :return: ModedPCAModel
        """
        return self._model

    def get_number_of_modes(self):
        """
        Returns the number of modes of the model
        :return: the number of modes
        """
        return self._model.get_number_of_modes()

    def get_size(self):
        """
        Returns the number of points
        :return: Number of points
        """
        return self.get_mean_shape().get_size()

    def get_aligned_shapes(self):
        """
        Returns the gpa aligned shapes
        :return: A list of Shape objects
        """
        return self._aligned_shapes

    def get_shapes(self):
        """
        Returns the shapes used to make the model
        :return: A list of Shape objects
        """
        return self._shapes

    def get_mean_shape_unaligned(self):
        """
        Returns the mean shape of the unaligned shapes
        :return: A Shape object
        """
        return self._shapes.get_mean_shape()

    def get_mean_shape(self):
        """
        Returns the mean shape of the model
        :return: A Shape object containing the mean shape
        """
        return self._aligned_shapes.get_mean_shape()

    def get_mean_shape_projected(self):
        """
        Returns the mean shape of the aligned shapes scaled and rotated to the
        mean of the unaligned shapes translated by the centroid of the mean of the unaligned shapes.
        This should be a good initial position of the model
        :return: A Shape object
        """
        return self.get_mean_shape().align(self.get_mean_shape_unaligned())

    def generate(self, factors):
        """
        Generates a shape based on a vector of factors of size
        equal to the number of modes of the model, with element
        values between -1 and 1
        :param factors: A vector of size modes() with values
        between -1 and 1
        :return: A Shape object containing the generated shape
        """
        return Shape.from_collapsed_vector(self._model.generate(factors))

    def generate_mode_shapes(self, m):
        """
        Returns the modal shapes of the model (Variance limits)
        :param m: A list of Shape objects containing the modal shapes
        """
        num_modes = self.get_number_of_modes()
        if m < 0 or m >= num_modes:
            raise ValueError('Number of modes must be within [0,modes()-1]')
        factors = np.zeros(num_modes)
        mode_shapes = []
        for i in range(-1, 2):
            factors[m] = i
            mode_shapes.append(self.generate(factors))
        return mode_shapes

    # def fit(self, shape):
    #     """
    #     Refer Protocol 1 - Page 9 of
    #      An Active Shape Model based on
    #     Cootes, Tim, E. R. Baldock, and J. Graham.
    #     "An introduction to active shape models."
    #     Image processing and analysis (2000): 223-248.
    #     :param shape: The shape to fit the model to
    #     :return The fitted Shape and the mean squared error
    #     """
    #     factors = np.zeros(self._model.get_number_of_modes())
    #     current_fit = Shape.from_collapsed_vector(self._model.generate(factors))
    #     num_iters = 0
    #     error = float("inf")
    #     for num_iters in range(self._shape_fit_max_iters):
    #         old_factors = factors.copy()
    #         current_fit = Shape.from_collapsed_vector(self._model.generate(factors))
    #         collapsed_shape = shape.align(current_fit).as_collapsed_vector()
    #         error, _, factors = self._model.fit(collapsed_shape)
    #         if np.linalg.norm(old_factors - factors) < self._shape_fit_tol:
    #             break
    #     return current_fit.align(shape), error, num_iters
    def fit(self, shape):
        factors = np.zeros(self._model.get_number_of_modes())
        current_fit = self.generate(factors)
        hmat = current_fit.get_transformation(shape)
        num_iters = 0
        error = float("inf")
        for num_iters in range(self._shape_fit_max_iters):
            old_factors = factors.copy()
            current_fit = self.generate(factors)
            hmat = current_fit.get_transformation(shape)
            collapsed_shape = shape.transform(np.linalg.pinv(hmat)).project_to_tangent_space(
                current_fit).as_collapsed_vector()
            # collapsed_shape = shape.align(current_fit).as_collapsed_vector()
            error, _, factors = self._model.fit(collapsed_shape)
            if np.linalg.norm(old_factors - factors) < self._shape_fit_tol:
                break
        return current_fit.transform(hmat), error, num_iters


class GreyModel:
    """ A grey level point model based on
    Cootes, Timothy F., and Christopher J. Taylor.
     "Active Shape Model Search using Local Grey-Level Models:
     A Quantitative Evaluation." BMVC. Vol. 93. 1993.
     and
     An Active Shape Model based on
        Cootes, Tim, E. R. Baldock, and J. Graham.
        "An introduction to active shape models."
        Image processing and analysis (2000): 223-248.

        Attributes:
            _point_models: The list of underlying point grey models (GaussianModel or ModedPCAModel)

        Authors: David Torrejon and Bharath Venkatesh

    """

    def __init__(self, training_images, training_shape_list, patch_num_pixels, search_num_pixels, use_gradient=False,
                 normalize_patch=False, use_moded_pca_model=False, mpca_variance_captured=0.9,
                 normal_point_neighborhood=4):
        self._point_models = []
        self._using_pca_model = use_moded_pca_model
        self._search_num_pixels = search_num_pixels
        self._patch_num_pixels = patch_num_pixels
        self._use_gradient = use_gradient
        self._normalize = normalize_patch
        self._normal_neighborhood = normal_point_neighborhood
        for i in range(training_shape_list[0].get_size()):
            patch_data_list = []
            for j in range(len(training_images)):
                coordinate_list, _, _ = self._get_point_coordinates_along_normal(training_images[j],
                                                                                 training_shape_list[j], i,
                                                                                 self._patch_num_pixels)
                levels = self._get_grey_data(training_images[j], coordinate_list)
                patch_data_list.append(levels)
            patch_data = np.array(patch_data_list)
            if self._using_pca_model:
                self._point_models.append(ModedPCAModel(patch_data, pca_variance_captured=mpca_variance_captured))
            else:
                self._point_models.append(GaussianModel(patch_data))

    def _get_grey_data(self, image, coordinate_list):
        data = np.array([image[coordinate[1], coordinate[0]] for coordinate in coordinate_list])
        if self._use_gradient:
            data = np.gradient(data)
        if self._normalize:
            norm_val = np.sum(np.abs(data))
            if norm_val > 0:
                data = np.divide(data, norm_val)
        return data

    def _get_point_coordinates_along_normal(self, image, shape, point_index, number_of_pixels, break_on_error=True):
        h, w = image.shape
        coordinate_list = []
        generator = shape.get_normal_at_point_generator(point_index, self._normal_neighborhood)
        increments = range(-number_of_pixels, number_of_pixels + 1)
        for increment in increments:
            coordinates = np.int32(np.round(generator(increment)))
            if 0 <= coordinates[1] < h and 0 <= coordinates[0] < w:
                coordinate_list.append(coordinates)
            elif break_on_error:
                raise ValueError("Index exceeds image dimensions")
        return coordinate_list, generator, increments

    def get_size(self):
        """
        Returns the number of grey point models - i.e the number of landmarks
        :return: Number of point models
        """
        return len(self._point_models)

    def get_point_grey_model(self, point_index):
        """
        :param point_index: The index of the landmark
        :return: The modedPCAModel for the landmark
        """
        return self._point_models[point_index]

    def search(self, test_image, initial_shape):
        """
        Searches for the best positions of the shape points in the test image
        :param test_image: The test image
        :param initial_shape: The initial shape
        :return: The new shape, and the array of errors - empty if the shape hasnt moved
        """
        point_list = []
        error_list = []
        for point_index in range(self.get_size()):
            coordinate_list, _, _ = self._get_point_coordinates_along_normal(test_image, initial_shape, point_index,
                                                                             self._search_num_pixels,
                                                                             break_on_error=False)
            test_patch = self._get_grey_data(test_image, coordinate_list)
            grey_model = self._point_models[point_index]
            patch_size = len(grey_model.get_mean())
            if len(test_patch) < patch_size:
                return initial_shape, np.array([])
            min_index = 0
            select_range = range(min_index, min_index + patch_size)
            min_error, _, _ = grey_model.fit(test_patch[select_range])
            for i in range(1 + len(test_patch) - patch_size):
                select_range = range(i, i + patch_size)
                error, _, _ = grey_model.fit(test_patch[select_range])
                if error < min_error:
                    min_index = i
                    min_error = error
            point_list.append(coordinate_list[min_index + self._patch_num_pixels])
            error_list.append(min_error)
        return Shape(np.array(point_list)), np.array(error_list)


class ActiveShapeModel:
    """ An Active Shape Model based on
    Cootes, Tim, E. R. Baldock, and J. Graham.
    "An introduction to active shape models."
    Image processing and analysis (2000): 223-248.

    Attributes:
    _pdm The underlying point distribution model (PointDistributionModel)
    _gm The underlying grey model (ModedPCAModel or GaussianModel)
    Authors: David Torrejon and Bharath Venkatesh
    """

    def __init__(self, point_distribution_model, grey_model):
        self._pdm = point_distribution_model
        self._gm = grey_model

    def get_pdm(self):
        """
        Returns the underlying PointDistributionModel
        :return: PointDistributionModel
        """
        return self._pdm

    def get_gm(self):
        """
        Returns the underlying GreyModel
        :return: GreyModel
        """
        return self._gm

    def get_default_initial_shape(self):
        """
        Returns the default initial shape
        :return: Shape object
        """
        return self._pdm.get_mean_shape_projected()

    def fit(self, test_image, tol=0.5, max_iters=10000, initial_shape=None):
        """
        Fits the shape model to the test image to find the shape
        :param test_image: The test image in the form of a numpy matrix
        :param tol: Fraction of points changed
        :param max_iters: Maximum number of iterations
        :param initial_shape: The starting Shape - if None get_default_initial_shape() is used
        :return: The final Shape, the fit error and the number of iterations performed
        """
        if initial_shape is None:
            current_shape = self.get_default_initial_shape()
        else:
            current_shape = initial_shape.round()
        num_iter = 0
        fit_error = float("inf")
        for num_iter in range(max_iters):
            previous_shape = Shape(current_shape.as_numpy_matrix().copy())
            new_shape_grey, error_list = self._gm.search(test_image, current_shape)
            current_shape, fit_error, num_iters = self._pdm.fit(new_shape_grey)
            moved_points = np.sum(
                np.sum(current_shape.as_numpy_matrix().round() - previous_shape.as_numpy_matrix().round(),
                       axis=1) > 0)
            if moved_points / float(self._pdm.get_size()) < tol:
                break
        return current_shape, fit_error, num_iter