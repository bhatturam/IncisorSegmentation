import math
import numpy as np
from bisect import bisect_left


class PCAModel:
    """ A Principal Component Analysis Model to perform
        dimensionality reduction. Implementation is based on
        Appendix B and C of

        Cootes, Tim, E. R. Baldock, and J. Graham.
        "An introduction to active shape models."
        Image processing and analysis (2000): 223-248.

        Attributes:
        _x  The centered data matrix (Nxd)
        _mean The mean of the input data (1xd)
        _lambdas The eigenvalues of the PCA Matrix (1xd)
        _eigenvectors The eigenvectors of the PCA Matrix (Nxd)

        Authors: David Torrejon and Bharath Venkatesh

    """

    def _matrix(self):
        """
        Returns the matrix to use to perform
        PCA depending on the shape of the input
        matrix
        :return: The Matrix to be eigendecomposed
        """
        n, d = self._x.shape
        if n < d:
            return np.dot(self._x, self._x.T) / float(n - 1)
        return np.dot(self._x.T, self._x) / float(n - 1)

    def _eigen(self):
        """
        Perform the eigendecomposition and persist ALL the
        eigenvalues and eigenvectors
        """
        n, d = self._x.shape
        # Get and decompose the covariance matrix

        c = self._matrix()
        nc, _ = c.shape
        l, w = np.linalg.eigh(c)
        if nc == n:
            w = np.dot(self._x.T, w)

        # Normalize the eigenvectors
        self._w = w / np.linalg.norm(w, axis=0)

        # Sort the eigenvectors according to the largest
        # eigenvalues
        indices = np.argsort(l)[::-1][:d]
        self._lambdas = l[indices]
        self._w = self._w[:, indices]

    def __init__(self, x):
        """
        Constructs and fits the PCA Model with the given data matrix x
        :param x: The input data matrix
        """
        # Compute the mean and center the matrix
        self._mean = np.mean(x, axis=0)
        self._x = x - self._mean
        # Perform the fit
        self._eigen()

    def mean(self):
        """
        Returns the mean of the input matrix
        :return: The vector containing the mean
        """
        return self._mean

    def eigenvectors(self):
        """
        Returns the eigenvectors of the covariance matrix.
        They are scaled to have norm 1
        :return: The matrix containing the eigenvalues
        """
        return self._w

    def eigenvalues(self):
        """
        Returns the eigenvalues of the covariance matrix
        :return:  The vector of the eigenvalues
        """
        return self._lambdas

    def k_cutoff(self, variance_captured=0.9):
        """
        Returns the number of prinicipal components that have to be used
        in order for the model to capture the given fraction
        of total variance
        :param variance_captured: The fraction of the total variance

        :return: The number of components
        """
        vf = np.cumsum(self._lambdas / np.sum(self._lambdas))
        return bisect_left(vf, variance_captured)

    def project(self, x=None, k=0):
        """
        Use the fitted model to project a set of points
        :param x: The data that has to be projected.
         Defaults to the data used to fit the model.
        :param k: The number of principal components to be used.
        Defaults to all the principal components.
        :return: The matrix of projections of the input data
        """
        _, d = self._x.shape
        if k < 1 or k > d:
            k = d
        if x is not None:
            return np.dot(x - np.mean(x, axis=0), self._w[0:k])
        return np.dot(self._x, self._w[0:k])

    def reconstruct(self, y=None, k=0):
        """
        Reconstructs the input projections by mapping back to the
        original space.
        :param y: The projections produced by the model.
        Defaults to the projections of the data used to fit
        the model
        :param k: The number of components to be used. This
        must be compatible with the input matrix y. Defaults to all
        the principal components
        :return: The matrix of reconstructed points
        """
        _, d = self._x.shape
        if k < 1 or k > d:
            k = d
        if y is None:
            y = self.project(k)
        return np.dot(self._w[0:k], y.T) + self._mean


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
        self._modes = self._model.k_cutoff(pca_variance_captured)
        self._b_max = 3 * ((self._model.eigenvalues()[0:self._modes]) ** 0.5)

    def mean(self):
        """
        Returns the mean
        :return: A vector containing the model mean
        """
        return self._model.mean()

    def modes(self):
        """
        Returns the number of modes of the model
        :return: the number of modes
        """
        return self._modes

    def _p(self):
        return self._model.eigenvectors()[:, 0:self._modes]

    def generate_deviation(self, factors):
        """
        Generates the deviation to be added to the mean based
        on a vector of factors of size equal to the number of
        modes of the model, with element
        values between -1 and 1
        :param factors: A vector of size modes() with values
        between -1 and 1
        :return: A vector containing the generated deviation from mean
        """
        p = self._p()
        pb = np.dot(p, factors * self._b_max)
        return pb

    def fit(self, data):
        """
        Fits the model to a new data point and returns the best factors
        :param data: A data point - array of the size of the mean
        :return: fit error,factors array of the size of the mean
        """
        if data.shape == self.mean().shape:
            bcand = np.squeeze(np.dot(self._p().T, (data - self.mean()))).tolist()
            factors = np.zeros(len(bcand))
            for i in range(len(bcand)):
                val = bcand[i] / self._b_max[i]
                if val > 1:
                    val = 1
                elif val < -1:
                    val = -1
                factors[i] = val
            pred = self.mean() + self.generate_deviation(factors)
            error = math.sqrt(np.sum((data - pred) ** 2))
            return error, factors
        else:
            raise TypeError("Data has to be of the same size as model - Expected " + str(self.mean().shape) + " Got " +
                            str(data.shape))
