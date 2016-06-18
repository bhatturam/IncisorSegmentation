import numpy as np
import math
import cv2


class Shape:
    """
    Represents a shape (an array of points)

    Attributes:
        _data The points stored as a nxd numpy matrix

    Authors: David Torrejon and Bharath Venkatesh
    """

    def __init__(self, data):
        """
        Initializes a shape
        :param data: points as a nxd numpy matrix
        """
        self._data = data

    @classmethod
    def from_homogeneous_coordinates(cls, homogeneous_raw):
        return Shape(np.squeeze(homogeneous_raw[:, 0:2]))

    @classmethod
    def from_collapsed_shape(cls, shape_vector):
        return cls(np.reshape(shape_vector, (len(shape_vector.tolist()) / 2, 2)))

    @classmethod
    def from_list_of_points(cls, points):
        """
        Creates a shape from the given list of lists
        :param points: points as a list of lists
        :return: A Shape object
        """
        return cls(np.array(points))

    @classmethod
    def from_coordinate_lists_2d(cls, x_list, y_list):
        """
        Creates a shape from two same-sized coordinate lists
        :param x_list: The list of x-coordinated
        :param y_list: The list of y-coordinates of
        the same length as x_list
        :return: A Shape object
        """
        if len(x_list) != len(y_list):
            raise AssertionError("The number of x and y coordinates are different")
        points = []
        for i in range(len(x_list)):
            points.append([x_list[i], y_list[i]])
        return Shape.from_list_of_points(points)

    def mean(self):
        """
        Returns the mean point of the shape points
        :return: The d-dimensional mean vector of the shape
        """
        return np.mean(self._data, axis=0)

    def norm(self):
        """
        Returns the norm of the shape points
        :return: The d-dimensional norm of the shape
        """
        return np.linalg.norm(self._data, axis=0)

    def raw(self):
        """
        Returns the data as a numpy matrix
        :return: points as a nxd numpy matrix
        """
        return self._data

    def tolist(self):
        """
        Returns the data as a list of lists
        :return: A list of list of points
        """
        return self._data.tolist()

    def size(self):
        """
        Returns the number of points in the shape
        :return: The number of points in the shape
        """
        r, _ = self._data.shape
        return r

    def center(self):
        """
        Translates the shape such that
        the mean is at the origin
        :return: A Shape object, with mean as
        zero vector
        """
        return Shape(self._data - self.mean())

    def raw_homogeneous(self):
        return np.concatenate((self.raw(), np.ones((self.size(), 1), dtype=float)), axis=1)

    def normalize(self):
        """
        Returns a new shape containing this shape
        scaled to unit norm

        :return: A Shape object, scaled to unit norm
        """
        return Shape(self._data / self.norm())

    def transformation_matrix(self, other):
        """
        Returns the translation,scaling and rotation that
        is used to align the current shape
        to the other shape by
        solving the
        least squares solution of the equation
        Refer the https://en.wikipedia.org/wiki/Kabsch_algorithm
        for more details
        self*a= other
        :param other: The other shape
        :return: translation array, scale array, rotation matrix
        """
        translation = other.mean() - self.mean()
        scale = other.center().norm() / self.center().norm()
        other_data = other.center().normalize().raw()
        this_data = self.center().normalize().raw()
        cov = np.dot(other_data.T, this_data)
        U, _, VT = np.linalg.svd(cov, full_matrices=True, compute_uv=True)
        rotation = np.dot(VT.T, U.T)
        # d = np.sign(np.linalg.det(rotation))
        # correction = np.array([[1, 0], [0, d]], dtype=float)
        # rotation = np.dot(VT.T, np.dot(correction, U.T))
        return translation, scale, rotation

    def homogeneous_transformation_matrix(self, other):
        translation, scaling, rotation = self.transformation_matrix(other)
        return np.array([[scaling[0] * rotation[0, 0], rotation[0, 1], translation[0]],
                         [rotation[1, 0], scaling[1] * rotation[1, 1], translation[1]],
                         [0, 0, 1]])

    def align(self, other):
        """
        Aligns the current shape
        to the other shape  by
        finding a transformation by solving the
        least squares solution of the equation
        self*a= other
        :param other: The other shape
        :return: A shape aligned to other
        """
        translation, scaling, rotation = self.transformation_matrix(other)
        return Shape(np.dot(self._data, rotation)).scale(scaling).translate(translation)

    def collapse(self):
        """
        Collapses the shape into a vector
        :return: A vector of 2*size() points
        """
        n, _ = self._data.shape
        return np.reshape(self._data, 2 * n)

    def concatenate(self, other):
        """
        Returns a new shape combining the points in other and the current shape
        :param other: The  other shape
        :return: A shape containing all the points in the current and other shapes
        """
        return Shape(np.concatenate((self._data, other.raw())))

    def _get_neighborhood(self, point_index, num_neighbors):
        """
        Returns the neighborhood around a given point.
        :param point_index: The index of the query point in the shape
        :param num_neighbors: The number of neighborhood points needed ON EITHER SIDE
        :return: An array of neighborhood points
        """
        if 0 <= point_index < self.size():
            neighborhood_index_increments = range(-num_neighbors, 0, 1) + range(1, num_neighbors + 1, 1)
            neighborhood_indices = [(point_index + incr) % self.size() for incr in neighborhood_index_increments]
            return np.array([self._data[index] for index in neighborhood_indices])
        return np.array([])

    def get_normal_at_point_generator(self, point_index, normal_neighborhood):
        """
        Returns a function that can be used to generate coordinates of the normal at the given point
        :param point_index: The index of the query point in the shape
        :param normal_neighborhood: The number of neighborhood points needed on EITHER SIDE
        :return: A function that accepts a parameter and generates points along the normal based on the input parameter
        """
        neighborhood = self._get_neighborhood(point_index, normal_neighborhood)
        line = cv2.fitLine(np.int32(neighborhood), 2, 0, 0.01, 0.01)
        slope = np.array([-line[1], line[0]])
        unit_slope = np.squeeze(slope / math.sqrt(np.sum(slope ** 2)))
        point = self.get_point(point_index)

        def normal_generator(increment):
            return point + increment * unit_slope

        return normal_generator

    def get_point(self, point_index):
        """
        Returns the point at the given index
        :param point_index:  The index of the query point in the shape
        :return: A 2D array containing the point coordinated
        """
        if 0 <= point_index < self.size():
            return self._data[point_index]
        return np.array([])

    def round(self):
        """
        Returns a new shape with coordinates rounded to the nearest integer
        :return:
        """
        return Shape(np.int32(np.round(self.raw())))

    def pyr_down(self):
        """
        Returns a new shape with the coordinates scaled down by 2
        :return: A shape object
        """
        return Shape(self.raw() / 2)

    def pyr_up(self):
        """
        Returns a new shape with the coordinates scaled up by 2
        :return: A shape object
        """
        return Shape(np.round(self.raw() * 2))

    def scale(self, factor):
        """
        Returns the current shape scaled by the factor
        :param factor: the scaling factor
        :return: A shape object containing the scaled shape
        """
        return Shape(factor * self._data)

    def translate(self, displacement):
        """
        Returns the current shape translated by the specified displacement vector
        :param displacement: The 2d vector of displacements
        :return: A shape object containing the displaced shape
        """
        return Shape(self._data + displacement)

    def rotate_radians(self, theta_in_radians):
        """
        Returns the current shape rotated by theta radians
        :param theta_in_radians: The rotation angle in radians
        :return: A shape object containing the rotated shape
        """
        rotation = np.array(
            [[math.cos(theta_in_radians), -math.sin(theta_in_radians)],
             [math.sin(theta_in_radians), math.cos(theta_in_radians)]])
        return Shape(np.dot(self._data, rotation))

    def rotate(self, rotation):
        """
        Returns the current shape rotated by the rotation matrix
        :param rotation: The rotation matrix
        :return: A shape object containing the rotated shape
        """
        return Shape(np.dot(self._data, rotation))

    def transform(self, transformation_matrix):
        translation = np.array([transformation_matrix[0, 2], transformation_matrix[1, 2]])
        sintheta = transformation_matrix[0, 1]
        costheta = np.sqrt(1 - (sintheta ** 2))
        scaling = np.array([transformation_matrix[0, 0], transformation_matrix[1, 1]]) / costheta
        rotation = np.array(
            [[costheta, -sintheta],
             [sintheta, costheta]])
        return self.rotate(rotation).scale(scaling).translate(translation)

    def to_contour(self):
        clist = []
        for point in self._data.tolist():
            clist.append([point])
        return np.int32(np.array(clist))


class ShapeList:
    """
    A wraper class for a list of Shapes

    Attributes:
        _shapes a numpy matrix of shapes

    Authors: David Torrejon and Bharath Venkatesh
    """

    def __init__(self, shapes):
        self._shapes = shapes

    def __iter__(self):
        return self._shapes.__iter__()

    def __len__(self):
        return len(self._shapes)

    def __getitem__(self, index):
        return self._shapes[index]

    def raw(self):
        """
        Returns the shape list as a number of shapes x number of points x 2 array
        :return: A numpy matrix
        """
        return np.array([shape.raw() for shape in self._shapes])

    def tolist(self):
        """
        Returns the shapes as a list of shape objects
        :return: A list of shape objects
        """
        return self._shapes

    def bounding_box(self):
        raw_data = self.raw()
        min_coords = np.min(np.min(raw_data, axis=0), axis=0)
        max_coords = np.max(np.max(raw_data, axis=0), axis=0)
        return np.array([min_coords, max_coords])

    def mean(self):
        return Shape(np.mean(self.raw(), axis=0))

    def collapse(self):
        """
        Returns the shapes as a numpy matrix.
        :return: a Nxd*d numpy matrix containing the shapes
        """
        return np.array([shape.collapse() for shape in self._shapes])

    def concatenate(self, other):
        """
        Returns a new ShapeList concatenating the object and other
        :param other: The other ShapeList
        :return: A ShapeList
        """
        return ShapeList(self._shapes + other.tolist())

    # def pyr_down(self):
    #     """
    #     Returns a new shape list with all the shapes scaled down
    #     :return: A ShapeList
    #     """
    #     return ShapeList([shape.pyr_down() for shape in self._shapes])
    #
    # def pyr_up(self):
    #     """
    #     Returns a new shape list with all the shapes scaled up
    #     :return: A ShapeList
    #     """
    #     return ShapeList([shape.pyr_up() for shape in self._shapes])
    #
    # def round(self):
    #     """
    #     Returns a new shape list with all the shapes rounded to the nearest integer
    #     :return: A ShapeList
    #     """
    #     return ShapeList([shape.round() for shape in self._shapes])


    def align(self, tol=1e-7, max_iters=10000):
        """
        Performs Generalized Procrustes Analysis to align a list of shapes
        to a common coordinate system. Implementation based on Appendix A of

        Cootes, Tim, E. R. Baldock, and J. Graham.
        "An introduction to active shape models."
        Image processing and analysis (2000): 223-248.
        :param tol: The convergence threshold for gpa
        (Default: 1e-7)
        :param max_iters: The maximum number of iterations
        permitted (Default: 10000)
        :return: A new shape list with aligned shapes
        """
        aligned_shapes = [shape.center().normalize() for shape in self._shapes]
        mean_shape = aligned_shapes[0].normalize()
        for num_iters in range(max_iters):
            for i in range(len(aligned_shapes)):
                aligned_shapes[i] = aligned_shapes[i].align(mean_shape)
            previous_mean_shape = mean_shape
            mean_shape = Shape(
                np.mean(np.array([shape.raw() for shape in aligned_shapes]), axis=0)).center().normalize()
            if np.linalg.norm(mean_shape.raw() - previous_mean_shape.raw()) < tol:
                break
        return ShapeList(aligned_shapes)
