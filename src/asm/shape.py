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

    def normalize(self):
        """
        Returns a new shape containing this shape
        scaled to unit norm

        :return: A Shape object, scaled to unit norm
        """
        return Shape(self._data / self.norm())

    def align(self, other):
        """
        Aligns the current shape (HAS TO BE CENTERED)
        to the other shape (HAS TO BE CENTERED AS WELL) by
        finding a transformation matrix  r by solving the
        least squares solution of the equation
        self*r = other
        :param other: The other shape
        :return: A shape aligned to other
        """
        other_data = other.raw()
        cov = np.dot(other_data.T, self._data)
        btb = np.dot(other_data.T, other_data)
        pic = np.linalg.pinv(cov)
        a = np.dot(pic, btb)
        s=math.sqrt(np.mean(np.sum(a**2,axis=0)))
        r = a/s
        return Shape(np.dot(self._data, a)),r

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
        slope = np.squeeze(np.array([-line[1], line[0]]) / math.sqrt(np.sum(line[0:2] ** 2)))
        point = self.get_point(point_index)

        def normal_generator(increment):
            return point + increment * slope

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

    def pyr_down(self):
        """
        Returns a new shape with the coordinates scaled down by 2
        :return: A shape object
        """
        return Shape(np.round(self.raw() / 2))

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

    def rotate(self, theta_in_radians):
        """
        Returns the current shape rotated by theta radians
        :param theta_in_radians: The rotation angle in radians
        :return: A shape object containing the rotated shape
        """
        rotation = np.array(
            [[math.cos(theta_in_radians), -math.sin(theta_in_radians)],
             [math.sin(theta_in_radians), math.cos(theta_in_radians)]])
        return Shape(np.dot(self._data, rotation))

    def to_contour(self):
        clist = []
        for point in self._data.tolist():
            clist.append([point])
        return np.int32(np.array(clist))


class AlignedShapeList:
    """
    A list of Aligned Shapes

    Attributes:
        _aligned_shapes a list of shapes aligned by generalized procrustes analysis
        _mean_shape the mean shape of the shapes

    Authors: David Torrejon and Bharath Venkatesh
    """

    def __init__(self, shapes, tol=1e-7, max_iters=10000):
        """
        Performs Generalized Procrustes Analysis to align a list of shapes
        to a common coordinate system. Implementation based on Appendix A of

        Cootes, Tim, E. R. Baldock, and J. Graham.
        "An introduction to active shape models."
        Image processing and analysis (2000): 223-248.

        :param shapes: A list of Shape objects
        :param tol: The convergence threshold for gpa
        (Default: 1e-7)
        :param max_iters: The maximum number of iterations
        permitted (Default: 10000)
        :return: centered_shapes The centered list of shapes
                 mean_shape The mean shape of the given list
        """
        self._aligned_shapes = [shape.center() for shape in shapes]
        self._mean_shape = self._aligned_shapes[0].normalize()
        for num_iters in range(max_iters):
            rotation_list = []
            for i in range(len(self._aligned_shapes)):
                self._aligned_shapes[i],r = self._aligned_shapes[i].align(self._mean_shape)
                rotation_list.append(r)
            self._mean_rotation = np.mean(np.array(rotation_list),axis=0)
            previous_mean_shape = self._mean_shape
            self._mean_shape = Shape(
                np.mean(np.array([shape.raw() for shape in self._aligned_shapes]), axis=0)).center().normalize()
            if np.linalg.norm(self._mean_shape.raw() - previous_mean_shape.raw()) < tol:
                break

    def mean_rotation(self):
        return self._mean_rotation

    def mean_shape(self):
        """
        Returns the mean shape
        :return: A shape object containing the mean shape
        """
        return self._mean_shape

    def shapes(self):
        return self._aligned_shapes

    def raw(self):
        """
        Returns the shapes as a numpy matrix.
        :return: a Nxd*d numpy matrix containing the shapes
        """
        return np.array([shape.collapse() for shape in self._aligned_shapes])


def get_bounding_box(shape_list):
    aligned_shapes = np.array([shape.raw() for shape in shape_list])
    align_min = np.min(np.min(aligned_shapes, axis=0),axis=0)
    align_max = np.max(np.max(aligned_shapes, axis=0),axis=0)
    return np.array([align_min,align_max])
