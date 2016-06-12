import os
import numpy as np
import cv2
from asm.shape import Shape, ShapeList


def load_landmark(filepath, mirrored=False, width=0):
    """
    Creates a shape from a file containing 2D points
    in the following format
        x1
        y1
        x2
        y2
        ...
        xn
        yn
    :param filepath: The path to the landmark file
    :param mirrored: True when reading a vertically mirrored landmark
    :param width: The image width, needed when reading a mirrored landmark
    :return: A Shape object
    """
    y_list = []
    x_list = []
    if mirrored and width == 0:
        raise ValueError("Need a nonzero width for a mirrored landmark")
    with open(filepath) as fd:
        for i, line in enumerate(fd):
            if i % 2 == 0:
                x_list.append(float(line) + width)
            else:
                y_list.append(float(line))
    return Shape.from_coordinate_lists_2d(x_list, y_list)


def load_image(filepath):
    """
    Creates a 2D Array from a file containing the
    image/segmentation
    :param filepath:  The path to the image/segmentation
    :return: A 2D Numpy Array
    """
    return cv2.imread(filepath, 0)


def parse_segmentation(img):
    """
    Creates a list of pixels from a 2D Array containing
    the segmentation
    :param img: The img as a 2D Array
    :return: A list of coordinates indices for non zero pixels
    """
    img2 = np.uint8(img.copy())
    img2[img2 > 0] = 1
    return img2


# def parse_segmentation(img):
#     """
#     Creates a list of pixels from a 2D Array containing
#     the segmentation
#     :param img: The img as a 2D Array
#     :return: A list of coordinates indices for non zero pixels
#     """
#     return np.array(zip(*np.where(img > 0))).tolist()


class Dataset:
    """
        Class to represent the data for the assignment
    """
    _training_image_count = 14
    _test_image_count = 16
    _tooth_count = 8
    TOP_TEETH = range(4, 8)
    BOTTOM_TEETH = range(4)
    ALL_TEETH = range(_tooth_count)
    ALL_TRAINING_IMAGES = range(_training_image_count)

    def _build_landmark_filepath(self, image_index, tooth_index, mirrored=False):
        landmarks_filepath_prefix = os.path.join(self._data_folder, 'Landmarks')
        if mirrored:
            landmarks_filepath_prefix = os.path.join(landmarks_filepath_prefix, 'mirrored')
            image_index += self._training_image_count
        else:
            landmarks_filepath_prefix = os.path.join(landmarks_filepath_prefix, 'original')
        return os.path.join(landmarks_filepath_prefix,
                            'landmarks' + str(image_index + 1) + '-' + str(tooth_index + 1) + '.txt')

    def _build_segmentation_filepath(self, image_index, tooth_index):
        segmentations_filepath_prefix = os.path.join(self._data_folder, 'Segmentations')
        return os.path.join(segmentations_filepath_prefix,
                            str(image_index + 1).zfill(2) + '-' + str(tooth_index) + '.png')

    def _build_image_filepath(self, image_index):
        radiograph_filepath_prefix = os.path.join(self._data_folder, 'Radiographs')
        return os.path.join(radiograph_filepath_prefix, str(image_index + 1).zfill(2) + '.tif')

    def _build_extra_image_filepath(self, image_index):
        radiograph_filepath_prefix = os.path.join(self._data_folder, os.path.join('Radiographs', 'extra'))
        return os.path.join(radiograph_filepath_prefix, str(image_index + 1).zfill(2) + '.tif')

    def _process_tooth_landmarks(self, image_index, tooth_index, width):
        original_landmark = load_landmark(self._build_landmark_filepath(image_index, tooth_index))
        mirrored_landmark = load_landmark(self._build_landmark_filepath(image_index, tooth_index, True), True, width)
        return original_landmark, mirrored_landmark

    def _process_tooth_segmentations(self, image_index, tooth_index):
        segmentation_img = load_image(self._build_segmentation_filepath(image_index, tooth_index))
        original_segmentation = parse_segmentation(segmentation_img)
        mirrored_segmentation = parse_segmentation(cv2.flip(segmentation_img, 1))
        return original_segmentation, mirrored_segmentation

    def _process_radiograph(self, image_index):
        original_image = load_image(self._build_image_filepath(image_index))
        _, width = original_image.shape
        mirrored_image = cv2.flip(original_image, 1)
        return original_image, mirrored_image, width

    def _read_extra_images(self):
        self._extra_images = []
        for image_index in range(self._test_image_count):
            self._extra_images.append(
                load_image(self._build_extra_image_filepath(image_index + self._training_image_count)))

    def _read_training_data(self):
        for image_index in range(self._training_image_count):
            original_image, mirrored_image, width = self._process_radiograph(image_index)
            self._training_images.append(original_image)
            self._training_images_mirrored.append(mirrored_image)
            landmarks = []
            segmentations = []
            landmarks_mirrored = []
            segmentations_mirrored = []
            for tooth_index in range(self._tooth_count):
                original_landmark, mirrored_landmark = self._process_tooth_landmarks(image_index, tooth_index, width)
                landmarks.append(original_landmark)
                landmarks_mirrored.append(mirrored_landmark)
                original_segmentation, mirrored_segmentation = self._process_tooth_segmentations(image_index,
                                                                                                 tooth_index)
                segmentations.append(original_segmentation)
                segmentations_mirrored.append(mirrored_segmentation)
            self._training_landmarks.append(landmarks)
            self._training_landmarks_mirrored.append(landmarks_mirrored)
            self._training_segmentations.append(segmentations)
            self._training_segmentations_mirrored.append(segmentations_mirrored)

    def get_training_images(self, image_indices):
        images = []
        mirrored_images = []
        for image_index in image_indices:
            mirrored_images.append(self._training_images_mirrored[image_index])
            images.append(self._training_images[image_index])
        return images, mirrored_images

    def get_extra_images(self, image_indices):
        images = []
        for image_index in image_indices:
            images.append(self._extra_images[image_index])
        return images

    def get_training_image_segmentations(self, image_indices, tooth_indices):
        segmentations = []
        mirrored_segmentations = []
        for image_index in image_indices:
            combined_segmentation = np.uint8(np.zeros(self._training_segmentations[image_index][0].shape))
            combined_segmentation_mirrored = np.uint8(np.zeros(self._training_segmentations[image_index][0].shape))
            for tooth_index in tooth_indices:
                combined_segmentation = np.bitwise_or(combined_segmentation,
                                                      self._training_segmentations[image_index][tooth_index])
                combined_segmentation_mirrored = np.bitwise_or(combined_segmentation_mirrored,
                                                               self._training_segmentations_mirrored[image_index][
                                                                   tooth_index])
            segmentations.append(combined_segmentation)
            mirrored_segmentations.append(combined_segmentation_mirrored)
        return segmentations, mirrored_segmentations

    # def get_training_image_segmentations(self, image_indices, tooth_indices, combine=False):
    #     segmentations = []
    #     mirrored_segmentations = []
    #     for image_index in image_indices:
    #         image_segmentations = []
    #         image_segmentations_mirrored = []
    #         final_segmentation = None
    #         final_segmentation_mirrored = None
    #         for tooth_index in tooth_indices:
    #             segmentation = self._training_segmentations[image_index][tooth_index]
    #             mirrored_segmentation = self._training_segmentations_mirrored[image_index][tooth_index]
    #             if not combine:
    #                 image_segmentations.append(segmentation)
    #                 image_segmentations_mirrored.append(mirrored_segmentation)
    #             elif final_segmentation is None:
    #                 final_segmentation = segmentation
    #                 final_segmentation_mirrored = mirrored_segmentation
    #             else:
    #                 final_segmentation = np.concatenate((final_segmentation, segmentation))
    #                 final_segmentation_mirrored = np.concatenate((final_segmentation_mirrored, mirrored_segmentation))
    #         if not combine:
    #             segmentations.append(image_segmentations)
    #             mirrored_segmentations.append(image_segmentations_mirrored)
    #         else:
    #             segmentations.append(final_segmentation)
    #             mirrored_segmentations.append(final_segmentation_mirrored)
    #     return segmentations, mirrored_segmentations

    # def get_training_image_landmarks(self, image_indices, tooth_indices, combine=False):
    #     """
    #     This returns the landmarks for the given image and teeth indices
    #     :param image_indices: A list containing the image indices for which the landmarks must be fetched
    #     :param tooth_indices: A list containing the tooth indices for which the landmarks must be fetched
    #         e.g TOP_TEETH is [4,5,6,7], BOTTOM_TEETH is [0,1,2,3]
    #     :param combine: True if the shapes must be combined - i,e a single landmark per image for all tooth_indices
    #                     If False, there are len(tooth_indices) landmarks returned per image
    #     :return: Two ShapeLists - containing the mirrored and unmirrored landmarks
    #     """
    #     landmarks = []
    #     mirrored_landmarks = []
    #     for image_index in image_indices:
    #         image_landmarks = []
    #         image_landmarks_mirrored = []
    #         final_landmark = None
    #         final_landmark_mirrored = None
    #         for tooth_index in tooth_indices:
    #             landmark = self._training_landmarks[image_index][tooth_index]
    #             mirrored_landmark = self._training_landmarks_mirrored[image_index][tooth_index]
    #             if not combine:
    #                 image_landmarks.append(landmark)
    #                 image_landmarks_mirrored.append(mirrored_landmark)
    #             elif final_landmark is None:
    #                 final_landmark = landmark
    #                 final_landmark_mirrored = mirrored_landmark
    #             else:
    #                 final_landmark = final_landmark.concatenate(landmark)
    #                 final_landmark_mirrored = final_landmark_mirrored.concatenate(mirrored_landmark)
    #         if not combine:
    #             landmarks.append(image_landmarks)
    #             mirrored_landmarks.append(image_landmarks_mirrored)
    #         else:
    #             landmarks.append(final_landmark)
    #             mirrored_landmarks.append(final_landmark_mirrored)
    #     return landmarks, mirrored_landmarks

    def get_training_image_landmarks(self, image_indices, tooth_indices):
        """
        This returns the landmarks for the given image and teeth indices
        :param image_indices: A list containing the image indices for which the landmarks must be fetched
        :param tooth_indices: A list containing the tooth indices for which the landmarks must be fetched
            e.g TOP_TEETH is [4,5,6,7], BOTTOM_TEETH is [0,1,2,3]
        :return: Two ShapeLists - containing the mirrored and unmirrored landmarks
        """
        landmarks = []
        mirrored_landmarks = []
        for image_index in image_indices:
            final_landmark = None
            final_landmark_mirrored = None
            for tooth_index in tooth_indices:
                landmark = self._training_landmarks[image_index][tooth_index]
                mirrored_landmark = self._training_landmarks_mirrored[image_index][tooth_index]
                if final_landmark is None:
                    final_landmark = landmark
                    final_landmark_mirrored = mirrored_landmark
                else:
                    final_landmark = final_landmark.concatenate(landmark)
                    final_landmark_mirrored = final_landmark_mirrored.concatenate(mirrored_landmark)
            landmarks.append(final_landmark)
            mirrored_landmarks.append(final_landmark_mirrored)
        return ShapeList(landmarks), ShapeList(mirrored_landmarks)

    def __init__(self, data_folder):
        self._training_images = []
        self._training_images_mirrored = []
        self._training_landmarks = []
        self._training_landmarks_mirrored = []
        self._training_segmentations = []
        self._training_segmentations_mirrored = []
        self._data_folder = data_folder
        self._read_training_data()
        self._read_extra_images()


class LeaveOneOutSplitter:
    def __init__(self, data, images_indices=Dataset.ALL_TRAINING_IMAGES, shapes_indices=Dataset.ALL_TEETH):
        img, mimg = data.get_training_images(images_indices)
        l, ml = data.get_training_image_landmarks(images_indices, shapes_indices)
        s, ms = data.get_training_image_segmentations(images_indices, shapes_indices)
        images = img + mimg
        shapes = l.concatenate(ml)
        segmentations = s + ms
        self._images = images
        self._shapes = shapes
        self._segmentations = segmentations
        self._test_idx = -1
        self._training_idx = []

    def get_training_set_size(self):
        return len(self._training_idx)

    def get_test_index(self):
        return self._test_idx

    def get_training_set(self):
        return [self._images[idx] for idx in self._training_idx], ShapeList([self._shapes[idx] for idx in self._training_idx]), [
            self._segmentations[idx] for idx in self._training_idx]

    def get_test_example(self):
        return self._images[self._test_idx], self._shapes[self._test_idx], self._segmentations[self._test_idx]

    def get_dice_error_on_test(self, shape, use_landmark=False):
        bin_truth = self._segmentations[self._test_idx]
        if use_landmark:
            bin_truth = np.uint8(np.zeros(self._images[self._test_idx].shape))
            cv2.drawContours(bin_truth, [self._shapes[self._test_idx].to_contour()], -1, (255, 255, 255), -1)
            bin_truth[bin_truth > 0] = 1
        bin_predicted = np.int8(np.zeros(bin_truth.shape))
        cv2.drawContours(bin_predicted, [shape.to_contour()], -1, (255, 255, 255), -1)
        bin_predicted[bin_predicted > 0] = 1
        intersection = float(np.sum(np.sum(np.bitwise_and(bin_truth, bin_predicted))))
        union = float(np.sum(np.sum(np.bitwise_or(bin_truth, bin_predicted))))
        return intersection / union

    def __iter__(self):
        return self

    def next(self):
        if self._test_idx > len(self._images) - 2:
            raise StopIteration
        else:
            self._test_idx += 1
            self._training_idx = range(0, self._test_idx) + range(self._test_idx + 1, len(self._images))
            return self
