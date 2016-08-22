import numpy as np
import cv2
from incisorseg.dataset import Dataset,LeaveOneOutSplitter,appearance_model_eight_teeth,appearance_model_four_teeth,load_image,load_landmark,gaussian_pyramid_down,tooth_splitter,tooth_models,gaussian_pyramid_up
from incisorseg.utils import *
from active_shape_models.models import GreyModel,PointDistributionModel,ActiveShapeModel
from active_shape_models.shape import Shape, ShapeList,LineGenerator
import json
data = Dataset('../data/')
roi_y1 = 600    
roi_y2 = 1400
roi_x1 = 1200    
roi_x2 = 1800
p_h = 20
p_w = 0
lsize = 320
isize = 15
nn = 4
ma_order = 5
bsize = p_h*2+1
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
curves = np.zeros((lsize,isize,bsize))
for index,split in enumerate(LeaveOneOutSplitter(data,Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH)):
    if index > 0:
        pass#break
    training_images,training_landmarks,training_segmentations = split.get_training_set()
    pdm = PointDistributionModel(training_landmarks)
    test_image,test_landmark,big_test_segmentation = split.get_test_example()

    roi2 = test_image[roi_y1:roi_y2,roi_x1:roi_x2] 
    roi2 = cv2.medianBlur(roi2,11)
    
    roi_landmark = test_landmark.translate([-roi_x1,-roi_y1])
    #roi_landmark = roi_landmark.get_convex_hull()
    #img2 = overlay_points_on_image(roi2,roi_landmark.as_list_of_points())
        
    for point_index in range(test_landmark.get_size()):
        normal_points = []        
        point = roi_landmark.get_point(point_index)
        tangent_slope_vector, normal_slope_vector = test_landmark.get_slope_vectors_at_point(point_index,nn)
        normal_coordinates_generator = LineGenerator(point, normal_slope_vector)
        normal_coordinate_list = normal_coordinates_generator.generate_two_sided(p_h)
        for coordinates in normal_coordinate_list:
            tangent_coordinates_generator = LineGenerator(coordinates, tangent_slope_vector)
            tangent_coordinate_list=tangent_coordinates_generator.generate_two_sided(p_w)
            normal_points = normal_points + tangent_coordinate_list
        for index2,point in enumerate(normal_points):
            val = roi2[point[1], point[0]]            
            curves[point_index,index,index2] = val
#curves = curves.tolist()
#print curves
for i in range(lsize):
    for j in range(isize):
        valarray = np.squeeze(curves[i,j,:])
        valarray = (valarray-np.mean(valarray))/np.std(valarray)
        mavg = moving_average(valarray,ma_order)
        valarray[0:len(mavg)] = mavg
        curves[i,j,:] = valarray
    plot_many_lines(np.squeeze(curves[i,:,:]).tolist())
#print curves    