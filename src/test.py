import numpy as np
import cv2
from incisorseg.dataset import Dataset,LeaveOneOutSplitter,appearance_model_eight_teeth,appearance_model_four_teeth,load_image,load_landmark,gaussian_pyramid_down,tooth_splitter,tooth_models,gaussian_pyramid_up
from incisorseg.utils import *
from active_shape_models.pca import PCAModel
from active_shape_models.models import GreyModel,PointDistributionModel,ActiveShapeModel
from active_shape_models.shape import Shape, ShapeList,LineGenerator
import json
data = Dataset('../data/')

def roi_demo(test_image,test_landmark):
    roi_y1 = 600    
    roi_y2 = 1400
    roi_x1 = 1200    
    roi_x2 = 1800
    roi2 = test_image[roi_y1:roi_y2,roi_x1:roi_x2] 
    roi2 = cv2.medianBlur(roi2,11)
    roi_landmark = test_landmark.translate([-roi_x1,-roi_y1])
    roi_landmark = roi_landmark.get_convex_hull()
    return overlay_points_on_image(roi2,roi_landmark.as_list_of_points())
    

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def dummy(param):
    return param

def scaled_smoothed_profile(valarray):
    ma_order = 5
    minval = np.min(valarray)
    maxval = np.max(valarray)
    rangeval = maxval-minval
    return moving_average((valarray-minval)/rangeval,ma_order)

def median_filtered_image(img):
    ksize = 11
    return cv2.medianBlur(img,ksize)
    

def extract(img,test_landmark,p_h,img_trans=dummy,vec_trans = dummy):
    nn = 4
    imgt = img_trans(img)
    all_points = []
    for point_index in range(test_landmark.get_size()):
        temp = []
        normal_points = []        
        point = test_landmark.get_point(point_index)
        tangent_slope_vector, normal_slope_vector = test_landmark.get_slope_vectors_at_point(point_index,nn)
        normal_coordinates_generator = LineGenerator(point, normal_slope_vector)
        normal_coordinate_list = normal_coordinates_generator.generate_two_sided(p_h)
        for coordinates in normal_coordinate_list:
            tangent_coordinates_generator = LineGenerator(coordinates, tangent_slope_vector)
            tangent_coordinate_list=tangent_coordinates_generator.generate_two_sided(0)
            normal_points = normal_points + tangent_coordinate_list
        for point in normal_points:
            temp.append(imgt[point[1], point[0]])            
        all_points.append(vec_trans(temp))
    return all_points
        

p_h = 20
s_h = 6*p_h
curves =  []
scurves = []
for index,split in enumerate(LeaveOneOutSplitter(data,Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH)):
    if index > 0:
        pass#break
    training_images,training_landmarks,training_segmentations = split.get_training_set()
    test_image,test_landmark,big_test_segmentation = split.get_test_example()
    curves.append(extract(test_image,test_landmark,p_h))
    scurves.append(extract(test_image,test_landmark,s_h))
pdata = np.swapaxes(np.array(curves),0,1)
sdata = np.swapaxes(np.array(scurves),0,1)
lsize,isize,psize =  pdata.shape
lsize,isize,ssize = sdata.shape
print lsize,isize,psize,ssize
for i in range(lsize):
        p = np.squeeze(pdata[i,:,:]).tolist()
        s= np.squeeze(sdata[i,:,:]).tolist()
        #model = PCAModel(data)
        #print np.sum(model.get_eigenvalues())
        #print model.get_varfrac()[0:5]
        plot_many_lines(p)
        plot_many_lines(s)