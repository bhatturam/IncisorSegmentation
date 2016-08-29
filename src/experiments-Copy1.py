
# coding: utf-8

# # Imports 

# In[1]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import cv2
from incisorseg.dataset import Dataset,LeaveOneOutSplitter,appearance_model_eight_teeth,appearance_model_four_teeth,load_image,load_landmark,gaussian_pyramid_down,tooth_splitter,tooth_models
from incisorseg.utils import *
from active_shape_models.shape import Shape, ShapeList
from active_shape_models.models import AppearanceModel, GreyModel,PointDistributionModel,ActiveShapeModel
from active_shape_models.pca import PCAModel
import json


# # Reading data

# In[2]:

data = Dataset('../data/')


# # Image preprocessing

# In[3]:

'''
i = 0
for split in LeaveOneOutSplitter(data):
    #training_images,training_landmarks,training_segmentations = split.get_training_set()
    test_image,test_landmark,test_segmentation = split.get_test_example()
    sample = test_image[500:1500,1200:1800]
    transformed_sample = sample
    #transformed_sample = cv2.adaptiveBilateralFilter(transformed_sample,(9,9),5)
    #image_laplacian = cv2.Laplacian(transformed_sample,cv2.CV_32F)
    #image_scaled_laplacian = cv2.convertScaleAbs(image_laplacian,1,0)
    #transformed_sample = cv2.add(transformed_sample,image_scaled_laplacian)
    transformed_sample = cv2.medianBlur(sample,9)
    #canny_upper,_ = cv2.threshold(transformed_sample,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #canny_lower = 0.1 * canny_upper
    #transformed_sample = cv2.bilateralFilter(transformed_sample,5,30,30)
    transformed_sample = cv2.Canny(transformed_sample,2,20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    transformed_sample = cv2.morphologyEx(transformed_sample,cv2.MORPH_CLOSE,kernel)
    #imshow2(image_laplacian)
    imshow2(transformed_sample)
    #imshow2(sample)
    i+=1
    if i >0:
        pass#break
    '''


# # Appearance Model

# In[4]:

'''
pyr_levels = 0
for split in LeaveOneOutSplitter(data):
    training_images,training_landmarks,training_segmentations = split.get_training_set()
    test_image,test_landmark,test_segmentation = split.get_test_example()
    shape,ta,tb = appearance_model_two_teeth(training_images,training_landmarks,test_image,pyr_levels,top_extent=[5,5],top_scale=[1,1],bottom_extent=[5,5],bottom_scale=[1,1])
    plot_shapes([test_landmark,shape.round()])
'''


# # Point Distribution Models - PCA Number of Components

# In[5]:

'''
overall_error_list = []
for num_parts in [1,2,4,8]:
    for split_part in range(num_parts):
        error_list = []
        for nc in range(2,26):
            error = 0
            for split in LeaveOneOutSplitter(data):
                _,all_training_landmarks,_ = split.get_training_set()
                _,all_test_landmark,_ = split.get_test_example()
                split_training_landmarks = tooth_splitter(all_training_landmarks,num_parts)
                split_test_landmark = tooth_splitter([all_test_landmark],num_parts)
                training_landmarks = split_training_landmarks[split_part]
                test_landmark = split_test_landmark[split_part][0]
                model = PointDistributionModel(training_landmarks,project_to_tangent_space=False,
                                               pca_number_of_components=nc)
                _,fit_error,_ = model.fit(test_landmark)
                error += fit_error/test_landmark.get_size()
            error_list.append(error) 
        overall_error_list.append(error_list)
with open('../results/pdm_errors.json', 'w') as outfile:
    json.dump(overall_error_list, outfile)
'''


# In[6]:

'''
with open('../results/pdm_errors.json') as data_file:    
    overall_error_list = json.load(data_file)
for error_list in overall_error_list:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(error_list)
    ax.set_title('Number of Components vs Error')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Error')
'''    


# In[7]:

'''
for split in LeaveOneOutSplitter(data):
    _,all_training_landmarks,_ = split.get_training_set()
    _,all_test_landmark,_ = split.get_test_example()
    model_list, training_landmarks_list, test_landmark_list, error_list = tooth_models(all_training_landmarks,all_test_landmark,pca_variance_captured=[0.94,0.96,0.98,0.99])
    print error_list
    #print model_list
'''


# # Ideal case - testing the shape model fitting

# In[8]:

'''
errors = []
for split in LeaveOneOutSplitter(data):
    training_images,training_landmarks,training_segmentations = split.get_training_set()
    test_image,test_landmark,test_segmentation = split.get_test_example()
    shape_model = PointDistributionModel(training_landmarks,pca_variance_captured=0.98)
    fitted_shape,_,num_iters =shape_model.fit(test_landmark)
    errors.append(split.get_dice_error_on_test(fitted_shape))
print np.mean(np.array(errors))
'''


# # Ideal case - testing the grey model fitting

# In[9]:

'''
pyr_levels=4
for split in LeaveOneOutSplitter(data):
    training_images,training_landmarks,training_segmentations = split.get_training_set()
    test_image,test_landmark,test_segmentation = split.get_test_example()
    scaled_training_images,scaled_training_landmarks = gaussian_pyramid_down(
        training_images,num_levels=pyr_levels, training_landmarks=training_landmarks)
    scaled_test_image,scaled_test_landmark = gaussian_pyramid_down([test_image],training_landmarks=[test_landmark],num_levels=pyr_levels)
    scaled_test_image = scaled_test_image[0]
    scaled_test_landmark = scaled_test_landmark[0]
    all_teeth_shape_model = PointDistributionModel(scaled_training_landmarks,pca_variance_captured=0.99)
    all_teeth_app_model = AppearanceModel(scaled_training_images,all_teeth_shape_model,[6,6])
    fitted_shape_appearance = all_teeth_app_model.fit(transformed_test_image)
    
    upper_teeth_shape_model = PointDistributionModel(upper_teeth_training_landmarks,pca_variance_captured=0.9)
    lower_teeth_shape_model = PointDistributionModel(lower_teeth_training_landmarks,pca_variance_captured=0.9)
    transformed_test_image = scaled_test_image
    transformed_training_images = scaled_training_images
    #transformed_training_images = [cv2.Laplacian(image,ddepth=cv2.CV_32F) for image in scaled_training_images]
    #transformed_test_image = cv2.Laplacian(scaled_test_image,ddepth=cv2.CV_32F)
    transformed_training_images = [cv2.medianBlur(image,3) for image in scaled_training_images]
    transformed_test_image = cv2.medianBlur(scaled_test_image,3)
    grey_model = GreyModel(transformed_training_images, upper_teeth_training_landmarks,patch_num_pixels=5*(5-pyr_levels), 
                           search_num_pixels=40*(5-pyr_levels), use_gradient=True,
                 normalize_patch=True, use_moded_pca_model=False, mpca_variance_captured=0.9,
                 normal_point_neighborhood=1)
    model=ActiveShapeModel(grey_model=grey_model,point_distribution_model=upper_teeth_shape_model,appearance_model=None)
    new_shape,fit_error,num_iters = model.fit(scaled_test_image,0.2,100, initial_shape=upper_teeth_initial_position)
    print num_iters
    plot_shapes([upper_teeth_test_landmark,new_shape])
    #fitted_shape_grey,_ =grey_model.search(transformed_test_image,upper_teeth_initial_position.round())
    #fitted_shape,_,_=upper_teeth_shape_model.fit(fitted_shape_grey)
    #fitted_shape_grey,_ =grey_model.search(transformed_test_image,fitted_shape.round())
    #fitted_shape,_,_=upper_teeth_shape_model.fit(fitted_shape_grey)
    #plot_shapes([upper_teeth_test_landmark,fitted_shape_grey,fitted_shape])
    #for j in range(0,pyr_levels):
    #    fitted_shape = fitted_shape.pyr_up()
    #error_list.append(split.get_dice_error_on_test(fitted_shape))

#plot_shapes([test_landmark,fitted_shape])
print np.mean(np.array(error_list))
'''


# In[10]:

k = 0
for split in LeaveOneOutSplitter(data):
    training_images, all_training_landmarks,_ = split.get_training_set()
    test_image, all_test_landmark,_ = split.get_test_example()
    model_list, training_landmarks_list, test_landmark_list, error_list = tooth_models(all_training_landmarks,all_test_landmark,
                                                                                      project_to_tangent_space=True,pca_variance_captured=[0.94,0.96,0.98,0.99])
    transformed_train_imgs = training_images
    transformed_test_imgs = test_image
    transformed_test_imgs = cv2.medianBlur(test_image,9)
    transformed_train_imgs = [cv2.medianBlur(sample,9) for sample in training_images]
    initial_shape = appearance_model_eight_teeth(training_images,all_training_landmarks,test_image,pyr_levels=0,top_extent=[6,6])
    initial_shape_2 = appearance_model_four_teeth(training_images,all_training_landmarks,test_image,pyr_levels=0,top_extent=[7,7],top_scale=[1.1,1.1],bottom_extent=[6,6],bottom_scale=[1.05,1.1])
    initial_error = split.get_dice_error_on_test(initial_shape)
    initial_error_2 = split.get_dice_error_on_test(initial_shape_2)
    initial_error_3 = split.get_dice_error_on_test(all_test_landmark)
    print 0,initial_error,initial_error_2,initial_error_3
    new_shape = initial_shape
    new_shape_2 = initial_shape_2
    new_shape_3 = all_test_landmark
    plot_shapes([all_test_landmark,new_shape.round(),new_shape_2.round(),new_shape_3.round()])
    search_num_pixels = [50,50,50,50]
    for i in range(1,len(model_list)):
        init_shapes = ShapeList.from_shape(new_shape, len(model_list[i])) #splits shapes
        init_shapes_2 = ShapeList.from_shape(new_shape_2, len(model_list[i])) #splits shapes
        init_shapes_3 = ShapeList.from_shape(new_shape_3, len(model_list[i])) #splits shapes
        new_shape = None
        new_shape_2 = None
        new_shape_3 = None
        for j in range(len(model_list[i])):
            grey_model = GreyModel(transformed_train_imgs,training_landmarks_list[i][j],
                               patch_num_pixels=30,
                               search_num_pixels=search_num_pixels[i],
                               use_gradient=True,normalize_patch = True)
            model = ActiveShapeModel(grey_model=grey_model,point_distribution_model=model_list[i][j],appearance_model=None)
            new_shape_part,fit_error,num_iters = model.fit(transformed_test_imgs,0.3,1000, initial_shape=init_shapes[j])
            new_shape_part_2,fit_error,num_iters_2 = model.fit(transformed_test_imgs,0.3,1000, initial_shape=init_shapes_2[j])
            new_shape_part_3,fit_error,num_iters_3 = model.fit(transformed_test_imgs,0.3,1000, initial_shape=init_shapes_3[j])
            print "iters: ", num_iters, num_iters_2,num_iters_3,"i:", i
            if new_shape is None:
                new_shape = new_shape_part
            else:
                new_shape=new_shape.concatenate(new_shape_part)
            if new_shape_2 is None:
                new_shape_2 = new_shape_part_2
            else:
                new_shape_2=new_shape_2.concatenate(new_shape_part_2)
            if new_shape_3 is None:
                new_shape_3 = new_shape_part_3
            else:
                new_shape_3=new_shape_3.concatenate(new_shape_part_3)
        print i, split.get_dice_error_on_test(new_shape),split.get_dice_error_on_test(new_shape_2),split.get_dice_error_on_test(new_shape_3)
        plot_shapes([all_test_landmark,new_shape.round(),new_shape_2.round(),new_shape_3.round()])
    k+=1
    if k>=3:
        pass#break


# In[11]:

'''
i = 0
pyr_levels = 4
for split in LeaveOneOutSplitter(data):
    if i >=1:
        pass#break
    training_images,training_landmarks,training_segmentations = split.get_training_set()
    test_image,test_landmark,test_segmentation = split.get_test_example()
    scaled_training_images = []
    scaled_training_landmarks = []
    scaled_test_images = []
    scaled_test_landmarks = []
    scaled_training_images.append(training_images)
    scaled_training_landmarks.append(ShapeList([ShapeList.from_shape(landmark,1)[0] for landmark in training_landmarks]))
    scaled_test_images.append(test_image)
    scaled_test_landmarks.append(ShapeList.from_shape(test_landmark,1)[0])
    for j in range(1,pyr_levels):
        scaled_test_images.append(cv2.pyrDown(scaled_test_images[j-1]))
        scaled_test_landmarks.append(scaled_test_landmarks[j-1].pyr_down())
        scaled_training_images.append([cv2.pyrDown(image) for image in scaled_training_images[j-1]])
        scaled_training_landmarks.append(ShapeList([shape.pyr_down() for shape in scaled_training_landmarks[j-1]]))
    new_shape = None
    #print 'Test Landmarks Accuracy %f',split.get_dice_error_on_test(test_landmark)
    detected_shapes = []
    for j in range(pyr_levels-1,-1,-1):
        shape_model = PointDistributionModel(scaled_training_landmarks[j],pca_variance_captured=0.9)
        grey_model = GreyModel(scaled_training_images[j],scaled_training_landmarks[j],
                               patch_num_pixels=5*(pyr_levels-j),
                               search_num_pixels=7*(pyr_levels-j))
        app_model = AppearanceModel(scaled_training_images[j],shape_model,[6,6])
        model = ActiveShapeModel(grey_model=grey_model,point_distribution_model=shape_model,appearance_model=app_model)
        new_shape,fit_error,num_iters = model.fit(scaled_test_images[j],0.5,100, initial_shape=new_shape)
        if j == pyr_levels-1:     
            detected_shapes.append(model.get_default_initial_shape())
        detected_shapes.append(new_shape)
        if j == 0:
            imshow2(overlay_shapes_on_image(scaled_test_images[j],[new_shape.round()]))
        #plot_shapes([new_shape])
        new_shape = new_shape.pyr_up()
    i +=1
    for j in range(len(detected_shapes)):
        if j==0:
            scale_start = 1
        else:
            scale_start = j
        for k in range(scale_start,pyr_levels):
            detected_shapes[j]=detected_shapes[j].pyr_up()
        
        #print 'Level:' , j, 'Accuracy',split.get_dice_error_on_test(detected_shapes[j]) 
'''
