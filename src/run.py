# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 23:11:34 2016

@author: bharath
"""
import cPickle as pickle
import numpy as np
import cv2
from incisorseg.dataset import *
from incisorseg.utils import *
from active_shape_models.shape import *
#from active_shape_models.pca import PCAModel
from active_shape_models.imgproc import extract_patch_normal, image_transformation_default, patch_transformation_default
from active_shape_models.models import PointDistributionModel,GreyModel,ActiveShapeModel,AppearanceModel
#from sklearn.decomposition import PCA
#from scipy.stats import pearsonr)

    
data = Dataset('../data/')

def get_grey_models(data,modelfileprefix,train_width):
    fname = modelfileprefix + '_default_'+str(train_width)+'.dat'
    if os.path.isfile(fname):
        f = open(fname)
        gms = pickle.load(f)
        f.close()
        return gms
    else:
        gms = []
        for index,split in enumerate(LeaveOneOutSplitter(data,Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH)):
            training_images,training_landmarks,_ = split.get_training_set()
            test_image,test_landmark,_ = split.get_test_example()
            gms.append(GreyModel(training_images,training_landmarks,train_width,50,image_transformation_function=image_transformation_default,patch_transformation_function=patch_transformation_default))
        f = open(fname, 'w')
        pickle.dump(gms,f)
        f.close()
        return gms


gms = get_grey_models(data,'picklegms',20)

#class GreyModel:
#    """ A grey level point model based on
#    Cootes, Timothy F., and Christopher J. Taylor.
#     "Active Shape Model Search using Local Grey-Level Models:
#     A Quantitative Evaluation." BMVC. Vol. 93. 1993.
#     and
#     An Active Shape Model based on
#        Cootes, Tim, E. R. Baldock, and J. Graham.
#        "An introduction to active shape models."
#        Image processing and analysis (2000): 223-248.
#
#        Attributes:
#            _point_models: The list of underlying point grey models (GaussianModel or ModedPCAModel)
#
#        Authors: David Torrejon and Bharath Venkatesh
#
#    """
#
#    def __init__(self, training_images, training_shape_list, patch_num_pixels_length,search_num_pixels,image_transformation_function,patch_transformation_function):
#        self._search_num_pixels = search_num_pixels
#        self._patch_num_pixels_length = patch_num_pixels_length
#        self._patch_num_pixels_width = 0
#        self._patch_trans_func=patch_transformation_function
#        self._img_trans_func=image_transformation_function
#        self._build_model(training_images, training_shape_list)
#        
#    def _build_model(self,training_images,training_shape_list):
#        all_patches = []
#        for i in range(len(training_images)):
#            patch,_ = extract_patch_normal(training_images[i],training_landmarks[i],20,0,image_transformation_function=self._img_trans_func,patch_transformation_function=self._patch_trans_func)
#            all_patches.append(np.array(patch))
#        all_grey_data = np.swapaxes(np.array(all_patches),0,1)
#        npts,nimgs,ph,pw = all_grey_data.shape
#        all_grey_data = all_grey_data.reshape(npts,nimgs,ph*pw)
#        self._point_models = []
#        for i in range(npts):
#            data = np.squeeze(all_grey_data[i,:,:])
#            self._point_models.append(GaussianModel(data))
#
#    def get_size(self):
#        """
#        Returns the number of grey point models - i.e the number of landmarks
#        :return: Number of point models
#        """
#        return len(self._point_models)
#
#    def get_point_grey_model(self, point_index):
#        """
#        :param point_index: The index of the landmark
#        :return: The modedPCAModel for the landmark
#        """
#        return self._point_models[point_index]
#    
#    def _search_point(self,point_model,test_patch):
#        test_length = len(test_patch)
#        patch_length = len(point_model.get_mean())
#        errors = []
#        for i in range(1+test_length-patch_length):
#            test_subpatch = test_patch[i:(i+patch_length)]
#            error,_,_ = point_model.fit(test_subpatch)
#            errors.append(error)
#        errors = np.array(errors)
#        new_index = np.argmin(errors)
#        location = (new_index + int(patch_length/2))
#        return location,errors
#            
#    
#    def search(self,test_image,starting_landmark):
#        new_points = []
#        test_patches,test_points = extract_patch_normal(test_image,starting_landmark,self._search_num_pixels,self._patch_num_pixels_width,image_transformation_function=self._img_trans_func,patch_transformation_function=self._patch_trans_func)       
#        test_patches = np.squeeze(np.array(test_patches))
#        npts,ph= test_patches.shape
#        original_point_location = int(ph/2)
#        point_idxs = []
#        displacements = []
#        for index,point_model in enumerate(self._point_models):
#            location,errors = self._search_point(point_model,np.squeeze(test_patches[index,:]))
#            displacement = location-original_point_location
#            displacements.append(displacement)
#        mean_disp = np.abs(np.array(displacements)).mean()
#        shape_points = []
#        updated_point_locations = []       
#        for index,displacement in enumerate(displacements):
#            if np.abs(displacement) > mean_disp:
#                displacement = np.sign(displacement)*np.round(mean_disp)
#            updated_point_location = int(original_point_location + displacement)
#            updated_point_locations.append([updated_point_location,index])
#            point_coords = np.uint32(np.round(np.squeeze(test_points[index,updated_point_location,:]))).tolist()
#            shape_points.append(point_coords)
#        return Shape(np.array(shape_points)),updated_point_locations
    

#def process(training_images,training_landmarks,test_image,test_landmark,gm):
    #aligned_training_landmarks = training_landmarks.align()
    #mean_shape = aligned_training_landmarks.get_mean_shape()
    #X = aligned_training_landmarks.as_collapsed_vector()
    #print X.shape
    #for index,shape in enumerate(training_landmarks):
    #    plot_shapes([shape,mean_shape.align(shape)],['original','aligned mean'])
    #sk_pca = PCA()
    #sk_pca.fit(X)
    #my_pca = PCAModel(X)
    #eigenvalues = my_pca.get_eigenvalues()
    #varfrac = my_pca.get_varfrac()
    #plot_line(eigenvalues,'PDM - PCA - Eigenvalue Plot','Index','Eigenvalue')
    #plot_line(np.cumsum(varfrac),'PDM - PCA - Variance Plot','Number of Principal Components','Fraction of Variance Explained')
    #d,n=sk_pca.components_.shape
    #for i in range(d):
    #    print np.sqrt(np.sum((sk_pca.components_[i,:]-my_pca.get_eigenvectors().T[i,:])**2)), np.linalg.norm(my_pca.get_eigenvectors().T[i,:])
    #my_mpca = ModedPCAModel(X,pca_variance_captured=0.99,pca_model=my_pca)
    #num_modes = my_mpca.get_number_of_modes()    
    #for i in range(num_modes):
    #    mode_shapes = []
    #    for j in range(-1, 2):
    #        factors = np.zeros(num_modes)
    #        factors[i] = j
    #        plot_shapes([Shape.from_collapsed_vector(my_mpca.generate(factors))],labels=['Mode:' + str(i) + ' S.D:' + str(3*j)])
    #for index,shape in enumerate(training_landmarks):
    #    error,shape_vector,_ = my_mpca.fit(shape.align(mean_shape).as_collapsed_vector())
    #    #plot_shapes([shape,Shape.from_collapsed_vector(shape_vector).align(shape)],['original','model fit'])
    #errorTest,shape_vector,_ = my_mpca.fit(test_landmark.align(mean_shape).as_collapsed_vector())  
    #plot_shapes([test_landmark,Shape.from_collapsed_vector(shape_vector).align(test_landmark)],['original','model fit'])
    #pdm = PointDistributionModel(training_landmarks,pca_variance_captured=0.99,shape_fit_max_iters=10000,shape_fit_tol=1e-7)
    #final_shape,error,num_iters = pdm.fit(test_landmark)
    #print error,num_iters
    #plot_shapes([test_landmark,final_shape],['original','model fit'])
    #pdata=np.squeeze(np.array(extract_patch_normal(test_image,test_landmark,20,0,image_transformation_function=image_transformation_default,patch_transformation_function=patch_transformation_default)))
    #pdata2=np.squeeze(np.array(extract_patch_normal(test_image,test_landmark,100,0,image_transformation_function=image_transformation_default,patch_transformation_function=patch_transformation_default)))    
    #gm = GreyModel(training_images,training_landmarks,15,50,image_transformation_function=image_transformation_default,patch_transformation_function=patch_transformation_default)
    #test_patches,test_points = extract_patch_normal(test_image,test_landmark,50,0,image_transformation_function=image_transformation_default,patch_transformation_function=patch_transformation_default)
    #_,points= gm.search(test_image,test_landmark)    
    #imshow2(pdata)
    #imshow2(overlay_points_on_image(np.squeeze(np.array(test_patches)),points,width=2))
#   pass




#pcamodels = []
#for index,split in enumerate(LeaveOneOutSplitter(data,Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH)):
#    training_images,training_landmarks,training_segmentations = split.get_training_set()
#    test_image,test_landmark,big_test_segmentation = split.get_test_example()
#    pcamodels.append(PCAModel(training_landmarks.align().as_collapsed_vector()))
#
#a_train_errors = []
#a_test_errors = []
#for nc in range(2,2*(len(pcamodels)-2)):
#    s_train_errors = []
#    s_test_errors = []
#    for index,split in enumerate(LeaveOneOutSplitter(data,Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH)):
#        pdm = PointDistributionModel(training_landmarks,pca_number_of_components=nc,pca_model=pcamodels[index])
#        s_train_errors.append(pdm.get_training_error())
#        _,test_error,_= pdm.fit(test_landmark)
#        s_test_errors.append(test_error)
#    a_train_errors.append(np.mean(np.array(s_train_errors)))
#    a_test_errors.append(np.mean(np.array(s_test_errors)))
#
#plot_line(a_train_errors,'PDM Fit Error on LOO-Training Erro vs Number of Modes','Number of Modes','MSE')
#plot_line(a_test_errors,'PDM Fit Error on LOO-Test vs Number of Modes','Number of Modes','MSE')
    
    
#for index,split in enumerate(LeaveOneOutSplitter(data,Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH)):
    #if index > 0:
    #    pass#break
    #training_images,training_landmarks,training_segmentations = split.get_training_set()
    #test_image,test_landmark,big_test_segmentation = split.get_test_example()
    #process(training_images,training_landmarks,test_image,test_landmark,gms[index])


i = 0
for index,split in enumerate(LeaveOneOutSplitter(data,Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH)):
    training_images,training_landmarks,training_segmentations = split.get_training_set()
    test_image,test_landmark,big_test_segmentation = split.get_test_example()
    transformed_test_image = test_image
    transformed_training_images = training_images
    split_training_landmarks = tooth_splitter(training_landmarks,2)
    all_split_training_landmarks = tooth_splitter(training_landmarks,8)
    two_upper_teeth = all_split_training_landmarks[1]
    two_lower_teeth = all_split_training_landmarks[5]
    ltl=[]
    utl = []
    mtl = []
    for i in range(len(all_split_training_landmarks[0])):
        ashape = all_split_training_landmarks[1][i].concatenate(all_split_training_landmarks[2][i])
        bshape = all_split_training_landmarks[5][i].concatenate(all_split_training_landmarks[6][i])
        cshape = ashape.concatenate(bshape)
        utl.append(ashape)
        ltl.append(bshape)
        mtl.append(cshape)
    two_upper_teeth=ShapeList(utl)
    two_lower_teeth=ShapeList(ltl)
    four_middle_teeth = ShapeList(mtl)
    shape_model = PointDistributionModel(training_landmarks,pca_variance_captured=0.99,use_transformation_matrix=True,project_to_tangent_space=True)
    upper_shape_model = PointDistributionModel(split_training_landmarks[0],pca_variance_captured=0.99,use_transformation_matrix=True,project_to_tangent_space=True)
    lower_shape_model = PointDistributionModel(split_training_landmarks[1],pca_variance_captured=0.99,use_transformation_matrix=True,project_to_tangent_space=True)
    grey_model = GreyModel(training_images,training_landmarks,20,50,image_transformation_function=image_transformation_default,patch_transformation_function=patch_transformation_default)
    active_shape_model = ActiveShapeModel(shape_model,grey_model)
    appearance_model = AppearanceModel(transformed_training_images,shape_model,[1,1],[1.1,1.5],5)
    upper_appearance_model = AppearanceModel(transformed_training_images,upper_shape_model,[1,1],[1.1,1.5],5)
    lower_appearance_model = AppearanceModel(transformed_training_images,lower_shape_model,[1,1],[1.1,1.5],5)
    centroid_shape = appearance_model.get_default_shape()
    upper_centroid_shape = upper_appearance_model.get_default_shape()
    lower_centroid_shape = lower_appearance_model.get_default_shape()
    appearance_shape,val = appearance_model.fit(transformed_test_image)
    upper_appearance_shape,upper_val = upper_appearance_model.fit(transformed_test_image)
    lower_appearance_shape,lower_val = lower_appearance_model.fit(transformed_test_image)
    appearance_shape2 = upper_appearance_shape.concatenate(lower_appearance_shape)
    plot_shapes([test_landmark,centroid_shape],labels=['test_landmark','centroid initialization'])
    plot_shapes([test_landmark,appearance_shape],labels=['test_landmark','template match initialization'])
    plot_shapes([test_landmark,appearance_shape2],labels=['test_landmark','split template match initialization'])
    print 'Initialization',split.get_dice_error_on_test(centroid_shape),split.get_dice_error_on_test(appearance_shape),split.get_dice_error_on_test(appearance_shape2)
    newest_shape0,_,_ = active_shape_model.fit(transformed_test_image,initial_shape=centroid_shape,tol=0.2, max_iters=10)
    newest_shape1,_,_ = active_shape_model.fit(transformed_test_image,initial_shape=appearance_shape,tol=0.2, max_iters=10)
    newest_shape2,_,_ = active_shape_model.fit(transformed_test_image,initial_shape=appearance_shape2,tol=0.2, max_iters=10)
    newest_shape3,_,_ = active_shape_model.fit(transformed_test_image,initial_shape=test_landmark,tol=0.2, max_iters=10)
    plot_shapes([test_landmark,newest_shape0],labels=['test_landmark','asm with centroid initialization'])
    plot_shapes([test_landmark,newest_shape1],labels=['test_landmark','asm with template match initialization'])
    plot_shapes([test_landmark,newest_shape2],labels=['test_landmark','asm with split template match initialization'])
    plot_shapes([test_landmark,newest_shape3],labels=['test_landmark','asm with manual initialization'])
    print 'ASM',split.get_dice_error_on_test(newest_shape0),split.get_dice_error_on_test(newest_shape1),split.get_dice_error_on_test(newest_shape2),split.get_dice_error_on_test(newest_shape3)    

    i = i+1
    if i >= 1:
        pass#break
