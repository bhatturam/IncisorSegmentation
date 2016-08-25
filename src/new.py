# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 23:11:34 2016

@author: bharath
"""
import numpy as np
import cv2
from incisorseg.dataset import *
from incisorseg.utils import *
from active_shape_models.shape import *
from active_shape_models.pca import PCAModel
from active_shape_models.imgproc import extract_patch_normal
from active_shape_models.models import ModedPCAModel,PointDistributionModel
from sklearn.decomposition import PCA
#from scipy.stats import pearsonr)

data = Dataset('../data/')

def process(training_images,training_landmarks,test_image,test_landmark):
    #aligned_training_landmarks = training_landmarks.align()
    #mean_shape = aligned_training_landmarks.get_mean_shape()
    #X = aligned_training_landmarks.as_collapsed_vector()
    #print X.shape
    #for index,shape in enumerate(training_landmarks):
    #    plot_shapes([shape,mean_shape.align(shape)],['original','aligned mean'])
    #sk_pca = PCA()
    #sk_pca.fit(X)
    #my_pca = PCAModel(X)
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
    #errorTot = 0;
    #for index,shape in enumerate(training_landmarks):
    #    error,shape_vector,_ = my_mpca.fit(shape.align(mean_shape).as_collapsed_vector())
    #    errorTot+=error
        #plot_shapes([shape,Shape.from_collapsed_vector(shape_vector).align(shape)],['original','model fit'])
    #errorTest,shape_vector,_ = my_mpca.fit(test_landmark.align(mean_shape).as_collapsed_vector())  
    #plot_shapes([test_landmark,Shape.from_collapsed_vector(shape_vector).align(test_landmark)],['original','model fit'])
    #print errorTot,errorTest
    #pdm = PointDistributionModel(training_landmarks,pca_variance_captured=0.99,shape_fit_max_iters=10000,shape_fit_tol=1e-7)
    #final_shape,error,num_iters = pdm.fit(test_landmark)
    #print error,num_iters
    #plot_shapes([test_landmark,final_shape],['original','model fit'])
    pdata=np.array(extract_patch_normal(test_image,test_landmark,20,0))
    imshow2(np.squeeze(pdata))
    print pdata.shape
    

for index,split in enumerate(LeaveOneOutSplitter(data,Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH)):
    if index > 0:
        pass#break
    training_images,training_landmarks,training_segmentations = split.get_training_set()
    test_image,test_landmark,big_test_segmentation = split.get_test_example()
    process(training_images,training_landmarks,test_image,test_landmark)