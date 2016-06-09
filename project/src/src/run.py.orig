
# coding: utf-8

# # H02A5A Computer Vision Project - Incisor Segmentation
# 
# ## Imports

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import numpy as np
import cv2
#from incisorseg.dataset import Dataset
#from incisorseg.utils import *
from asm.shape import Shape
#from asm.shapemodel import ActiveShapeModel


# ## Reading the dataset

# In[2]:

#data = Dataset('/home/bharath/workspace/CV/Project/data/')


# In[6]:

#img,mimg = data.get_training_images([0])
#l,ml = data.get_training_image_landmarks([0],Dataset.ALL_TEETH)
#lc,mlc = data.get_training_image_landmarks([0],Dataset.ALL_TEETH,True)
#plot_shapes(lc)
#imshow2(overlay_shapes_on_image(img[0],lc))
#plot_shapes(mlc)
#imshow2(overlay_shapes_on_image(mimg[0],mlc))
#lc,mlc = data.get_training_image_landmarks(Dataset.ALL_TRAINING_IMAGES,Dataset.ALL_TEETH,True)
#landmarks = lc + mlc
#model = ActiveShapeModel(landmarks)
#plot_shapes(model.aligned_shapes())
#plot_shapes([model.mean_shape()])
#plot_shapes(model.mode_shapes(1))

import numpy as np
import cv2
from incisorseg.dataset import Dataset
from incisorseg.utils import *
from asm.shape import Shape
from asm.shapemodel import ActiveShapeModel

def get_point_normal_pixel_coordinates(shape,point_index,nn,npx):
    point = shape.get_point(point_index)
    neighborhood = shape.get_neighborhood(point_index, nn)
    line = cv2.fitLine(neighborhood, cv2.DIST_L2, 0, 0.01, 0.01);
    slope = line[0:2] / np.sqrt(np.sum(line[0:2] ** 2))
    return [[int(point[1] + (incr * slope[0]) + 0.5), int(point[0] - (incr * slope[1]) + 0.5)] for incr in
            range(-npx, npx + 1)]

#data = Dataset('/home/bharath/workspace/CV/Project/data/')
data = Dataset('/home/r0607273/workspace/cv/Project/data/')

img,_ = data.get_training_images([0])
l,_ = data.get_training_image_landmarks([0],[0])
shape = l[0][0]

normal_pixels = []
for i in range(shape.size()):
    normal_pixels += get_point_normal_pixel_coordinates(shape,i,4,20)
cv2.imshow("bleh",overlay_points_on_image(img[0],normal_pixels))
cv2.waitKey(0)
