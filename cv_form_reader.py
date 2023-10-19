#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import load_model


# In[2]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tensorflow.keras.datasets import mnist


# In[3]:


x_test_values = []
region_collector = []
space_list = []
return_values = []


# In[4]:


def ModifiedWay(rotate_img, angle):  # angle is in radians
    img_height, img_width = rotate_img.shape[0], rotate_img.shape[1]
    print("image_height: {0}, image_width: {1}".format(img_height, img_width))
    center_y, center_x = img_height//2, img_width//2
    rotationMatrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    sinofRotationMatrix = np.abs(rotationMatrix[0][0])
    cosofRotationMatrix = np.abs(rotationMatrix[0][1])

    newImageHeight = int((img_height * sinofRotationMatrix) + (img_width * cosofRotationMatrix))
    newImageWidth = int((img_height * cosofRotationMatrix) + (img_width * sinofRotationMatrix))
    print("NEW image height: {0}, NEW image width: {1}".format(img_height, img_width))

    # updating values of rotation matrix
    print(rotationMatrix[0][2])
    rotationMatrix[0][2] += (newImageWidth/2) - center_x
    rotationMatrix[1][2] += (newImageHeight/2) - center_y

    # performing image rotation below
    rotating_image = cv2.warpAffine(rotate_img, rotationMatrix, (newImageWidth, newImageHeight))
    print(rotating_image.shape)
    return rotating_image


# In[5]:


def finding_corner_points(image):
    ret, thresh = cv2.threshold(image_gray, 210, 255, cv2.THRESH_BINARY_INV)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]  # reading the foam as a contour
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c = sorted_contours[0]
    top_right = sorted(c, key=lambda x: x[0][0] - x[0][1], reverse=True)[0]


    bottom_left = sorted(c, key=lambda x: x[0][0] - x[0][1], reverse=False)[0]

    bottom_right = sorted(c, key=lambda x: x[0][0] + x[0][1], reverse=True)[0]

    top_left = sorted(c, key=lambda x: x[0][0] + x[0][1], reverse=False)[0]
    print('top right: {0}  top left {1}  bottom right {2}  bottom left {3}'
          .format(top_right, top_left, bottom_right, bottom_left))
    x_diff = (top_right[0][0] - bottom_right[0][0])

    y_diff = (bottom_left[0][1] - bottom_right[0][1])# self explanitory
    # center = (top_left[0][0] + x_diff/2, top_left[0][1] + y_diff/2)


    if x_diff != 0:
        angle_radian = math.atan(y_diff/x_diff)
        angle = angle_radian * 180/math.pi
    else:
        angle = 0


    if angle > 1:
        image_new = ModifiedWay(image, angle)
        finding_corner_points(image_new)
    else:
        return image, top_left, bottom_left, top_right, bottom_right


# In[6]:


def box_generation_algorithm(roi):
    # 18 rows, 26 columns
    map_ret, map_thresh = cv2.threshold(roi, 190, 255, cv2.THRESH_BINARY)
    map_contours, map_heiarchy = cv2.findContours(map_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    center_list = []
    final_list = []
    box_list = []
    l = 0
    cushon = -1
    for contour in map_contours:
        if map_heiarchy[0, l, 3] == -1 and cv2.contourArea(contour) > 2750:
            # NEED THE SPACE, NOT THE CONTOUR
            ext_left = tuple(contour[contour[:, :, 0].argmin()][0]) 
            ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
            ext_top = tuple(contour[contour[:, :, 0].argmin()][0]) 
            ext_bottom = tuple(contour[contour[:, :, 0].argmax()][0]) 
            x_center = (ext_left[0] + ext_right[0])/2 # for indexing
            y_center = (ext_top[1] + ext_bottom[1])/2
            center_list.append((x_center, y_center))
            box = roi[ext_top[1]:ext_bottom[1], ext_left[0]:ext_right[0]]
            box = cv2.rectangle(img=box, pt1=(0, 0), pt2=(box.shape[1],box.shape[0] ), color=255, thickness=2)
            box_list.append(box)
        l += 1

    a = sorted(zip(center_list, box_list), key=lambda x: x[0][0])
    for iterable in range(26):
        b = a[18*iterable:18*(iterable+1)]
        b = sorted(b, key = lambda y:y[0][1])
        boxes_sorted = [x for _,x in b]
        for item in boxes_sorted:
            final_list.append(item)
    return final_list


# In[7]:


def filler_handling(image_list, process_param):  # NEED TO REDO
    loop = 0  # for show
    index_list = []  # OUTPUT USED FOR INDEXING NUMBER TO BOX
    num_img_list = []  # NEED THIS FOR COLLECTING CHARS IN SEQUENCE
    ret_list = []
    for box in image_list:
        interlum_list = []
        x_list = []
        i = 0
        cnts = 0
        ret, find_char = cv2.threshold(box, 235, 255, cv2.THRESH_BINARY)  # worked best @ 235
        contours, heiarchy = cv2.findContours(find_char, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(heiarchy)
        for contour in contours:
            # GOES THROUGH ALL FOUND CONTOURS, COLLECTS OR REMOVES ACCORDINGLY IF AREA > 25
            cut = 15
            if cv2.contourArea(contour) < cut:
                pass
            elif heiarchy[0, i, 3] == -1:
                pass
            elif heiarchy[0, i, 3] == 0:
#                     loop += 1
#                     print(loop)
                M = cv2.moments(contour)
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                x_list.append(cX)
                for xa in contour:
                    for xb in xa:
                        xb[0] = xb[0] - cX + 15
                        xb[1] = xb[1] - cY + 15
                space = np.zeros((30,30))
                space = cv2.fillPoly(space, pts =[contour], color=(255,255,255))
                interlum_list.append(space)
                cnts += 1
            elif heiarchy[0, i, 3] > 1:
#                     loop += 1
#                     print(loop)
                M = cv2.moments(contour)
                cX = int(M['m10'] / M['m00'])
                for xa in contour:
                    for xb in xa:
                        xb[0] = xb[0] - cX + 15
                        xb[1] = xb[1] - cY + 15
                space = interlum_list[len(interlum_list) - 1]
                del interlum_list[len(interlum_list) - 1]
                space = cv2.fillPoly(space, pts =[contour], color=(0,0,0))
                interlum_list.append(space)

            else:
                pass
            i += 1
        loop += 1
        # print(loop)
        a = sorted(zip(x_list, interlum_list), key=lambda x: x[0])
        int_list = [x for _,x in a]
        for ints in int_list:
            num_img_list.append(ints)
        
        index_list.append(cnts)
        
    for item in num_img_list:
        if process_param == 0:
            ret_img = cv2.resize(item, (28,28), interpolation=cv2.INTER_LINEAR)
        if process_param == 1:
            ret, ret_img = cv2.threshold(item, 0, 255, cv2.THRESH_BINARY)
            ret_img = cv2.resize(ret_img, (28,28), interpolation=cv2.INTER_LINEAR)
        if process_param == 2:
            ret_img = cv2.GaussianBlur(item, (5,5), 0)
            ret, ret_img = cv2.threshold(ret_img, 127, 255, cv2.THRESH_BINARY)
            ret_img = cv2.resize(ret_img, (28,28), interpolation=cv2.INTER_LINEAR)
        if process_param == 3:
            ret_img = cv2.erode(item, (5,5))
            ret_img = cv2.dilate(ret_img, (5,5))
            ret_img = cv2.resize(ret_img, (28,28), interpolation=cv2.INTER_LINEAR)
        if process_param == 4:
            ret_img = cv2.resize(item, (28,28), interpolation=cv2.INTER_LINEAR)
            ret_img = cv2.erode(ret_img, (5,5))
            ret_img = cv2.dilate(ret_img, (5,5))
        ret_list.append(ret_img)
        
    return ret_list, index_list


# In[8]:


def image_prep(list_of_images):
    array_prep = []
    for image in list_of_images:
        image = image/image.max()
        image = np.expand_dims(image, 2)
        array_prep.append(image)
    output = np.array(array_prep)
    return output


# In[9]:


def predicting_numbers(region_collection, index_list):
    
    prediction_list = []
    a = 0
    b = 0
    loaded_model = load_model("trying_this_now")
    output = loaded_model.predict_classes(region_collection)
    for index in index_list:
        predictions = []
        if index == 0:
            predictions.append(None)
        else:
            b += index
            for char in output[a:b]:
                predictions.append(char)
            a += index
        prediction = tuple(predictions)
        prediction_list.append(prediction)
    return prediction_list
    


# In[10]:


image = cv2.imread('digital_data.JPG')


# In[11]:


image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[12]:


img, top_left, bottom_left, top_right, bottom_right = finding_corner_points(image_gray)


# In[13]:


roi = img[top_left[0][1]:bottom_right[0][1], top_left[0][0]:bottom_right[0][0]]
roi = roi[157:roi.shape[0], 32:roi.shape[1]]


# In[14]:


region_list = box_generation_algorithm(roi)


# In[15]:


fill_list_0, box_num_index_list_0 = filler_handling(region_list, process_param=0)
fill_list_1, box_num_index_list_1 = filler_handling(region_list, process_param=1)
fill_list_2, box_num_index_list_2 = filler_handling(region_list, process_param=2)
fill_list_3, box_num_index_list_3 = filler_handling(region_list, process_param=3)
fill_list_4, box_num_index_list_4 = filler_handling(region_list, process_param=4)


# In[16]:


finale_0 = image_prep(fill_list_0)
finale_1 = image_prep(fill_list_1)
finale_2 = image_prep(fill_list_2)
finale_3 = image_prep(fill_list_3)
finale_4 = image_prep(fill_list_4)


# In[17]:


a = predicting_numbers(finale_0, box_num_index_list_0)
b = predicting_numbers(finale_1, box_num_index_list_1)
c = predicting_numbers(finale_2, box_num_index_list_2)
d = predicting_numbers(finale_3, box_num_index_list_3)
e = predicting_numbers(finale_4, box_num_index_list_4)
for i in range(468):
    print(a[i])
    print(b[i])
    print(c[i])
    print(d[i])
    print(e[i])
    print("--------------")


# In[18]:


def read_single_img(image):
    plt.imshow(image)
    array_prep = []
    image = image/image.max()
    image = np.expand_dims(image, 2)
    array_prep.append(image)
    output = np.array(array_prep)
    a = predicting_numbers(output, [1])
    print(a)
    


# In[19]:


img = fill_list_0[51]
read_single_img(img)


# In[ ]:





# In[ ]:





# In[ ]:




