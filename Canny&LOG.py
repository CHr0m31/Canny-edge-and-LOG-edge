# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:06:48 2020

@author: woocj
"""
import numpy as np
import os
import cv2 
mypath = os.path.dirname(os.path.abspath(__file__))+'\\'

def gaussianSmoothing(image):
    """
    Applies 7x7 Gaussian Filter to the image by convolution operation
    :type image: object
    """
    imageArray = np.array(image)
    gaussianArr = np.array(image)
    sum = 0

    for i in range(3, image.shape[0] - 3):
        for j in range(3, image.shape[1] - 3):
            sum = applyGaussianFilterAtPoint(imageArray, i, j)
            gaussianArr[i][j] = sum
    #result is the image after gaussian smoothing
    return gaussianArr

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    # Using G(x,y) = e^((-x^2-y^2)/2σ^2)/(2πσ^2) to create the kernel
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return kernel

#applying gaussian filter at each and every individual point
def applyGaussianFilterAtPoint(imageData, row, column):
    sum = 0
    for i in range(row - 3, row + 4):
        for j in range(column - 3, column + 4):
            sum += gaussian_filter[i - row + 3][j - column + 3] * imageData[i][j]

    return sum

#applying log 2nd derivative edge detection to create the kernel
def log_2nd_derivative(sigma = 1):
    #using the size 7
    size = int(2*(np.ceil(3*sigma))+1)
    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))
    
    # using G(x,y) = e^(-(x^2+y^2)/2σ^2)/(2πσ^2)*(x^2+y^2-σ^2)/σ^4 to create the kernel
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * \
        np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal
    return kernel


def convolve(image, kernel):
    xmax = image.shape[0]
    ymax = image.shape[1]
    Kmax = kernel.shape[0]
    Koffset = Kmax//2
    convo = np.zeros([xmax, ymax], dtype=np.int32)
    # Border pixels were not convolved, which will be removed from the final image.
    for i in range (Koffset, xmax-Koffset):
        for j in range (Koffset, ymax-Koffset):
            sum = 0
            for a in range (0, Kmax):
                for b in range (0, Kmax):
                    sum += kernel[a][b]*image[i+a-Koffset][j+b-Koffset]
            convo[i][j] = sum
    return convo

 
    
#Using the sobel filter to convolve the image    
def sobel_filters(img):
    Sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], np.float32)
    Sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], np.float32)
    # Get Derivatives in x direction and y direction
    Ix = convolve(img, Sobel_x)
    Iy = convolve(img, Sobel_y)
    # Compute Gradient and Angle based on x and y derivatives.
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    angle = np.arctan2(Iy, Ix)
    return (G, angle)

def non_max_suppression(image_matrix, angle_matrix):
    x, y = image_matrix.shape
    Z = np.zeros((x, y), dtype = np.int32)
    # Convert radian into degree
    angle_matrix_360 = angle_matrix * 180/np.pi
    
    # Convert angle into 0-180 for comparison
    angle_matrix_360[angle_matrix_360 < 0 ] += 180
    for i in range (1, x-1):
        for j in range (1, y-1):
            p = 255
            q = 255
            
            # See the neighbors of the edges to see if it satisfies the non-maximum
            if (0 <= angle_matrix_360[i,j] < 22.5) or (157.5 <= angle_matrix_360[i, j ] < 180):
                p = image_matrix[i, j+1]
                q = image_matrix[i, j-1]
            elif(22.5 <= angle_matrix_360[i,j] < 67.5):
                p = image_matrix[i+1, j-1]
                q = image_matrix[i-1, j+1]
            elif(67.5 <= angle_matrix_360[i,j] < 112.5):
                p = image_matrix[i+1, j]
                q = image_matrix[i-1, j]
            elif(112.5 <= angle_matrix_360[i,j] < 157.5):
                p = image_matrix[i-1, j-1]
                q = image_matrix[i+1, j+1]
            if (image_matrix[i,j] >= p) and (image_matrix[i, j] >= q):
                Z[i,j] = image_matrix[i,j]
            else:
                Z[i,j] = 0
    return Z

#checking every file with .raw 
for subdir, dirs, files in os.walk(mypath):
        for file in files:
            if file.endswith('.raw'):
                mypath = os.path.join(subdir, file)
                name = str(file.split('.',1)[0])
                #printing the name of the file that is being processed 
                print(name)  
                
                #changing the raw file to binary array
                file = open(mypath, 'r+b')
                data = file.read()
                raw_array = []
                i = 1
                image_row = 0
                matrix = []
                for element in data:
                    if i == 1 or i == 3:
                        previous_data = element
                    if i == 2:
                        row = 256*element+previous_data
                    if i == 4:
                        col = 256*element+previous_data
                    
                    if i > 5:   
                        if image_row < row:
                            raw_array.append(element)
                            image_row += 1
                            if image_row == row:
                                matrix.append(raw_array)
                        else:
                            raw_array = [element]
                            image_row = 1
                    i += 1
                    
                np_array = np.array(matrix)
                

                #choosing the gassian_kernel to do edge detection using the kernel size 7
                gaussian_filter = gaussian_kernel(size = 7)     
                
                #applying gaussian smoothing
                gaussianData = gaussianSmoothing(np_array)
                
                photo , angle = sobel_filters(gaussianData)
                
                #using non max suppression to get edge thinned across the edge contour
                almost = non_max_suppression(photo,angle)
                
                output = 'Outputs/'+name+'_1st.jpg'
                cv2.imwrite(output, almost)
                
                #using log 2nd derivative to do tha edge detection
                gaussian_filter = log_2nd_derivative()
                
                gaussianData = gaussianSmoothing(np_array)
                photo , angle = sobel_filters(gaussianData)
                
                almost = non_max_suppression(photo,angle)
                output = 'Outputs/'+name+'_2nd.jpg'
                cv2.imwrite(output, almost)