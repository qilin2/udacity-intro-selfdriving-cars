#Project: Traffic Light Classifier
#Name   : Brandon Qilin Hung

# Libraries
import cv2 # computer vision library
import helpers # helper functions

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

%matplotlib inline

# Directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Load Training Data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

# The first image in IMAGE_LIST is displayed below (without information about shape or label)
selected_image = IMAGE_LIST[1186][0] # 1186 is the last image(green) yellow at 725
plt.imshow(selected_image)

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im,(32,32)) #Resizing image to size (32,32)
    rows = 4
    cols = 6
    i = standard_im.copy()
    #Cropping 4 rows from both upper and lower end of image 
    i = i[rows:-rows, cols:-cols, :]
    #Applying gaussian blur to image to remove noise
    i = cv2.GaussianBlur(i, (3, 3), 0)
    return i


## Given a label - "red", "green", or "yellow" - return a one-hot encoded label
def one_hot_encode(label):
    
    one_hot_encoded = [0,0,0] 
    if(label == "red"):
        one_hot_encoded[0] = 1
    elif(label == "yellow"):
        one_hot_encoded[1] = 1
    elif(label == "green"):
        one_hot_encoded[2] = 1
    else:
        raise TypeError('Please input red, yellow, or green. Not ', label)
    return one_hot_encoded


# Livbraries
import test_functions
tests = test_functions.Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)


def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)


sum1 = 0
sum2 = 0
sum3 = 0
diff1 = 758 - 723
diff2 = 723 - 0
diff3 = 1186 - 758
sat = 3.2
bri = 0.32
smallest_r = 999.0
smallest_y = 999.0
smallest_g = 999.0
largest_r = 0.0
largest_y = 0.0
largest_g = 0.0
no_y = 0
no_r = 0
no_g = 0
red_threshold = 4
yellow_threshold = 10


for j in range(723,758):

    i = STANDARDIZED_LIST[j][0]    #end 757 start 723

    hsv = cv2.cvtColor(i, cv2.COLOR_RGB2HSV)
    height = hsv.shape[0]
    width = hsv.shape[1]
    area = height * width

#     hsv[...,1] = hsv[...,1]*sat
#     hsv[...,2] = hsv[...,2]*bri


    # lower_red = np.array([1,70,50])
    # upper_red = np.array([10,255,255])
    # mask0 = cv2.inRange(hsv, lower_red, upper_red)
    #
    # lower_red = np.array([170,70,50])
    # upper_red = np.array([180,255,255])
    # mask1 = cv2.inRange(hsv, lower_red, upper_red)


    # mask = mask0+mask1

    lower_yellow = np.array([15,55,55])
    upper_yellow = np.array([35,255,255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    s = np.sum(mask)
    s = s / area
    if(s > yellow_threshold):
        no_y += 1
    if(smallest_y > s):
        smallest_y = s
    if(largest_y < s):
        largest_y = s
    sum2 += s

# sum1 /= diff
sum2 /= diff2
# sum3 /= diff3
# print("sum_y = " + str(sum1))
print("sum_y = " + str(sum2))
# print("sum_g = " + str(sum3))
print("smallest_y = " + str(smallest_y))
# print("smallest_r = " + str(smallest_r))
# print("smallest_g = " + str(smallest_g))
print("largest_y = " + str(largest_y))
# print("largest_r = " + str(largest_r))
# print("largest_g = " + str(largest_g))
print("no_y = " + str(no_y))
# print("no_r = " + str(no_r))
# print("no_g = " + str(no_g))


# Convert and image to HSV colorspace
# Visualize the individual color channels

image_num = 700
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

# Print image label
print('Label [red, yellow, green]: ' + str(test_label))

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(test_im)
ax2.set_title('H channel')
ax2.imshow(h, cmap='gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def avg_value(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    area = hsv.shape[0] * hsv.shape[1]
    #sum up the value to know color intensity 
    sum1 = np.sum(hsv[:, :, 2])
    #Find average color intensity of image
    avg1 = sum1 / area
    return int(avg1)

def create_feature(rgb_image):
    img = rgb_image.copy()
    #Create 3 slices of image vertically.
    upper_slice = img[0:7, :, :]
    middle_slice = img[8:15, :, :]
    lower_slice = img[16:24, :, :]
    #Find avergae value of each image.
    #To decide which traffic light might be on.
    u1 = avg_value(upper_slice)
    m1 = avg_value(middle_slice)
    l1 = avg_value(lower_slice)
    return u1,m1,l1


# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    u1,m1,l1 = create_feature(rgb_image)
 
    if(u1 > m1 and u1 > l1):
        return [1,0,0]
    elif(m1 > l1):
        return [0,1,0]
    else:
        return [0,0,1]
    
    
# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)



# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)
def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

# Visualize misclassified example(s)
## TODO: Display an image in the `MISCLASSIFIED` list 
## TODO: Print out its predicted label - to see what the image *was* incorrectly classified as
n = 1
plt.imshow(MISCLASSIFIED[n][0])
print(MISCLASSIFIED[n][1])


# Libraries
import test_functions
tests = test_functions.Tests()

if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")