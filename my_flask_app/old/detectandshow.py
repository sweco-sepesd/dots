import argparse
#import pathlib
from PIL import Image, ImageFilter
import cv2
import numpy
#import time
#import skimage
import uuid

#Create uuid
user_uuid = uuid.uuid4()

#template_by_filename = {
#    'template.jpg': ({
#          98: numpy.array([[[1970.0, 1474.0],[ 2169.0, 1474.0],[ 2169.0, 1673.0],[ 1970.0, 1673.0]]], dtype=numpy.float32)
#        , 62: numpy.array([[[236.0, 1671.0],[ 37.0, 1671.0],[ 37.0, 1472.0],[ 236.0, 1472.0]]], dtype=numpy.float32)
#        , 203: numpy.array([[[46.0, 35.0],[ 245.0, 35.0],[ 245.0, 234.0],[ 46.0, 234.0]]], dtype=numpy.float32)
#        , 23: numpy.array([[[1970.0, 28.0],[ 2169.0, 28.0],[ 2169.0, 227.0],[ 1970.0, 227.0]]], dtype=numpy.float32)
#    }, (2200,1700), (231,1474,246,1968))
#}
#template_by_filename = {
#    'templateSkane.png': ({
#         98: numpy.array([[[711.0, 683.0],[ 802.0, 683.0],[ 802.0, 776.0],[ 694.0, 776.0]]], dtype=numpy.float32)
#      ,  62: numpy.array([[[18.0, 680.0],[ 112.0, 680.0],[ 112.0, 774.0],[ 18, 774.0]]], dtype=numpy.float32)
#      ,  203: numpy.array([[[ 17.0, 17.0],[ 105, 17.0],[ 105.0, 105.0],[ 17.0, 105.0]]], dtype=numpy.float32)
#      ,  23: numpy.array([[[ 709.0, 17.0],[ 800.0, 17.0],[ 800.0, 110.0],[ 709.0, 110.0]]], dtype=numpy.float32)
#    }, (818,794), (98,760,48,738)) #(819,794), (106,710,106,682)
#}
#
#Define template for detected markers
template_by_filename = {
    'templateSkane2.png': ({
         98: numpy.array([[[890.0, 671.0],[ 996.0, 671.0],[ 996.0, 777.0],[ 890.0, 777.0]]], dtype=numpy.float32)
      ,  62: numpy.array([[[15.0, 671.0],[ 121.0, 671.0],[ 121.0, 777.0],[ 15, 777.0]]], dtype=numpy.float32)
      ,  203: numpy.array([[[15.0, 15.0],[ 121, 15.0],[ 121.0, 121.0],[ 15.0, 121.0]]], dtype=numpy.float32)
      ,  23: numpy.array([[[890.0, 15.0],[ 996.0, 15.0],[ 996.0, 110.0],[ 890.0, 110.0]]], dtype=numpy.float32)
    }, (1010,792), (15,777,136,875)) #(819,794), (106,710,106,682)
}

#Takes a set of marker IDs, checks if they match any templates, and returns the best match if found
def search_template(ids):
    matched = []
    for filename, (markers,size,roi) in template_by_filename.items():
        if ids - markers.keys(): continue
        matching = ids & markers.keys()
        if matching:
            matched.append((len(matching) * -100.0 / len(markers.keys()), filename, {k:markers[k] for k in matching},size,roi))
    if not matched:
        return False, None
    return True, sorted(matched)[0][1:]



# Prepare a lookup table for possible names of "OpenCV Aruco Dictionaries" and utility function.
# These will be used for the command line help message and validation
dictionaries = {n[5:]:getattr(cv2.aruco,n) for n in dir(cv2.aruco) if n.startswith('DICT_')}
def get_dictionary(string):
    return cv2.aruco.getPredefinedDictionary(dictionaries[string])

parser = argparse.ArgumentParser(description='Find markers in image')
parser.add_argument(
      '--dictionary'
    , choices=dictionaries.keys()
    , type=get_dictionary
    , default=get_dictionary('6X6_250'))
parser.add_argument('-c', '--crop', action='store_true', default=False)
args = parser.parse_args()

detector = cv2.aruco.ArucoDetector()
detector.setDictionary(args.dictionary)




'''
corners	vector of detected marker corners. For each marker, its four corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array is Nx4. The order of the corners is clockwise.
ids	vector of identifiers of the detected markers. The identifier is of type int (e.g. std::vector<int>). For N detected markers, the size of ids is also N. The identifiers have the same order than the markers in the imgPoints array.
rejectedImgPoints	contains the imgPoints of those squares whose inner code has not a correct codification. Useful for debugging purposes.
'''
# Initialize the camera
capture = cv2.VideoCapture(0)  # 0 är laptopkameran, 1 är första externa
#possible = capture.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY)
#print(possible)
i = 0

# Check if the camera opened successfully
if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

do_crop = args.crop
do_resize = False
do_threshold = False
do_blur = False
do_photo = True
do_mask = True
do_smooth = True
img_counter = 0

while True:
    ok, im = capture.read()
    

    # Converts the image into HSV colorspace
    hsvImage = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Defines tolerances for green color in H S V. Mask: lower: 50, 60, 80 upper: 89,255,255
    lowerGreen = numpy.array([43, 38, 55]) 
    upperGreen = numpy.array([111, 255, 255]) 

    # Red Mask: lower 
    lowerRed = numpy.array([150, 35, 40]) # 83, 86, 145 # RBG: 155, 64, 79
    upperRed = numpy.array([179, 177, 184])
    
    lowerColor = lowerGreen
    upperColor = upperGreen
    
#H 102/75 S 177/25 V 141/46
#H min/max 

    if not ok: continue
    
    # Creates a mask containing green pixels
    mask = cv2.inRange(hsvImage, lowerColor, upperColor)
    
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected_pts = detector.detectMarkers(im)
    if ids is None or not len(ids): continue
    detected = dict(zip(numpy.ravel(ids), corners)) 
    ok, matched = search_template(detected.keys())
    if not ok: 
        continue
    filename, markers, size, roi = matched
    i += 1
    #print(i, len(markers), filename)
    pts_src = []
    pts_dst = []
    for id in markers.keys():
        pts_src.append(detected[id].reshape((4,2)))
        pts_dst.append(markers[id].reshape((4,2)))
        
    h, status = cv2.findHomography(numpy.concatenate(pts_src), numpy.concatenate(pts_dst))

    im = cv2.warpPerspective(im, h, size)
    # mask = cv2.warpPerspective(mask, h, size)

    #mask = skimage.transform.warp(mask, h, output_shape=(1200, 1200)) #* 255.0

    #mask = cv2.warpPerspective(mask, h, size)
    if do_crop:
        #cropped = img[start_row:end_row, start_col:end_col]
        start_row, end_row, start_col, end_col = roi
        # im = im[start_row:end_row, start_col:end_col]  Ändrad från im till mask på bägge
        mask = mask[start_row:end_row, start_col:end_col]
    if do_resize:
        # h, w = im.shape
        # im = cv2.resize(im, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST_EXACT)
        # im = cv2.medianBlur(im,7)
        mask = cv2.resize(mask, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LANCZOS4)


    if do_blur:
        if do_mask == True:
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)


    if do_threshold:
        if do_mask == True:
            (T, mask) = cv2.threshold(mask, 0, 255,	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        else:    
            (T, im) = cv2.threshold(im, 0, 255,	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if do_smooth:
        # Convert to RGB and PIL image
        frame_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(frame_rgb)
        #Filter
        filtered_image = pil_image.filter(ImageFilter.MedianFilter(size=3))

        #Convert back to HSV and cv2
        mask = cv2.cvtColor(numpy.array(filtered_image), cv2.COLOR_RGB2GRAY)

    #im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3)

    if do_mask == True:
        cv2.imshow('image', mask)  #replace mask <-> im    
    else:
        cv2.imshow('image', im)  #replace mask <-> im
    #cv2.imshow('image', mask)  #replace mask <-> im
    k = cv2.waitKey(10) & 0XFF
    if k == 27:
        break
    elif k == ord('m'):
        do_mask = not do_mask
    elif k == ord('c'):
        do_crop = not do_crop
    elif k == ord('r'):
        do_resize = not do_resize
    elif k == ord('t'):
        do_threshold = not do_threshold
    elif k == ord('b'):
        do_blur = not do_blur
    elif k == ord('s'):
        do_smooth = not do_smooth


    elif k == ord('p'):
        img_name = "{}.png".format(user_uuid)
        #img_name = "masked_frame_{}.png".format(img_counter)
        # save_geotiff(mask, img_name)
        cv2.imwrite(img_name, mask)
        #print("{} written!".format(img_name))
        img_counter += 1
        break
        



print (user_uuid)
capture.release()
cv2.destroyAllWindows()

#print(numpy.concatenate(pts_src))
#print(numpy.concatenate(pts_dst))

# TODO: Make use of the result

# Alternative, capture images continuously and somehow yield the result, possibly rectify as well...:
#while True:
#    cm = cv2.VideoCapture(0)
#    ok, im = cm.read()
#    if not ok: continue
#    result = detector.detectMarkers(im)

'''
pts_src and pts_dst are numpy arrays of points
in source and destination images. We need at least
4 corresponding points.
'''
#h, status = cv2.findHomography(pts_src, pts_dst)
 
'''
The calculated homography can be used to warp
the source image to destination. Size is the
size (width,height) of im_dst
'''
 
#im_dst = cv2.warpPerspective(im_src, h, size)
