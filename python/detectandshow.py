import argparse
import pathlib

import cv2
import numpy
import time

template_by_filename = {
    'template.jpg': ({
          98: numpy.array([[[1970.0, 1474.0],[ 2169.0, 1474.0],[ 2169.0, 1673.0],[ 1970.0, 1673.0]]], dtype=numpy.float32)
        , 62: numpy.array([[[236.0, 1671.0],[ 37.0, 1671.0],[ 37.0, 1472.0],[ 236.0, 1472.0]]], dtype=numpy.float32)
        , 203: numpy.array([[[46.0, 35.0],[ 245.0, 35.0],[ 245.0, 234.0],[ 46.0, 234.0]]], dtype=numpy.float32)
        , 23: numpy.array([[[1970.0, 28.0],[ 2169.0, 28.0],[ 2169.0, 227.0],[ 1970.0, 227.0]]], dtype=numpy.float32)
    }, (2200,1700), (234,1471,0,2200))
}
#cropped = img[start_row:end_row, start_col:end_col]

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
#capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
#possible = capture.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY)
#print(possible)
i = 0
do_crop = args.crop
do_resize = True
do_threshold = 0
do_blur = True
do_fullscreen = False
do_warp = False

if do_fullscreen:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ok, im = capture.read()
    if not ok: continue
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected_pts = detector.detectMarkers(im)
    if ids is None or not len(ids): continue
    detected = dict(zip(numpy.ravel(ids), corners)) 
    ok, matched = search_template(detected.keys())
    if not ok: 
        continue
    filename, markers, size, roi = matched
    i += 1
    pts_src = []
    pts_dst = []
    for id in markers.keys():
        pts_src.append(detected[id].reshape((4,2)))
        pts_dst.append(markers[id].reshape((4,2)))
        
    h, status = cv2.findHomography(numpy.concatenate(pts_src), numpy.concatenate(pts_dst))
    
    if do_warp:
        im = cv2.warpPerspective(im, h, size)
        if do_crop:
            #cropped = img[start_row:end_row, start_col:end_col]
            start_row, end_row, start_col, end_col = roi
            im = im[start_row:end_row, start_col:end_col]
    if do_resize:
        h, w = im.shape
        im = cv2.resize(im, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST_EXACT)
    #im = cv2.medianBlur(im,7)
    
    if do_blur:
        im = cv2.GaussianBlur(im, (7, 7), 0)
    if do_threshold:
        if 1 == do_threshold:
            (T, im) = cv2.threshold(im, 0, 255,	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        elif 2 == do_threshold:
            im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3)
    
    print(i, len(markers), filename, im.shape)
    cv2.imshow('image', im)
    k = cv2.waitKey(10) & 0XFF
    if k == 27:
        break
    elif k == ord('c'):
        do_crop = not do_crop
    elif k == ord('r'):
        do_resize = not do_resize
    elif k == ord('t'):
        do_threshold = (do_threshold + 1) % 3
    elif k == ord('b'):
        do_blur = not do_blur
    elif k == ord('w'):
        do_warp = not do_warp
    elif k == ord('f'):
        do_fullscreen = not do_fullscreen
        cv2.destroyWindow('image')
        if do_fullscreen:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
'''
im = cv.imread(cv.samples.findFile("lena.jpg"))
cv.namedWindow("foo", cv.WINDOW_NORMAL)
cv.setWindowProperty("foo", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cv.imshow("foo", im)
cv.waitKey()
cv.destroyWindow("foo")
'''
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
