import argparse
import pathlib

import cv2
import numpy

template_by_filename = {
    'template.jpg': ({
          98: numpy.array([[[1970.0, 1474.0],[ 2169.0, 1474.0],[ 2169.0, 1673.0],[ 1970.0, 1673.0]]], dtype=numpy.float32)
        , 62: numpy.array([[[236.0, 1671.0],[ 37.0, 1671.0],[ 37.0, 1472.0],[ 236.0, 1472.0]]], dtype=numpy.float32)
        , 203: numpy.array([[[46.0, 35.0],[ 245.0, 35.0],[ 245.0, 234.0],[ 46.0, 234.0]]], dtype=numpy.float32)
        , 23: numpy.array([[[1970.0, 28.0],[ 2169.0, 28.0],[ 2169.0, 227.0],[ 1970.0, 227.0]]], dtype=numpy.float32)
    }, (2200,1700), (231,1474,246,1968))
}

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
parser.add_argument('-c', '--crop', action='store_true', default=True)
parser.add_argument('src_image', type=pathlib.Path)
parser.add_argument('dst_image', type=pathlib.Path)
args = parser.parse_args()

detector = cv2.aruco.ArucoDetector()
detector.setDictionary(args.dictionary)

im = cv2.imread(args.src_image, cv2.IMREAD_GRAYSCALE)


'''
corners	vector of detected marker corners. For each marker, its four corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of this array is Nx4. The order of the corners is clockwise.
ids	vector of identifiers of the detected markers. The identifier is of type int (e.g. std::vector<int>). For N detected markers, the size of ids is also N. The identifiers have the same order than the markers in the imgPoints array.
rejectedImgPoints	contains the imgPoints of those squares whose inner code has not a correct codification. Useful for debugging purposes.
'''
corners, ids, rejected_pts = detector.detectMarkers(im)
#print (corners)

detected = dict(zip(numpy.ravel(ids), corners)) #(numpy.ravel(c).tolist() for c in corners)))


ok, matched = search_template(detected.keys())
if not ok: 
    raise Exception('no template matched')
filename, markers, size, roi = matched
pts_src = []
pts_dst = []
for id in markers.keys():
    pts_src.append(detected[id].reshape((4,2)))
    pts_dst.append(markers[id].reshape((4,2)))
    
h, status = cv2.findHomography(numpy.concatenate(pts_src), numpy.concatenate(pts_dst))


im_dst = cv2.warpPerspective(im, h, size)
if args.crop:
    #cropped = img[start_row:end_row, start_col:end_col]
    start_row, end_row, start_col, end_col = roi
    im_dst = im_dst[start_row:end_row, start_col:end_col]
    
    
im_dst = cv2.medianBlur(im_dst,5)

im_dst = cv2.adaptiveThreshold(im_dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            
cv2.imwrite(args.dst_image, im_dst)


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
