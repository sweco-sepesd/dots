import argparse
from PIL import Image, ImageFilter
import cv2
import numpy
import uuid

#Create uuid
user_uuid = uuid.uuid4()

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
def main():
    # Initialize the camera
    capture = cv2.VideoCapture(1)  # 0 är laptopkameran, 1 är första externa
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

        if do_crop:
            #cropped = img[start_row:end_row, start_col:end_col]
            start_row, end_row, start_col, end_col = roi
            # im = im[start_row:end_row, start_col:end_col]  Ändrad från im till mask på bägge
            mask = mask[start_row:end_row, start_col:end_col]
        if do_resize:
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


        if do_mask == True:
            cv2.imshow('image', mask)  #replace mask <-> im    
        else:
            cv2.imshow('image', im)  #replace mask <-> im
        #cv2.imshow('image', mask)  #replace mask <-> im
        
        k = cv2.waitKey(10) & 0XFF
       
          
        if k == ord('p'):
            img_name = "{}.png".format(user_uuid)
            cv2.imwrite(img_name, mask)
            img_counter += 1
            break
        



    #print (user_uuid)
    return user_uuid
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
