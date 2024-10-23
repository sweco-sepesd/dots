from flask import Flask, render_template, Response
import cv2
import argparse
import numpy
from detectandshow5 import get_dictionary, search_template
from PIL import Image, ImageFilter
import uuid

app = Flask(__name__)

#Create uuid
user_uuid = None
print ("created", user_uuid)

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
# Initialize the webcam
# Capture frame-by-frame



def generate_frames():
    global user_uuid
    user_uuid = uuid.uuid4()
    print ("in function", user_uuid)
    camera = cv2.VideoCapture(0)
    
    
    i=0
    do_crop = args.crop
    do_resize = False
    do_threshold = False
    do_blur = False
    do_photo = True
    do_mask = True
    do_smooth = True
    img_counter = 0
    while True:
        
        ok, im = camera.read()
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
        
        if not ok:
            continue
        # Creates a mask containing green pixels
        mask = cv2.inRange(hsvImage, lowerColor, upperColor)
        
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected_pts = detector.detectMarkers(im)
        if ids is None or not len(ids):
            continue
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
            
        
        
        k = cv2.waitKey(10) & 0XFF
      
      
        if k == ord('p'):
            img_name = "{}.png".format(user_uuid)
            cv2.imwrite(img_name, mask)
            print(f"Image saved as {img_name}")
            img_counter += 1
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', mask)
            mask = buffer.tobytes()  # Convert to bytes

            # Yield the frame in a format suitable for the HTML img tag
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + mask + b'\r\n')
        
    print (user_uuid)
    #return user_uuid
    camera.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('video.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4996)
