import sys
import time

import numpy as np
import cv2

KNOWN_CAMERA_RESOLUTIONS = [
    [
         (1280, 960) #  4:3 
        ,( 320, 240) #  4:3 
        ,( 960, 720) #  4:3 
        ,( 640, 480) #  4:3 
        ,( 160, 120) #  4:3 
        ,( 352, 288) #  11:9
        ,( 176, 144) #  11:9
     ],
     [
         ( 640, 360) # 16:9 
        ,(2560,1440) # 16:9 
        ,(1920,1080) # 16:9 
        ,(1280, 720) # 16:9 
        ,(2560,1920) #  4:3 
        ,( 640, 480) #  4:3 
     ]
]

# four point transform
def order_points(pts):
    # https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")
def orderpts(pts):
    # pts.mean(0) is centerpoint
    # dx,dy to centerpoint is greater than 0 is given weight 1 for x and 2 for y
    # 
    ordering = ((pts - pts.mean(0) > 0) * [1,2]).sum(1)
    tl, tr, bl, br = pts[np.argsort(ordering)]
    return np.array([tl, tr, br, bl], dtype="float32")
def four_point_transform(image, pts, dst):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    rect = orderpts(pts)
    #print(rect)
    (tl, tr, br, bl) = rect  
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
def find_and_draw_contours(im, min_rectangularity=0.8, min_covering=0.3, ksize=5):
    h,w = im.shape[:2]
    full_area = float(h*w)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # Otsu's thresholding after Gaussian filtering
    blurred = cv2.GaussianBlur(gray,(ksize,ksize),0)
    ok,binarized = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(im, contours, -1, (0,255,0), 3)
    results = []
    for cnt in contours:
        contour_area = cv2.contourArea(cnt)
        epsilon = 0.1 * cv2.arcLength(cnt,True)
        if contour_area / full_area < min_covering:
            #print(contour_area , full_area, contour_area / full_area)
            continue
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        if len(approx) == 4:
            approx = approx.reshape(4, 2)
            results.append((contour_area, approx))
            #print('success', approx)
            cv2.polylines(im,[approx],True,(0,255,0),2)
        else:
            pass #print('not used', approx.shape)
            cv2.drawContours(im,[cnt],0,(0,0,255),2)   # draw contours in red color
        rect = cv2.minAreaRect(cnt)             # rect = ((center_x,center_y),(width,height),angle)
        ((center_x,center_y),(width,height),angle) = rect
        rect_area = width * height
        if not rect_area:
            continue
        if contour_area / rect_area < min_rectangularity:
            continue
        points = cv2.boxPoints(rect)         # Find four vertices of rectangle from above rect
        #points = np.int8(np.around(points))     # Round the values and make it integers
        points = np.array(points, np.int32)
        #print(points)
        points = points.reshape((-1,1,2))
        #ellipse = cv2.fitEllipse(cnt)           # ellipse = ((center),(width,height of bounding rect), angle)

        #cv2.ellipse(im,ellipse,(0,0,255),2)        # draw ellipse in red color
        cv2.polylines(im,[points],True,(255,0,0),2)# draw rectangle in blue color
    return results, binarized

def warp(src,dst):
    im = cv2.imread(src)
    results, binarized = find_and_draw_contours(im)
    if len(results):
        area, pts = sorted(results)[-1] # sorted on area, choosing largest
        warped = four_point_transform(im, pts)
        cv2.imwrite(dst, warped)
        
    return 0

def main():
    camera_index = 0
    if 2 == len(sys.argv):
        camera_index = int(sys.argv[1])
    elif 4 == len(sys.argv):
        if 'warp' == sys.argv[1]:
            sys.exit(warp(sys.argv[2], sys.argv[3]))
    #print('Testing camera ', camera_index)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    time.sleep(2)
    res_index = 0
    ksize = 3
    grain_extract_ksize = 39
    #A3: 297 x 420, A4 210 x 297, A6 105 x 148
    dst_width = 1188
    dst_height = 840
    dst_pts = np.array([
              [dst_width,dst_height]
            , [        0,dst_height]
            , [        0,   0]
            , [dst_width,   0]
        ], dtype = "float32")
    resolutions = KNOWN_CAMERA_RESOLUTIONS[camera_index]
    w,h = resolutions[res_index]
    ok = cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) and cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not ok:
        print('Failed to set resolution', w,h)
        
    print('Camera is open:', cap.isOpened())
    if not cap.isOpened():
        cap.release()
        return 1
    settings_fmt = '{} - res: {}x{}, ksize: {}'
    settings = settings_fmt.format('Settings', *resolutions[res_index], ksize)
    while True:
        ok, im = cap.read()
        if not ok:
            break
        new_settings = settings_fmt.format('Settings', *resolutions[res_index], ksize)
        if settings != new_settings:
            print(new_settings)
            settings = new_settings
        results, binarized = find_and_draw_contours(im, min_rectangularity=0.8, min_covering=0.3, ksize=ksize)
        if len(results):
            area, pts = sorted(results)[-1] # sorted on area, choosing largest
            ordering = ((pts - pts.mean(0) > 0) * [1,2]).sum(1)
            tl, tr, bl, br = pts[np.argsort(ordering)]
            src_pts = np.array([tl, tr, br, bl], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(im, M, (dst_width, dst_height))
            hsv_image = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            h_channel,s_channel,v_channel = cv2.split(hsv_image)
            blurred = cv2.GaussianBlur(v_channel,(grain_extract_ksize,grain_extract_ksize),0)
            #blurred = cv2.medianBlur(blurred, grain_extract_ksize)
            cv2.imshow('blurred', blurred)
            grain_extracted = np.array(np.clip((np.array(v_channel, dtype="float32") - np.array(blurred, dtype="float32")/2) + 128, 0, 255), dtype="uint8")
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(grain_extracted)
            
            remounted = cv2.cvtColor(cv2.merge([h_channel,s_channel,equalized]), cv2.COLOR_HSV2BGR)
            
            cv2.imshow('warped', warped)
            s_channel= clahe.apply(s_channel)
            #ok,binarized_s_channel = cv2.threshold(s_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imshow('h_channel', h_channel)
            #cv2.imshow('grain_merged', grain_merged)
            #gray = cv2.cvtColor(grain_extracted, cv2.COLOR_BGR2GRAY)
            #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            # Otsu's thresholding after Gaussian filtering
            #blurred = cv2.GaussianBlur(gray,(ksize,ksize),0)
            #blurred = cv2.medianBlur(blurred, 7)
            #ok,binarized2 = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #cv2.imshow('warped', warped)
        cv2.imshow("image", im)
        cv2.imshow("binarized", binarized)
        key_code = cv2.waitKey(100) & 0xFF
        if key_code in [66,98]:
            #B, b
            ksize = max(1, ksize + [-2,2][key_code==66])
        if key_code == ord('r'):
            res_index = max(0, res_index - 1)
            w,h = resolutions[res_index]
            ok = cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) and cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if not ok:
                print('Failed to set resolution', w,h)
            #print(res_index, w, h, cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif key_code == ord('R'):
            res_index = min(len(resolutions) - 1, res_index + 1)
            w,h = resolutions[res_index]
            ok = cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) and cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if not ok:
                print('Failed to set resolution', w,h)
            #print(res_index, w, h, cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif key_code == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    return 0
if '__main__' == __name__:
    sys.exit(main())
    