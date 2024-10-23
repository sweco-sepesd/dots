import argparse
from PIL import Image, ImageFilter
import cv2
import numpy
import uuid

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
parser = argparse.ArgumentParser(description='Find markers in image')


def get_dictionary(string):
    return cv2.aruco.getPredefinedDictionary(dictionaries[string])

