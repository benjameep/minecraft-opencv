import numpy as np
import cv2 as cv

class Contour(np.ndarray):
    def __new__(self, input_array):
        obj = np.asarray(input_array).view(self)
        obj.next = None
        obj.prev = None
        obj.child = None
        obj.parent = None
        return obj
    
    def area(self):
        return cv.contourArea(self)
    
    def arcLength(self, closed=True):
        return cv.arcLength(self, closed)
    
    def approx_poly(self, percent=0.1, closed=True):
        epsilon = percent*self.arcLength(closed)
        return cv.approxPolyDP(self, epsilon, closed).view(Contour)
    
    def convex_hull(self, clockwise=True, returnPoints=True):
        return cv.convexHull(self, clockwise, returnPoints).view(Contour)
    
    def bounding_rect(self):
        return cv.boundingRect(self)
