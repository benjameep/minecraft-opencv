import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from .contour import Contour

class Frame(np.ndarray):
    def __new__(self, input_array):
        obj = np.asarray(input_array).view(self)
        return obj
    
    @property
    def width(self): return self.shape[1]
    
    @property
    def height(self): return self.shape[0]
    
    @classmethod
    def from_file(self, filename):
        img = cv.imread(filename)
        assert img is not None, 'Bad File'
        return img.view(Frame)
    
    def resize(self, dsize=None, percent=None, interpolation='nearest'):
        # nearest, linear, area, nearest, cubic, lanczos4
        interpolation = getattr(cv, 'INTER_'+interpolation.upper())
        if percent is not None:
            dsize = (int(self.width*percent), int(self.height*percent))
        return cv.resize(self, dsize, interpolation=interpolation).view(Frame)
    
    def gaussian_blur(self,*args,**kwargs):
        return cv.GaussianBlur(self,*args,**kwargs).view(Frame)
    
    def median_blur(self,*args,**kwargs):
        return cv.medianBlur(self,*args,**kwargs).view(Frame)
    
    def bilateral_filter(self, d=9, sigmaColor=75, sigmaSpace=None):
        if sigmaSpace is None:
            sigmaSpace = sigmaColor
        return cv.bilateralFilter(self, d, sigmaColor, sigmaSpace).view(Frame)
    
    def thresh(self, low, high, typ):
        typ = getattr(cv, 'THRESH_'+typ.upper())
        ret,thresh = cv.threshold(self,low,high,typ)
        return thresh.view(Frame)
    
    def canny(self, minVal, maxVal=None):
        maxVal = maxVal if maxVal is not None else minVal
        return cv.Canny(self, minVal, maxVal).view(Frame)
    
    def invert(self):
        return cv.bitwise_not(self).view(Frame)
    
    def gray(self):
        return cv.cvtColor(self, cv.COLOR_BGR2GRAY).view(Frame)
    
    def hsv_range(self, low, high):
        return cv.inRange(self.as_hsv(), low, high).view(Frame)
    
    def cvt_color(self, color):
        color = getattr(cv, 'COLOR_'+color.upper())
        return cv.cvtColor(self, color).view(Frame)
    
    def as_hsv(self):
        return self.cvt_color('bgr2hsv')
    
    def match_template(self, template, method):
        # ccoeff, ccoeff_normed, ccorr, ccorr_normed, sqdiff, sqdiff_normed
        method = getattr(cv, 'TM_'+method.upper())
        return cv.matchTemplate(self, template, method).view(Frame)
    
    def minMaxLoc(self):
        return cv.minMaxLoc(self)
    
    def apply_mask(self, mask):
        mask = mask>0
        masked = np.zeros_like(self,np.uint8)
        masked[mask] = self[mask]
        return masked.view(Frame)
    
    def erode(self, size=3, shape='rect', anchor=(-1,-1), iterations=1, kernel=None):
        if kernel is None:
            if type(size) != tuple:
                size = (size, size)
            # rect, cross, ellipse
            shape = getattr(cv, 'MORPH_'+shape.upper())
            kernel = cv.getStructuringElement(shape, size)
        return cv.erode(self, kernel=kernel, anchor=anchor, iterations=iterations).view(Frame)
    
    def dilate(self, size=3, shape='rect', anchor=(-1,-1), iterations=1, kernel=None):
        if kernel is None:
            if type(size) != tuple:
                size = (size, size)
            # rect, cross, ellipse
            shape = getattr(cv, 'MORPH_'+shape.upper())
            kernel = cv.getStructuringElement(shape, size)
        return cv.dilate(self, kernel=kernel, anchor=anchor, iterations=iterations).view(Frame)
        
    def hough_lines(self, rho=1, theta=2, threshold=0, minLineLength=20, maxLineGap=1):
        return cv.HoughLinesP(self, 
                              rho=rho, 
                              theta=np.pi/theta, 
                              threshold=threshold, 
                              minLineLength=minLineLength, 
                              maxLineGap=maxLineGap,
                             )
    
    def draw_text(self, text, org, fontScale=1, color=(255,255,255), fontFace='plain', thickness=1, lineType='line_8', bottomLeftOrigin=False, replace=False):
        # https://docs.opencv.org/4.5.2/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11
        # simplex, plain, duplex, complex, triplex, complex_small, script_simplex, script_complex, italic
        fontFace = fontFace.upper()
        if 'FONT_'+fontFace in dir(cv):
            fontFace = getattr(cv, 'FONT_'+fontFace)
        else:
            fontFace = getattr(cv, 'FONT_HERSHEY_'+fontFace)
        
        # https://docs.opencv.org/4.5.2/d6/d6e/group__imgproc__draw.html#gaf076ef45de481ac96e0ab3dc2c29a777
        # filled, line_4, line_8, line_aa
        lineType = getattr(cv, lineType.upper())
        
        img = self if replace else self.copy()
        
        cv.putText(img, text=text, org=org, fontScale=fontScale, fontFace=fontFace, color=color, thickness=thickness, lineType=lineType, bottomLeftOrigin=bottomLeftOrigin)
        
        return img
        
    
    def draw_lines(self, lines, color=(0,255,0), thickness=1, replace=False):
        img = self if replace else self.copy()
        for line in lines[:,0]:
            cv.line(img, line[:2], line[2:], color, thickness)
        return img
    
    def find_contours(self, mode='tree', method='simple'):
        # external, list, ccomp, tree, floodfill
        mode = getattr(cv,'RETR_'+mode.upper())
        # none, simple, tc89_l1, tc89_kcos
        method = getattr(cv, 'CHAIN_APPROX_'+method.upper())
        
        raw_contours, hiearchy = cv.findContours(self, mode, method)
        contours = [Contour(contour) for contour in raw_contours]
        if mode == 'tree':
            for c,(nxt,prev,child, parent) in zip(contours, hiearchy[0]):
                c.next = contours[nxt] if nxt != -1 else None
                c.prev = contours[prev] if prev != -1 else None
                c.child = contours[child] if child != -1 else None
                c.parent = contours[parent] if parent != -1 else None
        return contours

    # For list of colormaps    
    # https://docs.opencv.org/4.5.2/d3/d50/group__imgproc__colormap.html#ga9a805d8262bcbe273f16be9ea2055a65
    def apply_colormap(self, colormap):
        colormap = getattr(cv, 'COLORMAP_'+colormap.upper())
        return cv.applyColorMap(self, colormap).view(Frame)

    def draw_contour(self, contour, *args, **kwargs):
        return self.draw_contours([contour], 0, *args, **kwargs)

    def draw_contours(self, contours, index=-1, color=(0,255,0), thickness=1, replace=False):
        img = self if replace else self.copy()
        cv.drawContours(img, contours, index, color, thickness)
        return img
    
    def draw_rectangle(self, rect, color=(0,255,0), thickness=1, replace=False):
        img = self if replace else self.copy()
        cv.rectangle(img, rect.tl, rect.br, color, thickness)
        return img
    
    def draw_rectangles(self, rectangles, color=(0,255,0), thickness=1, replace=False):
        img = self if replace else self.copy()
        for rect in rectangles:
            cv.rectangle(img, rect.tl, rect.br, color, thickness)
        return img
    
    def write(self, filename):
        cv.imwrite(filename, self)
        return self
    
    def __repr__(self,*args):
        if(len(self.shape) == 3 and self.shape[2] == 3):
            plt.imshow(self[:,:,::-1])
            return '<Frame {}>'.format(self.shape)
        elif(len(self.shape) == 2):
            plt.imshow(self, cmap='gray')
            return '<Frame {}>'.format(self.shape)
        return super(self).__repr__(self,*args)
