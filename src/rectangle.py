import numpy as np
import cv2 as cv

class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        assert x1 <= x2 and y1 <= y2
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    @classmethod
    def from_contour(self, contour, mode='avg'):
        if mode == 'bound':
            x,y,w,h = cv.boundingRect(contour)
            return self(x,y,x+w,y+h)
        elif mode == 'avg':
            assert len(contour) == 4
            xs = np.sort(contour[:,0,0]).view(np.ndarray)
            ys = np.sort(contour[:,0,1]).view(np.ndarray)
            return self(
                xs[:2].mean().astype(int),
                ys[:2].mean().astype(int),
                xs[2:].mean().astype(int),
                ys[2:].mean().astype(int),    
            )
        else:
            raise Exception('Unknown Mode',mode)
    
    def intersects(self, other):
        assert isinstance(other, Rectangle)
        return self.x1 < other.x2 and self.x2 > other.x1 and \
               self.y1 < other.y2 and self.y2 > other.y1
    
    def copy(self):
        return Rectangle(self.x1, self.y1, self.x2, self.y2)
    
    def __add__(self, other):
        assert isinstance(other, Rectangle)
        return Rectangle(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2)
        )
    
    def __eq__(self, other):
        assert isinstance(other, Rectangle)
        return self.x1 == other.x1 and \
               self.y1 == other.y1 and \
               self.x2 == other.x2 and \
               self.y2 == other.y2
    
    @property
    def width(self): return self.x2-self.x1

    @property
    def height(self): return self.y2-self.y1
    
    @property
    def perimeter(self): return self.width*2+self.height*2
    
    @property
    def area(self): return self.width*self.height
    
    @property
    def x(self): return self.x1
    
    @property
    def y(self): return self.y1
    
    @property
    def cx(self): return self.x1 + self.width//2
    
    @property
    def cy(self): return self.y1 + self.height//2
    
    @property
    def tl(self): return self.x1, self.y1
    
    @property
    def br(self): return self.x2, self.y2
    
    @property
    def bl(self): return self.x1, self.y2
    
    @property
    def tr(self): return self.x2, self.y1
    
    @property
    def shape(self): return self.width, self.height
    
    def __repr__(self):
        return 'Rect({}, {}, {}, {})'.format(self.x, self.y, self.width, self.height)
