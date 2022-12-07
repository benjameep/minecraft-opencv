import cv2 as cv
from .frame import Frame

class VideoReader(cv.VideoCapture):
    def __init__(self, filename, _slice=slice(None, None, None)):
        self.filename = filename
        self.slice = _slice
        super().__init__(filename)
    
    # pos_msec
    # pos_frames
    # pos_avi_ratio
    # frame_width
    # frame_height
    # frame_count
    # fps
    # fourcc
    # bitrate
        
    def __setattr__(self, key, value):
        prop_name = 'CAP_PROP_'+key.upper()
        if prop_name in cv.__dict__:
            return self.set(getattr(cv, prop_name), value)
        return super().__setattr__(key, value)
    
    def __getattribute__(self, key):
        prop_name = 'CAP_PROP_'+key.upper()
        if prop_name in cv.__dict__:
            return self.get(getattr(cv, prop_name))
        return super().__getattribute__(key)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return VideoReader(self.filename, key)
        else:
            if self.pos == self.length:
                self.open(self.filename)
            assert key in range(self.length)
            if key != self.pos:
                self.pos = key
            ret, frame = self.read()
            return frame.view(Frame)
        
    
    def __iter__(self):
        for i in range(self.length)[self.slice]:
            yield self[i]
    
    def __enter__(self):
        return self
    
    def __exit__(self,*args):
        self.release()

    @property
    def pos(self):
        return int(self.get(cv.CAP_PROP_POS_FRAMES))
    
    @pos.setter
    def pos(self, i):
        self.set(cv.CAP_PROP_POS_FRAMES, i)
    
    @property
    def width(self):
        return int(self.get(cv.CAP_PROP_FRAME_WIDTH))
    
    @property
    def height(self):
        return int(self.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    @property
    def length(self):
        return int(self.get(cv.CAP_PROP_FRAME_COUNT))
    
    @property
    def size(self):
        return (self.width, self.height)



class VideoWriter(cv.VideoWriter):
    def __init__(self, filename, fourcc, fps, size):
        self.filename = filename
        if type(fourcc) == str:
            fourcc = cv.VideoWriter_fourcc(*fourcc.upper())
        super().__init__(filename, fourcc, fps, size)
    
    def __enter__(self):
        assert self.isOpened()
        return self
    
    def __exit__(self, *args):
        self.release()
