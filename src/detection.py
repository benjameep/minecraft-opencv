from .rectangle import Rectangle

def largest_rectangle(rectangles):
    sizes = [rect.area for rect in rectangles]
    return rectangles[sizes.index(max(sizes))]

def absorb_intersecting(target, rectangles):
    target = target.copy()
    intersecting_found = True
    while intersecting_found:
        intersecting_found = False
        not_used = []
        for other in rectangles:
            if target.intersects(other):
                intersecting_found = True
                target += other
            else:
                not_used.append(other)
        rectangles = not_used
    return target

def find_module(frame):
    blurred = frame.median_blur(19)
    selected_colors = blurred.hsv_range((0,0,190),(255,5,210)) | blurred.hsv_range((0,0,125),(255,10,145))
    stretched = selected_colors.dilate(10)
    contours = stretched.find_contours(mode='external')
    rectangles = [Rectangle.from_contour(contour, mode='bound') for contour in contours]
    if len(contours):
        rect = largest_rectangle(rectangles)
        rect = absorb_intersecting(rect, rectangles)
        dist = abs(rect.cx-frame.width/2) + abs(rect.cy-frame.height/2)
        dist /= frame.width+frame.height
        size = (rect.width+rect.height) / (frame.width+frame.height)
        if 0.2 < size < 0.5 and dist < 0.01:
            return rect