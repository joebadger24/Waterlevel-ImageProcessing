import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# Import image
x = cv2.imread("Photo_File_Path")
img = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
original = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

# Contrast
alpha = 1.1 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)
contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gaussian blur
kernel_size = 15
gaus_blur = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)


# Sobel distribution
scale = 1
delta = 0
ddepth = cv2.CV_16S
sbx = cv2.Sobel(gaus_blur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
sby = cv2.Sobel(gaus_blur, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

abs_sbx = cv2.convertScaleAbs(sbx)
abs_sby = cv2.convertScaleAbs(sby)

sobel = cv2.addWeighted(abs_sbx, 0.5, abs_sby, 0.5, 0)

# Canny
v = np.median(gaus_blur)
lower = int(max(0, (1.0 - 0.33) * v))
upper = int(min(255, (1.0 + 0.33) * v))

canny = cv2.Canny(gaus_blur,30,50)

# Hough transformation
hough1 = img.copy()
hough2 = img.copy()
height, width, channels = img.shape

#lines = cv2.HoughLinesP(sobel,1,np.pi/180,100,np.array([]),minLineLength=300,maxLineGap=10) j
lx1 = []
lx2 = []
ly1 = []
ly2 = []
avgy = []
avgx = []

jbx1 = []
jbx2 = []
jby1 = []
jby2 = []

lines2 = cv2.HoughLinesP(canny,rho = 1,theta = 1*np.pi/180,threshold = 150,minLineLength = 700,maxLineGap = 30)
for line in lines2:
    x1,y1,x2,y2 = line[0]
    cv2.line(hough2,(x1,y1),(x2,y2),(255,0,0),10)
    lx1.append(x1)
    ly1.append(y1)
    lx2.append(x2)
    ly2.append(y2)
    avgy.append((y2+y1)/2)
    avgx.append((x2+x1)/2)
    jbavgx = (x2+x1)/2
    if jbavgx > 550 and jbavgx < 615:
        jbx1.append(x1)
        jby1.append(y1)
        jbx2.append(x2)
        jby2.append(y2)


# this finds point of intersection

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
 
    # Method used to display X and Y coordinates
    # of a point
    def displayPoint(self, p):
        print(f"({p.x}, {p.y})")
 
 
def lineLineIntersection(A, B, C, D):
    # Line AB represented as a1x + b1y = c1
    a1 = B.y - A.y
    b1 = A.x - B.x
    c1 = a1*(A.x) + b1*(A.y)
 
    # Line CD represented as a2x + b2y = c2
    a2 = D.y - C.y
    b2 = C.x - D.x
    c2 = a2*(C.x) + b2*(C.y)
 
    determinant = a1*b2 - a2*b1
 
    if (determinant == 0):
        # The lines are parallel. This is simplified
        # by returning a pair of FLT_MAX
        return Point(10**9, 10**9)
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        return Point(x, y)
 
 
PN = 0

# Driver code
A = Point(jbx1[PN], jby1[PN])
B = Point(jbx2[PN], jby2[PN])
C = Point(0, 220)
D = Point(1000, 196)
 
intersection = lineLineIntersection(A, B, C, D)
 
if (intersection.x == 10**9 and intersection.y == 10**9):
    print("The given lines AB and CD are parallel.")
else:
    # NOTE: Further check can be applied in case
    # of line segments. Here, we have considered AB
    # and CD as lines
    #print("The intersection of the given lines AB " + "and CD is: ")
    intersection.displayPoint(intersection)

finalx1 = intersection.x
finaly1 = intersection.y
finalx2 = jbx2[PN]
finaly2 = jby2[PN]

diff_x = abs(finalx2 - finalx1)
diff_y = abs(finaly2 - finaly1)
distance = math.sqrt((diff_x * diff_x) + (diff_y * diff_y))

print(distance)


#////////////////////////////

cv2.line(hough1,(int(finalx1), int(finaly1)), (finalx2, finaly2), (255,0,0), 5)
cv2.line(hough1,(0, 220), (1000, 196), (0,0,255), 5)



# Plot results
plt.figure(1)
plt.subplot(331),plt.imshow(original, cmap ='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(332),plt.imshow(contrast,cmap = 'gray')
plt.title('Contrast'), plt.xticks([]), plt.yticks([])
plt.subplot(333),plt.imshow(gray,cmap = 'gray')
plt.title('Gray'), plt.xticks([]), plt.yticks([])
plt.subplot(334),plt.imshow(gaus_blur,cmap = 'gray')
plt.title('Gaussian'), plt.xticks([]), plt.yticks([])
plt.subplot(335),plt.imshow(canny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(336),plt.imshow(hough2,cmap = 'gray')
plt.title('Hough'), plt.xticks([]), plt.yticks([])

plt.figure(2)
plt.subplot(111),plt.imshow(hough2,cmap = 'gray')
plt.title('Hough'), plt.xticks([]), plt.yticks([])

plt.figure(3)
plt.subplot(111),plt.imshow(hough1,cmap = 'gray')
plt.title('Hough'), plt.xticks([]), plt.yticks([])

plt.show()



