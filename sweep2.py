import cv2
import math
import os.path
import numpy as np
from args import d_args
from scipy.spatial.transform import Rotation as R

""" PARAMETERS """
# Contour size parameters for contour extraction
cont_s1 = d_args.cont_s1
cont_s2 = d_args.cont_s2

# 'Directionality' parameter of the contours. The higher the parameter value, the less spread out the direction changes
# of the pixels in a contour
cont_d1 = d_args.cont_d1
cont_d2 = d_args.cont_d2

# Filter size for filtering the outliers
filter_size = d_args.filter_size

""" IMAGE CORRECTION """
# Load floor plan
print("Name of the floor plan:")
file_name = input()
image = cv2.imread(file_name)
dot_idx = file_name.find('.')
output_name = file_name[:dot_idx]

# Perform Gamma correction
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mid = 0.5
mean = np.mean(gray)
gamma = math.log(mid * 255) / math.log(mean)
print(f'Gamma correction with gamma value {gamma}')
image = np.power(image, gamma).clip(0, 255).astype(np.uint8)

image = cv2.bilateralFilter(image, 15, 50, 50)
scale_percent = 60  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)





square = []


def select_square(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(square) < 4:
        print(f'Corner added at coordinate: {x, y}')
        global c_x, c_y, wall
        c_x, c_y = x, y
        square.append([x, y])

cv2.namedWindow('Extracted walls', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Extracted walls', image)
print("Click on corner points of a rectangle in the image. Clock-wise, starting from left upper corner. ")
cv2.setMouseCallback('Extracted walls', select_square)

esc = 0
while esc == 0:
    cv2.imshow('Extracted walls', image)
    esc = cv2.waitKey() & 0xFF

if len(square) < 4:
    square = [[0, 0], [len(image)-1, 0], [len(image)-1, len(image[0])-1], [0, len(image[0])-1]]

cv2.destroyAllWindows()
image2 = image.copy()
rect = cv2.minAreaRect(np.array(square))
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(image2, [box], 0, (0, 0, 255), 2)

x, y, w, h = cv2.boundingRect(box)
cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
resquare = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
pts1 = np.float32(square)
pts2 = np.float32(resquare)
M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
image = dst










c_x, c_y = 0, 0
wall = []

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200)


def get_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Color added at coordinate: {x, y}')
        global c_x, c_y, wall
        c_x, c_y = x, y
        wall.append([x, y])


# Obtain colors of walls in the floor plan
cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image', image)
print("Click on walls in the image")
cv2.setMouseCallback('image', get_color)

esc = 0
while esc == 0:
    cv2.imshow('image', image)
    esc = cv2.waitKey() & 0xFF

cv2.destroyAllWindows()

""" WALL EXTRACTION """
print("Extracting contours")
try:
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
except:
    contours = []
upd_c = []
for c in contours:
    if len(c) > cont_s1:
        vecx = []
        vecy = []
        d = dict()
        for i in range(1, len(c)):
            vx = c[i][0][1] - c[i - 1][0][1]
            vy = c[i][0][0] - c[i - 1][0][0]
            av = abs(vx) + abs(vy)
            vx = vx / av
            vy = vy / av
            vecx.append(vx)
            vecy.append(vy)
            if str([vx, vy]) in d:
                d[str([vx, vy])] += 1
            else:
                d[str([vx, vy])] = 1
        sorted_val = []
        for n in d:
            d[n] = d[n] / len(vecx)
            sorted_val.append(d[n])
        sorted_val.sort()
        sorted_val.reverse()
        if sum(sorted_val[:4]) > cont_d2:
            upd_c.append(c)

contours = np.array(upd_c, dtype=object)

# Draw extracted contours from original floor plan
contour_image = image.copy()
try:
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
except:
    print('Could not extract contours directly from the floor plan')

print("Extracting wall colors")
# Extract the color of the operator designated walls
colors = []
for w in wall:
    colors.append(image[w[1]][w[0]])

masks = []
for c in colors:
    min_c = np.array((c[0] - 10, c[1] - 10, c[2] - 10))
    max_c = np.array((c[0] + 10, c[1] + 10, c[2] + 10))
    masks.append(cv2.inRange(image, min_c, max_c))

mask = masks[0]
for m in masks:
    mask = mask | m
mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
image = cv2.bitwise_and(image, mask3)
image = cv2.bilateralFilter(image, 15, 50, 50)

mask_image = image.copy()

print("Extracting edge pixels")
# Extracting edges from the original floor plan and comparing if their colour matches with the wall colours
edge_pix = []
for i in range(len(edges)):
    for j in range(len(edges[0])):
        if edges[i][j] > 0:
            s = np.array([0, 0, 0])
            s = s + np.array(image[i][j])
            try:
                for ii in range(-2, 3):
                    for jj in range(-2, 3):
                        s = s + np.array(image[i + ii][j + jj])
                        if sum(s) > 0:
                            break
            except:
                pass

            if sum(s) > 0:
                edge_pix.append([j, i])

pixels = []
for cont in contours:
    for c in cont:
        s = np.array([0, 0, 0])
        s = s + np.array(image[c[0][1]][c[0][0]])
        try:
            for i in range(-2, 3):
                for j in range(-2, 3):
                    s = s + np.array(image[c[0][1] + i][c[0][0] + j])
                    if sum(s) > 0:
                        break
        except:
            pass

        if sum(s) > 0:
            pixels.append([c[0][1], c[0][0]])

image[:] = [255, 255, 255]

for p in edge_pix:
    image[p[1]][p[0]] = [0, 0, 0]

mask_image2 = image.copy()

print("Extracting edge pixel contours")
# Extracting contours from the obtained edge pixels
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200)

try:
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
except:
    contours = []

upd_c = []
for c in contours:
    if len(c) > cont_s2:
        vecx = []
        vecy = []
        d = dict()
        for i in range(1, len(c)):
            vx = c[i][0][1] - c[i - 1][0][1]
            vy = c[i][0][0] - c[i - 1][0][0]
            av = abs(vx) + abs(vy)
            vx = vx / av
            vy = vy / av
            vecx.append(vx)
            vecy.append(vy)
            if str([vx, vy]) in d:
                d[str([vx, vy])] += 1
            else:
                d[str([vx, vy])] = 1
        sorted_val = []
        for n in d:
            d[n] = d[n] / len(vecx)
            sorted_val.append(d[n])
        sorted_val.sort()
        sorted_val.reverse()
        if sum(sorted_val[:4]) > cont_d2:
            upd_c.append(c)

contours = np.array(upd_c, dtype=object)

contour_image2 = image.copy()
cv2.drawContours(contour_image2, contours, -1, (0, 255, 0), 3)

scale_percent = 50
width = int(contour_image.shape[1] * scale_percent / 100)
height = int(contour_image.shape[0] * scale_percent / 100)
dim = (width, height)

# Displaying the results of each wall extraction step
contour_image = cv2.resize(contour_image, dim, interpolation=cv2.INTER_AREA)
mask_image = cv2.resize(mask_image, dim, interpolation=cv2.INTER_AREA)
contour_image2 = cv2.resize(contour_image2, dim, interpolation=cv2.INTER_AREA)
mask_image2 = cv2.resize(mask_image2, dim, interpolation=cv2.INTER_AREA)
hori1 = np.concatenate((contour_image, mask_image), axis=1)
hori2 = np.concatenate((contour_image2, mask_image2), axis=1)

vert = np.concatenate((hori1, hori2), axis=0)

esc = 0
while esc == 0:
    cv2.imshow('Mask contours', vert)
    esc = cv2.waitKey() & 0xFF
cv2.destroyAllWindows()

print("Calculating initial uncorrected walls")

pixels2 = []
for cont in contours:
    for c in cont:
        pixels2.append([c[0][1], c[0][0]])

image[:] = [255, 255, 255]

for p in pixels:
    image[p[0]][p[1]] = [0, 0, 0]
    try:
        image[p[0] + 1][p[1]] = [0, 0, 0]
        image[p[0] - 1][p[1]] = [0, 0, 0]
        image[p[0]][p[1] + 1] = [0, 0, 0]
        image[p[0]][p[1] - 1] = [0, 0, 0]
    except:
        pass

for p in pixels2:
    image[p[0]][p[1]] = [0, 0, 0]
    try:
        image[p[0] + 1][p[1]] = [0, 0, 0]
        image[p[0] - 1][p[1]] = [0, 0, 0]
        image[p[0]][p[1] + 1] = [0, 0, 0]
        image[p[0]][p[1] - 1] = [0, 0, 0]
    except:
        pass


def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]
    new_image = image.copy()

    for label in range(num_stats):
        if stats[label, cv2.CC_STAT_AREA] < filter_size:
            new_image[labels == label] = 0

    return new_image


# Clean up image from lone pixels
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

image = remove_isolated_pixels(gray)
image = cv2.bitwise_not(image)

cv2.imwrite(output_name + '_walls.jpg', image)

square = []


# def select_square(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN and len(square) < 4:
#         print(f'Corner added at coordinate: {x, y}')
#         global c_x, c_y, wall
#         c_x, c_y = x, y
#         square.append([x, y])
#
#
# """ REFINING RESULTS """
# cv2.namedWindow('Extracted walls', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('Extracted walls', image)
# # Straightening the obtained image. Clock-wise, starting from left upper corner
# print("Click on corner points of a rectangle in the image. Clock-wise, starting from left upper corner. ")
# cv2.setMouseCallback('Extracted walls', select_square)
#
# esc = 0
# while esc == 0:
#     cv2.imshow('Extracted walls', image)
#     esc = cv2.waitKey() & 0xFF
#
# if len(square) < 4:
#     square = [[0, 0], [len(image)-1, 0], [len(image)-1, len(image[0])-1], [0, len(image[0])-1]]
#
# cv2.destroyAllWindows()
# image2 = image.copy()
# rect = cv2.minAreaRect(np.array(square))
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(image2, [box], 0, (0, 0, 255), 2)
#
# x, y, w, h = cv2.boundingRect(box)
# cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
# resquare = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
# pts1 = np.float32(square)
# pts2 = np.float32(resquare)
# M = cv2.getPerspectiveTransform(pts1, pts2)
#
# dst = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

dst = image
eraser = False
drawer = False
radius = 10
draw_r = 2


def erase(x, y):
    cv2.circle(dst, (x, y), radius, (255, 255, 255), -1)
    cv2.imshow('Result', dst)


def draw(x, y):
    cv2.circle(dst, (x, y), 2, (0, 0, 0), -1)
    cv2.imshow('Result', dst)


def handleMouseEvent(event, x, y, flags, param):
    global eraser, drawer
    if event == cv2.EVENT_MOUSEMOVE:
        if eraser:
            erase(x, y)
        elif drawer:
            draw(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        eraser = False
    elif event == cv2.EVENT_LBUTTONDOWN:
        eraser = True
        erase(x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        drawer = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawer = True
        draw(x, y)


# manually correct the image. Wall pixels can be deleted with the left mouse click.
# They can be added with the right mouse click.
print('Correct the Image. Left click - erase. Right click - draw.')
cv2.namedWindow('Result', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Result', image)
cv2.setMouseCallback('Result', handleMouseEvent)

esc = 0
while esc == 0:
    cv2.imshow('Result', dst)
    esc = cv2.waitKey() & 0xFF
cv2.destroyAllWindows()

cv2.imwrite(output_name + '_walls_cor.jpg', dst)

""" CREATING A ROS MAP """


def measure(event, x, y, flags, param):
    global mx, my
    if event == cv2.EVENT_LBUTTONDOWN and len(mx) < 2:
        print(f'Point added at coordinate: {x, y}')
        mx.append(x)
        my.append(y)

        if len(mx) > 1:
            print("What is the distance in meters between the 2 points?")
            deltapx = float(input())
            dpx = math.sqrt((mx[1] - mx[0]) ** 2 + (my[1] - my[0]) ** 2)
            global resolution
            resolution = deltapx / dpx
            print(f'Calculated pixel resolution is: {resolution} meters')


image = dst
resolution = d_args.resolution
mx = []
my = []
# Measure the pixel size by manually setting a distance between given pixels
cv2.namedWindow('Measure Distance', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Measure Distance', image)
print("Click on two points in the image to measure distance")
cv2.setMouseCallback('Measure Distance', measure)

esc = 0
while esc == 0:
    cv2.imshow('Measure Distance', image)
    esc = cv2.waitKey() & 0xFF

cv2.destroyAllWindows()

o_x = d_args.originx
o_y = d_args.originy
d_x = 0
d_y = 0

draw_l = False


def draw_line(x, y, x2, y2):
    image_line = image.copy()
    image_line = cv2.cvtColor(image_line, cv2.COLOR_GRAY2RGB)
    cv2.line(image_line, (x, y), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Show Origin', image_line)


def get_origin(event, x, y, flags, param):
    global draw_l, o_x, o_y, d_x, d_y
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Origin added at coordinate: {x, y}')
        o_x, o_y = x, y
        draw_l = True
    elif event == cv2.EVENT_LBUTTONUP:
        print(f'Direction towards coordinate: {x, y}')
        d_x, d_y = x, y
        draw_l = False

    if draw_l:
        draw_line(o_x, o_y, x, y)


cv2.namedWindow('Show Origin', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Show Origin', image)
print("Select point of origin")
cv2.setMouseCallback('Show Origin', get_origin)

esc = 0
while esc == 0:
    cv2.imshow('Show Origin', image)
    esc = cv2.waitKey() & 0xFF

cv2.destroyAllWindows()

vector_1 = [1, 0]
vector_2 = [d_x - o_x, d_y - o_y]
unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
dot_product = np.dot(unit_vector_1, unit_vector_2)
angle = np.arccos(dot_product)

if vector_2[1] > 0:
    angle = -angle

vec = [-o_x * resolution, -(image.shape[0] - o_y) * resolution, 1]
print(f'Pose angle: {angle}')
rotation_radians = -angle
rotation_axis = np.array([0, 0, 1])
rotation_vector = rotation_radians * rotation_axis
rotation = R.from_rotvec(rotation_vector)
rotated_vec = rotation.apply(vec)

mapName = output_name + '_ros'

mapLocation = ''
completeFileNameMap = os.path.join(mapLocation, mapName + ".pgm")
completeFileNameYaml = os.path.join(mapLocation, mapName + ".yaml")
yaml = open(completeFileNameYaml, "w")
cv2.imwrite(completeFileNameMap, image)

# Writing the YAML file for the ROS map. ROS has a bug where the costmap does not rotate
# together with the map from the given origin pose. Therefore, we set the origin pose as 0.00, but save
# the calculated one as orig_angle for later use, if necessary. Additionally, saving parameters for
# scaling factors and biases for each axis.
yaml.write("image: " + mapName + ".pgm\n")
yaml.write("resolution: " + str(resolution) + "\n")
yaml.write("origin: [" + str(rotated_vec[0]) + "," + str(rotated_vec[1]) + "," + "0.00" + "]\n")
yaml.write("negate: 0\noccupied_thresh: " + str(d_args.occupied) + "\nfree_thresh: " + str(d_args.free) + "\n")
yaml.write("orig_angle: " + str(rotation_radians) + "\n")
yaml.write("x_scale: " + str(d_args.x_scale) + "\n")
yaml.write("x_bias: " + str(d_args.x_bias) + "\n")
yaml.write("y_scale: " + str(d_args.y_scale) + "\n")
yaml.write("y_bias: " + str(d_args.y_bias) + "\n")
yaml.close()
