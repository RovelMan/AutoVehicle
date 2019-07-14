import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_to_canny(image, kernel, minThresh, maxThresh):
  '''Transforming the image to a grayscale'''
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  '''Blurring it'''
  blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
  '''Detecting its edges'''
  canny = cv2.Canny(blur, minThresh, maxThresh)
  return canny

def region_of_interest(image):
  polygons = np.array([[(200, 650), (1200, 650), (650, 425)]])
  negPolygons = np.array([[(375, 650), (975, 650), (650, 450)]])
  # polygons = np.array([[(100, 530), (900, 530), (480, 300)]])
  # negPolygons = np.array([[(200, 530), (800, 530), (480, 330)]])
  mask = np.zeros_like(image)
  cv2.fillPoly(mask, polygons, 255)
  cv2.fillPoly(mask, negPolygons, 0)
  masked_image = cv2.bitwise_and(image, mask)
  return masked_image

def make_coordinates(image, line_parameters):
  slope, intercept = line_parameters
  y1 = image.shape[0]
  y2 = int(y1*(3/5.0))
  x1 = int((y1-intercept)/slope)
  x2 = int((y2-intercept)/slope)
  return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
  global last_left_fit_avg, last_right_fit_avg
  left_fit = []
  right_fit = []
  left_fit_average = None
  right_fit_average = None

  for line in lines:
    x1, y1, x2, y2 = line.reshape(4)
    parameters = np.polyfit((x1, x2), (y1, y2), 1)
    slope = parameters[0]
    intercept = parameters[1]
    if slope < 0:
      left_fit.append((slope, intercept))
    else:
      right_fit.append((slope, intercept))

  # print(sum(last_left_fit_avg-np.average(left_fit, axis=0))>100 or sum(last_left_fit_avg-np.average(left_fit, axis=0))<-100)

  if len(left_fit) > 0:
    left_fit_average = np.average(left_fit, axis=0)
    last_left_fit_avg = left_fit_average
  else:
    # print("No left line found! Using last")
    left_fit_average = last_left_fit_avg

  if len(right_fit) > 0:
    right_fit_average = np.average(right_fit, axis=0)
    last_right_fit_avg = right_fit_average
  else:
    # print("No right line found! Using last")
    right_fit_average = last_right_fit_avg

  return np.array([make_coordinates(image, left_fit_average), make_coordinates(image, right_fit_average)])

def display_lines(image, lines):
  line_image = np.zeros_like(image)
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line.reshape(4)
      cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
  return line_image

cap = cv2.VideoCapture("challenge.mp4")
last_left_fit_avg = None
last_right_fit_avg = None
last_averaged_lines = None
more_averaged_lines = []
five_averaged_lines = None

while(cap.isOpened()):
  _, frame = cap.read()

  '''Transforming the image'''
  canny = image_to_canny(frame, 5, 50, 150)

  '''Cropping the image to get region of interest'''
  cropped = region_of_interest(canny)
  # plt.imshow(cropped)
  # plt.show()

  '''Finding HoughLines in the cropped image'''
  hough_lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 100, minLineLength=40, maxLineGap=100)

  '''Finding the average lines of all the lines'''
  averaged_lines = None
  if hough_lines is not None:
    averaged_lines = average_slope_intercept(frame, hough_lines)
  else:
    averaged_lines = np.array([make_coordinates(frame, last_left_fit_avg), make_coordinates(frame, last_right_fit_avg)])
  
  '''Not using new averaged_lines that differ a lot from previous lines'''
  if last_averaged_lines is not None:
    diff = averaged_lines[0][0]-last_averaged_lines[0][0]
    if diff > 100 or diff < -100:
      averaged_lines = last_averaged_lines
  last_averaged_lines = averaged_lines

  more_averaged_lines.append(averaged_lines)
  
  if (len(more_averaged_lines) == 1):
    five_averaged_lines = averaged_lines
  
  if (len(more_averaged_lines) > 5):
    # print("list full", more_averaged_lines)
    five_averaged_lines = np.average(more_averaged_lines, axis=0).astype(int)
    more_averaged_lines = []

  '''Making an image containing only the lines'''
  line_image = display_lines(frame, five_averaged_lines)

  final = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

  '''Showing the image'''
  cv2.imshow("Lane Detection", final)
  cv2.waitKey(1)
