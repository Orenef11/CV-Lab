# /usr/bin/env python
import cv2
from math import sqrt
from random import randrange
import numpy as np

""" Important variables for identifying vehicles in video!"""
''' The minimum area of vehicle'''
MIN_AREA_SIZE = 400
''' Aspect ratio is the proportion between the width and the height of a picture '''
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 4.0
''' The minimun of width and height of vehicle'''
MIN_WIDTH = 30
MIX_HEIGHT = 30
MIN_DIAGONAL = 60.0
RATIO_AREA = 0.5


def color_random_lottery():
    red_color = randrange(0, 255)
    green_color = randrange(0, 255)
    blue_color = randrange(0, 255)

    return red_color, green_color, blue_color


car_idx = 1
class Blob(object):
    """ Class which identifies the vehicles (cars and also motorcycles)
    In addition, with the functions that we eliminate noises and objects that not vehicles! """

    def __init__(self, contour):
        global car_idx
        self._id = car_idx
        car_idx += 1
        self._contour = contour
        x, y, w, h = cv2.boundingRect(contour)
        self._bounding_rect = cv2.boundingRect(contour)
        self._centroids_positions = [(x + w // 2, y + h // 2)]
        self._diagonal_size = sqrt(float(pow(w, 2) + pow(h, 2)))
        self._aspect_ratio = w / float(h)
        self._rect_area = float(w * h)
        self._color = color_random_lottery()

        """ Variable kalman algorithm """
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        kalman.statePre = np.array([[self._centroids_positions[0][0]], [self._centroids_positions[0][1]], [0], [5]],
                                   np.float32)
        kalman.statePost = np.array([[self._centroids_positions[0][0]], [self._centroids_positions[0][1]], [0], [5]],
                                    np.float32)
        self._kalman_position = kalman
        self.predicted_next_position = None
        self.predict()

    def checking_that_vehicles(self):
        ratio_area = cv2.contourArea(self._contour) / float(self._rect_area)

        if self._rect_area > MIN_AREA_SIZE and MIN_ASPECT_RATIO < self._aspect_ratio < MAX_ASPECT_RATIO\
                and self._bounding_rect[2] > MIN_WIDTH and self._bounding_rect[3] > MIX_HEIGHT and self._diagonal_size \
                > MIN_DIAGONAL and ratio_area > RATIO_AREA:
            return True
        return False

    """ Predict the next step, as the Kalman algorithm """
    def predict(self):
        prediction = self._kalman_position.predict()
        self.predicted_next_position = prediction[0][0], prediction[1][0]

    def update(self, vehicle):
        # self._contour = vehicle.contour
        self._bounding_rect = vehicle.bounding_rect
        self._centroids_positions.append(vehicle.centroids_positions[0])
        # self._diagonal_size = vehicle.diagonal_size
        # self._aspect_ratio = vehicle.aspect_ratio
        # self._rect_area = vehicle.rect_area
        self._kalman_position.correct(np.array([[vehicle.centroids_positions[-1][0]],
                                                [vehicle.centroids_positions[-1][0]]], np.float32))
        self.predict()

    @property
    def contour(self):
        return self._contour

    @property
    def bounding_rect(self):
        return self._bounding_rect

    @property
    def centroids_positions(self):
        return self._centroids_positions

    @property
    def diagonal_size(self):
        return self._diagonal_size

    @property
    def aspect_ratio(self):
        return self._aspect_ratio

    @property
    def rect_area(self):
        return self._rect_area

    @property
    def color(self):
        return self._color

    @property
    def id(self):
        return self._id

    """ For debug, print all the parameters of the object. Created for QA and debug """
    def to_string(self):
        print("contour - ", self.contour)
        print("bounding_rect - ", self.bounding_rect)
        print("center_positions - ", self.center_positions)
        print("diagonal_size - ", self.diagonal_size)
        print("aspect_ratio - ", self.aspect_ratio)
        print("rect_area - ", self.rect_area)
        print("still_being_tracked - ", self.still_being_tracked)
        print("current_match_found_or_new_Vehicle - ", self.current_match_found_or_new_Vehicle)
        print("num_of_consecutive_frames_without_a_match - ", self.num_of_consecutive_frames_without_a_match)
        print("predicted_next_position - ", self.predicted_next_position)