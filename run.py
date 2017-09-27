# coding=utf-8
# /usr/bin/env python
"""
==================================================================================================================
Welcome to the program that detects cars witho the Kalman's algorithm!.
In addition, the program also attempts to count the number of cars that pass the "line"
Running the program is simple and easy, but it is important to use only MP4 format video files.
If not, we've seen many warnings due to the format (x264), but still the program will work properly and smoothly.

The code supports Python version 2.7 & 3.5

Information we used to build the project(From stackoverflow's website:
    http://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue?rq=1
    http://stackoverflow.com/questions/29012038/is-there-any-example-of-cv2-kalmanfilter-implementation
==================================================================================================================
"""
from check_modules_and_variable_Initialization import *
from tracker import *
import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from copy import copy
from pickle import load, dump
from os import path, makedirs, remove
from traceback import print_exc


""" Color codes I use them a lot """
WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)

""" CV & Python version by which the code is written """
MY_PYTHON_VERSION = r"2.7.11 Or 3.5.2 anaconda's version"
MY_CV_VERSION = r"3.1.0"

THRESHOLD = 30
DEBUG_MODE = False
MIN_DISTANCE = 1000


def draw_and_show_contours(image_shape, contours_list, window_name):
    image = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(image, contours_list, -1, WHITE_COLOR, -1)
    cv2.imshow(window_name, image)


def update_vehicle_by_frame(all_blobs_list, current_frame_blobs_list):
    new_all_blobs_list = []
    for current_frame_blob in current_frame_blobs_list:
        index_of_minimum_distance = 0
        minimum_distance = MIN_DISTANCE
        for idx, vehicle in enumerate(all_blobs_list):
            x = current_frame_blob.centroids_positions[-1]
            if x[0] == 419 and x[1] == 424:
                if idx == 2:
                    print()

            distance = euclidean(current_frame_blob.centroids_positions[-1], vehicle.centroids_positions[-1])
            if distance < minimum_distance:
                minimum_distance = distance
                index_of_minimum_distance = idx

        if float(minimum_distance) < current_frame_blob.diagonal_size * 0.5:
            all_blobs_list[index_of_minimum_distance].update(current_frame_blob)
            new_all_blobs_list.append(all_blobs_list[index_of_minimum_distance])
        else:
            new_all_blobs_list.append(current_frame_blob)

    if len(current_frame_blobs_list) == 0:
        return all_blobs_list
    del all_blobs_list
    return new_all_blobs_list


def draw_blob_info_on_image(all_blobs_list, frame):
    for vehicle_idx, vehicle in enumerate(all_blobs_list):
        # if vehicle.still_being_tracked is True:
        x, y, w, h = vehicle.bounding_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), vehicle.color, 2)
        car_center = vehicle.centroids_positions[-1]
        cv2.circle(frame, car_center, 1, RED_COLOR, 2)
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        if DEBUG_MODE:
            car_id = str(vehicle_idx)
        else:
            car_id = str(vehicle.id)
        cv2.putText(frame, car_id, car_center, font_face, 1, GREEN_COLOR, 2)


def check_blob_crossed(all_blob_list, horizontal_line_position, cars_count, direction):
    least_one_blob_crossed_the_line = False

    for blob in all_blob_list:
        index = len(blob.centroids_positions)
        if not blob.cross_line and index >= 4:
            prev_frame_index = index - 2
            current_frame_index = index - 1

            if blob.centroids_positions[prev_frame_index][1] > horizontal_line_position >= \
                    blob.centroids_positions[current_frame_index][1] and direction == 0:
                cars_count += 1
                least_one_blob_crossed_the_line = True
                blob.cross_line = True
            elif blob.centroids_positions[prev_frame_index][1] < horizontal_line_position <= \
                    blob.centroids_positions[current_frame_index][1] and direction == 1:
                cars_count += 1
                least_one_blob_crossed_the_line = True
                blob.cross_line = True

    return least_one_blob_crossed_the_line, cars_count


def draw_cars_count_on_image(cars_count, frame):
    font_face = cv2.FONT_HERSHEY_SIMPLEX

    bottom_left_text_position = (frame.shape[1] - 1 - 84, 30)
    cv2.putText(frame, str(cars_count), bottom_left_text_position, font_face, 1, RED_COLOR, 2)


def main():
    """ Print the doc on this project """
    print(__doc__)

    # Create the 'Output' folder
    if not path.isdir('Output'):
        makedirs('Output')

    """ Get arguments from command-line"""
    if DEBUG_MODE:
        with open('args.pickle', 'rb') as f:
            args = load(f)
        args.direction = 1
        args.speed = 1
        args.video_path = 'dataset/Highway1.avi'

    else:
        args = argument_parser()
        with open('args.pickle', 'wb') as fp:
            dump(args, fp)

    """ Check if arguments are valid """
    vehicle_direction, car_speed = check_valid_arguments(args)

    """ Checking if the versions of Python and CV same versions as I was used """
    check_version_of_python_and_cv(cv2.__version__)

    """ Creates all the variables necessary for running the program """
    video_output, capture_video, img_frame1, blobs_list, crossing_line, car_count, horizontal_line_position = \
        create_and_initialization_of_variables(args)

    first_loop_flag = True
    stop_loop = False
    digit_cv_version = cv2.__version__.split(".")[0]
    img_idx = 1
    # images_temp_folder_path = path.join('Output', path.split(args.video_path)[0])
    try:
        while capture_video.isOpened() and not stop_loop:
            end_of_video, img_frame2 = capture_video.read()
            if end_of_video is False:
                break

            current_frame_blobs_list = []
            img_frame1_copy = copy(img_frame1)
            img_frame2_copy = copy(img_frame2)

            """ Change the frame to Gray color """
            img_frame1_copy = cv2.cvtColor(img_frame1_copy, cv2.COLOR_BGR2GRAY)
            img_frame2_copy = cv2.cvtColor(img_frame2_copy, cv2.COLOR_BGR2GRAY)

            img_frame1_copy = cv2.GaussianBlur(img_frame1_copy, (5, 5), 0)
            img_frame2_copy = cv2.GaussianBlur(img_frame2_copy, (5, 5), 0)

            img_difference = cv2.absdiff(img_frame1_copy, img_frame2_copy)

            _, img_thresh = cv2.threshold(img_difference, THRESHOLD, 255, cv2.THRESH_BINARY)
            cv2.imshow("Threshold", img_thresh)

            structuring_elements5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))
            for i in range(2):
                img_thresh = cv2.dilate(img_thresh, structuring_elements5x5, None, (-1, -1), 1, cv2.BORDER_DEFAULT,
                                        (0, 0, 0))
                img_thresh = cv2.dilate(img_thresh, structuring_elements5x5, None, (-1, -1), 1, cv2.BORDER_DEFAULT,
                                        (0, 0, 0))
                img_thresh = cv2.erode(img_thresh, structuring_elements5x5, None, (-1, -1), 1, cv2.BORDER_DEFAULT,
                                       (0, 0, 0))

            img_thresh_copy = copy(img_thresh)
            if digit_cv_version == "3":
                _, contours, _ = cv2.findContours(img_thresh_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            elif digit_cv_version == "2":
                contours, _ = cv2.findContours(img_thresh_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            draw_and_show_contours(img_thresh.shape, contours, "Contours")
            convex_hulls_list = [cv2.convexHull(contours[i]) for i in range(contours.__len__())]
            draw_and_show_contours(img_thresh.shape, convex_hulls_list, "Convex Hulls")

            for i in range(convex_hulls_list.__len__()):
                blobs = Blob(convex_hulls_list[i])
                if blobs.checking_that_vehicles() is True:
                    current_frame_blobs_list.append(blobs)

            blobs_contours_list = [blob.contour for blob in current_frame_blobs_list]
            draw_and_show_contours(img_thresh.shape, blobs_contours_list, "Current Frame Blobs")

            if first_loop_flag is False:
                blobs_list = update_vehicle_by_frame(blobs_list, current_frame_blobs_list)
            else:
                blobs_list = list(current_frame_blobs_list)
                first_loop_flag = False

            img_frame2_copy = copy(img_frame2)
            draw_blob_info_on_image(blobs_list, img_frame2_copy)
            least_one_blob_crosses_the_line, car_count = \
                check_blob_crossed(blobs_list, horizontal_line_position, car_count, vehicle_direction)

            if least_one_blob_crosses_the_line is True:
                cv2.line(img_frame2_copy, crossing_line[0], crossing_line[1], GREEN_COLOR, 2)
            else:
                cv2.line(img_frame2_copy, crossing_line[0], crossing_line[1], RED_COLOR, 2)

            draw_cars_count_on_image(car_count, img_frame2_copy)
            cv2.imshow("Real Frame", img_frame2_copy)
            # Creates the color output video with vehicle identification
            cv2.imwrite('temp.jpg', img_frame2_copy, [cv2.IMWRITE_JPEG_QUALITY, 90])
            video_output.write(cv2.imread('temp.jpg'))
            remove('temp.jpg')
            img_idx += 1
            img_frame1 = copy(img_frame2)
            del current_frame_blobs_list
            """ Do not delete this loop !!!
                It is important to run the capture of the frame """
            k = 0xFF & cv2.waitKey(car_speed)
            if k == ord('p') or k == ord('P'):
                while 1:
                    k = 0xFF & cv2.waitKey(1)

                    if k == ord('p') or k == ord('P'):
                        break
                    if k == 27 or k == ord('q') or k == ord('Q'):
                        stop_loop = True
                        break
            if k == 27 or k == ord('q') or k == ord('Q'):
                break
    except Exception as e:
        print(e)
        print(print_exc)

    print(car_count)
    capture_video.release()
    video_output.release()
    cv2.destroyAllWindows()
    print("End of program!")


if __name__ == "__main__":
    main()
