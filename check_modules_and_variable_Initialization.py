# /usr/bin/env python
from sys import version
from argparse import ArgumentParser
import cv2
from os.path import isfile

""" CV & Python version by which the code is written """
MY_PYTHON_VERSION = r"2.7.11 Or 3.5.2 anaconda's version"
MY_CV_VERSION = r"3.1.0"

""" Default argument values for argument_parser function"""
VIDEO_PATH = r"dataset\1.mp4"
DIRECTION = 0
SPEED = 1
CAR_IDX = 1


def argument_parser():
    parser = ArgumentParser(description='')
    parser.add_argument('-v', '--video_path', type=str, help='The video file path', default=VIDEO_PATH)
    parser.add_argument('-d', '--direction', type=int,
                        help='Directing the traffic of vehicle. 1 marks movement'
                             ' from top to bottom, 0 marks a movement from bottom to top.', default=DIRECTION)
    parser.add_argument('-s', '--speed', type=int, help='The running speed of the frames.' \
                                                        'Number 1 mark a high speed, the Number 1000 marks a slow ' \
                                                        'speed. The The values are between 1-100', default=SPEED)
    parser.add_argument('-p', '--line_position', type=int, help='Line location on the photo, to count the vehicles.',
                        required=True)
    args = parser.parse_args()

    return args


def create_and_initialization_of_variables(args):
    capture_video = cv2.VideoCapture(args.video_path)

    if not capture_video.isOpened():
        exit("Video path incorrect Or cann't open this video format!")

    cv_major_ver = int(cv2.__version__.split('.')[0])
    if cv_major_ver == 3:
        width = capture_video.get(cv2.CAP_PROP_FPS)  # float
        height = capture_video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = capture_video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MPG4')

    else:
        width = capture_video.get(cv2.cv.CV_CAP_PROP_FPS)  # .cv.CV_CAP_PROP_FRAME_WIDTH)  # float
        height = capture_video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)  # .cv.CV_CAP_PROP_FRAME_HEIGHT)  # float
        fps = capture_video.get(cv2.cv.CV_CAP_PROP_FPS)
        fourcc = cv2.cv.CV_FOURCC(*'MPG4')

    fourcc, width, height, fps = (int(fourcc), int(width), int(height), int(fps))
    video_output = cv2.VideoWriter("output-1.mp4", fourcc, fps, (width, height))
    _, img_frame1 = capture_video.read()
    video_output.write(img_frame1)
    blobs_list = []
    crossing_line = []
    car_count = 0
    horizontal_line_position = int(round(img_frame1.shape[0] * args.line_position / 100, 2))
    crossing_line.append((0, horizontal_line_position))
    crossing_line.append(((img_frame1.shape[1] - 1), horizontal_line_position))

    return video_output, capture_video, img_frame1, blobs_list, crossing_line, car_count, horizontal_line_position


def check_exist_modules_libraries_check():
    if version[0] == 2 and version[2] == 7:
        from imp import find_module
        try:
            find_module('cv2')
        except ImportError:
            exit("You must install the openCV's library")
        try:
            find_module('numpy')
        except ImportError:
            exit("You must install the numpy's library")
    elif version[0] == 3:
        if version[2] <= 3:
            from importlib import find_loader
            try:
                find_loader('cv2')
            except ImportError:
                exit("You must install the openCV's library")
            try:
                find_loader('numpy')
            except ImportError:
                exit("You must install the numpy's library")
        elif version[2] >= 4:
            from importlib.util import find_spec
            try:
                find_spec('cv2')
            except ImportError:
                exit("You must install the openCV's library")
            try:
                find_spec('numpy')
            except ImportError:
                exit("You must install the numpy's library")


def check_version_of_python_and_cv(cv_version):
    python_version = version
    flag_change = 0

    cv_msg = ""
    python_msg = ""
    if cv_version != MY_CV_VERSION and python_version != MY_PYTHON_VERSION:
        cv_msg = "Your ~~CV version~~ ({0}) modified version with which the code is written ({1})" \
            .format(cv_version, MY_CV_VERSION)
        python_msg = "Your ~~Python version~~ ({0}) modified version with which the code is written ({1})" \
            .format(python_version, MY_PYTHON_VERSION)

        flag_change = 1
    elif cv_version != MY_CV_VERSION:
        cv_msg = "Your ~~CV version~~ ({0}) modified version with which the code is written ({1})" \
            .format(cv_version, MY_CV_VERSION)

        flag_change = 1
    elif python_version != MY_PYTHON_VERSION:
        python_msg = "Your ~~Python version~~ ({0}) modified version with which the code is written ({1})" \
            .format(python_version, MY_PYTHON_VERSION)

        flag_change = 1

    """ Checking the version of Python to know what print function to use """
    digit_python_version = python_version.split(".")[0]
    if digit_python_version == "2":
        if flag_change == 1:
            print("Warnings: Please note, could be a program does not work optimally because of"
                  " different versions you use")
        if cv_msg != "":
            print(cv_msg)
        if python_msg != "":
            print(python_msg)
    elif digit_python_version == "3":
        if flag_change == 1:
            print("Warnings: Please note, could be a program does not work optimally because of"
                  " different versions you use")
        if cv_msg != "":
            print(cv_msg)
        if python_msg != "":
            print(python_msg)


def check_valid_arguments(args):
    if args.direction != 1 and args.direction != 0:
        exit("Direction error: Entered value does not match the requirements")
    if 0 >= args.speed or args.speed > 100:
        exit("Speed error: Entered value does not match the requirements")
    if not isfile(args.video_path):
        exit("Video error: The path of video not correct")

    return args.direction, args.speed
