import argparse
import pathlib
import os

import numpy as np
import cv2

parser = argparse.ArgumentParser(description='video inputs, video input2, output video.')
parser.add_argument("-i", "--inp", nargs="+", required=True, help="Path to input videos file", type=pathlib.Path)
parser.add_argument("-o", "--output_file", default="./video.mp4", help="Path to output video file")
parser.add_argument("--fps", dest="fps", default=60, help="Frames per second for output video", type=float)

args = parser.parse_args()


def read_video(file):
    images = []
    cap = cv2.VideoCapture(file)
    cap.open(file)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        images.append(img)
    return images


for video in args.inp:
    print(os.path.exists(video))

first_video = os.path.abspath(args.inp[0])
cap = cv2.VideoCapture(first_video)
cap.open(first_video)
ret, img = cap.read()

cap.release()

FRAME_WIDTH = img.shape[1]
FRAME_HEIGHT = img.shape[0]
out = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'MP4V'), args.fps, (FRAME_WIDTH, FRAME_HEIGHT))

for video in args.inp:
    video_file = os.path.abspath(video)
    cap = cv2.VideoCapture(video_file)
    cap.open(video_file)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        out.write(img)
    cap.release()
out.release()
