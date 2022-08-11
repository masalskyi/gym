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
videos = []
for video in args.inp:
    print(os.path.exists(video))
    videos.append(read_video(os.path.abspath(video)))

FRAME_WIDTH = videos[0][0].shape[1]
FRAME_HEIGHT = videos[0][0].shape[0]
out = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'MP4V'), args.fps, (FRAME_WIDTH, FRAME_HEIGHT))
for video in videos:
    for img in video:
        out.write(img)
out.release()