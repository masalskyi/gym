import os.path
import pathlib

import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='video input, image input, output path')
parser.add_argument("input_video", help="Path to input video file", type=pathlib.Path)
parser.add_argument("text", help="Path to input video file")
parser.add_argument("--x", type=int, default=0)
parser.add_argument("--y", type=int, default=30)
parser.add_argument("--output_video", default="./result.mp4", help="Path to input video file", type=pathlib.Path)

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


images = read_video(os.path.abspath(args.input_video))
FRAME_WIDTH = images[0].shape[1]
FRAME_HEIGHT = images[0].shape[0]
fps = 30
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
video = cv2.VideoCapture(os.path.abspath(args.input_video))
video.open(os.path.abspath(args.input_video))
if int(major_ver) < 3:
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
else:
    fps = video.get(cv2.CAP_PROP_FPS)

print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
out = cv2.VideoWriter(os.path.abspath(args.output_video), cv2.VideoWriter_fourcc(*'MP4V'), fps, (FRAME_WIDTH, FRAME_HEIGHT))

for i in range(len(images)):
    cv2.putText(images[i], args.text, (args.x, args.y), cv2.FONT_ITALIC, 0.8, (0, 0, 0),
                thickness=2)
    out.write(images[i])
out.release()
