import argparse
import os

import numpy as np
import cv2

parser = argparse.ArgumentParser(description='video input1, video input2, output video.')
parser.add_argument("input1", help="Path to 1 input video file")
parser.add_argument("input2", help="Path to 2 input video file")
parser.add_argument("output_file", help="Path to output video file")
parser.add_argument("--fps", dest="fps", default=60, help="Frames per second for ouput video", type=float)

args = parser.parse_args()


cap1 = cv2.VideoCapture(os.path.abspath(args.input1))
cap1.open(os.path.abspath(args.input1))

cap2 = cv2.VideoCapture(os.path.abspath(args.input2))
cap2.open(os.path.abspath(args.input2))

ret1, img1 = cap1.read()
ret2, img2 = cap2.read()

FRAME_WIDTH = img1.shape[1]
FRAME_HEIGHT = img1.shape[0] + img2.shape[0]
out = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'MP4V'), args.fps, (FRAME_WIDTH, FRAME_HEIGHT))
while True:
    image = np.vstack([img1, img2])
    out.write(image)
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()
    if not (ret1 and ret2):
        break
out.release()