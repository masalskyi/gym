import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='video input1, video input2, output video.')
parser.add_argument("input1", help="Path to 1 input video file")
parser.add_argument("input2", help="Path to 2 input video file")
parser.add_argument("output_file", help="Path to output video file")
parser.add_argument("--fps", dest="fps", default=60, help="Frames per second for ouput video", type=float)
parser.add_argument("--hstack", default=False, action="store_true")

args = parser.parse_args()
def read_video(file):
    images = []
    cap = cv2.VideoCapture(file)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        images.append(img)
    return images

if not args.hstack:
    images1 = read_video(args.input1)
    images2 = read_video(args.input2)

    if images1[0].shape[1] != images2[0].shape[1]:
        print("Input videos must have the same width")
        exit(-1)
    FRAME_WIDTH = images1[0].shape[1]
    FRAME_HEIGHT = images1[0].shape[0] + images2[0].shape[0]
    out = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'MP4V'), args.fps, (FRAME_WIDTH, FRAME_HEIGHT))
    for i in range(min(len(images1), len(images2))):
        image = np.vstack([images1[i], images2[i]])
        out.write(image)
    out.release()
else:
    images1 = read_video(args.input1)
    images2 = read_video(args.input2)

    if images1[0].shape[0] != images2[0].shape[0]:
        print("Input videos must have the same width")
        exit(-1)
    FRAME_WIDTH = images1[0].shape[1] + images2[0].shape[1]
    FRAME_HEIGHT = images1[0].shape[0]
    out = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'MP4V'), args.fps, (FRAME_WIDTH, FRAME_HEIGHT))
    for i in range(min(len(images1), len(images2))):
        image = np.hstack([images1[i], images2[i]])
        out.write(image)
    out.release()