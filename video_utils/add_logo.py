import cv2
import numpy as np
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='video input, image input, output path')
parser.add_argument("input_video", help="Path to input video file")
parser.add_argument("input_image", help="Path to input video file")
parser.add_argument("output_file", help="Path to output video file")
parser.add_argument("--logo_width", default=80, type=int)
parser.add_argument("--logo_height", default=120, type=int)

args = parser.parse_args()


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()
    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))
    mask = a
    h, w, _ = overlay_color.shape
    roi = bg_img[y:y + h, x:x + w]
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)
    return bg_img


video = cv2.VideoCapture(args.input_video)
images = []
while video.isOpened():
    ret, img = video.read()
    if not ret:
        break
    images.append(img)
FRAME_WIDTH = images[0].shape[1]
FRAME_HEIGHT = images[0].shape[0]
fps = 30
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver) < 3:
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
else:
    fps = video.get(cv2.CAP_PROP_FPS)

print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
out = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, (FRAME_WIDTH, FRAME_HEIGHT))

logo = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)
logo = cv2.resize(logo, (args.logo_width,args.logo_height))
for i in range(len(images)):
    image = overlay_transparent(images[i], logo, FRAME_WIDTH-args.logo_width, FRAME_HEIGHT-args.logo_height)
    out.write(image)
out.release()

