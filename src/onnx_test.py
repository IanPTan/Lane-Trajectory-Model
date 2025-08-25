import onnxruntime as ort
import cv2
import numpy as np
from time import time
from argparse import ArgumentParser
import matplotlib.pyplot as plt


def letterbox_for_img(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))


    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding

    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)




parser = ArgumentParser()
parser.add_argument('--weight', type=str, default='pretrained/large.pth', help='model.pth path(s)')
parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--config', type=str, choices=["nano", "small", "medium", "large"], help='Model configuration')
parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
opt = parser.parse_args()

ort_session = ort.InferenceSession(opt.weight)
raw_img = cv2.imread(opt.source, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
img, ratio, pad = letterbox_for_img(raw_img, new_shape=opt.img_size, auto=True)
img = img.astype(np.float32).transpose(2, 0, 1)[None, ::-1, ...] / 255
outputs = ort_session.run(None, {"image": img})

crop = slice(int(pad[1]), int(img.shape[2] - pad[1]))

cv2.imwrite("bre.png", img[0].transpose(1, 2, 0) * 255)
#plt.imshow(img[0].transpose(1, 2, 0)[crop])
#plt.show()

cv2.imwrite("breh.png", outputs[0][0][1][crop] * 255)
#plt.imshow(outputs[0][0][1][crop])
#plt.show()

#plt.imshow(outputs[1][0][1][crop])
#plt.show()

