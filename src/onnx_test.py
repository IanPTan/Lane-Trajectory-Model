import onnxruntime as ort
import cv2
import numpy as np
from time import time
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from utils import *




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default='pretrained/large.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    opt = parser.parse_args()

    model = TwinLiteNet(opt.weight)

    raw_img = cv2.imread(opt.source, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # BGR

    outputs = model(raw_img)

    cv2.imwrite("breh.png", outputs[0][0][1] * 255)
    plt.imshow(outputs[0][0][1])
    plt.show()

