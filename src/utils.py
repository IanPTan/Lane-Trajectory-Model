import cv2
import numpy as np
import torch as pt
import onnxruntime as ort
from torch.distributions import Normal
import matplotlib.pyplot as plt
from time import time


def show_path(img, paths):
    plt.imshow(img[0], origin="lower")

    for path in paths:
        waypoints = np.cumsum(np.append(np.zeros(2), path).reshape(-1, 2), axis=-2) * 10 + np.array([64, 0])
        plt.plot(*waypoints.T)

    plt.show()


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


class TwinLiteNet:

    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)

    def __call__(self, images):
        imgs = []
        for image in images
            img, ratio, pad = letterbox_for_img(image)
            img = img.astype(np.float32).transpose(2, 0, 1)[::-1, ...] / 255
            imgs.append(img)

        imgs = np.array(imgs, dtype=np.float32)
        lane_segs, line_segs = self.ort_session.run(None, {"image": imgs})
        crop = slice(int(pad[1]), int(img.shape[2] - pad[1]))
        lane_segs = lane_segs[..., crop, :]
        line_segs = line_seg[..., crop, :]
        return lane_segs, line_segs


class SAL:

    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)

    def forward(self, image):
        mean, std = self.ort_session.run(None, {"image": image})
        dist = Normal(pt.tensor(mean), pt.tensor(std))
        return dist

    def __call__(self, image, samples=1):
        dist = self.forward(image)

        paths = []
        for sample in range(samples):
            paths.append(dist.sample())

        return np.array(paths)


def bird_eye(image, size, horizon, ratio, margins=100):

    image = image[horizon:]
    height, width, _ = image.shape

    left = (width - ratio * width) // 2
    right = width - left

    src_points = np.float32([
        [left, 0],
        [right, 0],
        [width, height],
        [0, height],
    ])

    dst_points = np.float32([
        [margins, margins],
        [size[0] - margins, margins],
        [size[0] - margins, size[1]],
        [margins, size[1]],
    ])

    output_width, output_height = size

    M = cv2.getPerspectiveTransform(src_points, dst_points)

    birds_eye_mask = cv2.warpPerspective(
        image,
        M,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST
    )

    return birds_eye_mask


def isolate(feature_map, seed_point=None, thresh=0.3):

    sharp_map = ((feature_map > thresh) * 255).astype(np.uint8)

    h, w = sharp_map.shape
    obj_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    if seed_point == None:
        seed_point = (int(w / 2), h - 1)

    cv2.floodFill(sharp_map, obj_mask, seed_point, 255, 10, 10, flags=cv2.FLOODFILL_MASK_ONLY)

    obj_mask = obj_mask[1:-1, 1:-1]
    obj_map = feature_map * obj_mask / obj_mask.max()

    return obj_map


def h_pinch(feature_map, seed_point=None, thresh=0.1, stride=1):

    h, w = feature_map.shape

    if seed_point:
        x, y = seed_point
    else:
        x = int(w / 2)
        y = h - 1

    left = x
    right = x
    while feature_map[y, left] > thresh or feature_map[y, right] > thresh:

        if feature_map[y, left] > thresh:
            left -= stride

        if feature_map[y, right] > thresh:
            right += stride

    center = int((left + right) / 2)

    return left, center, right


def get_slope(pixels, start, scope=None):

    h, w = pixels.shape

    if scope == None:
        scope = min(h, w)

    start = np.array(start)

    mass = np.array(np.where(pixels)).T

    distances = np.linalg.norm(mass - start, axis=-1)
    #weights = np.exp(-distances / scope)
    weights = (distances < scope) * 1.

    mean = np.sum(mass * weights[..., None], axis=0) / weights.sum()

    std = np.sqrt(np.sum((mass - mean) ** 2 * weights[..., None], axis=0) / (weights.sum() - 1))
    
    r = np.sum(np.prod((mass - mean) / std, axis=-1) * weights) / (weights.sum() - 1)

    direction = std
    direction[1] *= r
    direction /= np.linalg.norm(direction)

    """
    plt.scatter(*mass[distances < scope].T)
    line = np.array([mean, mean - direction * 100])
    plt.plot(*line.T, color="orange")
    plt.show()
    """

    return direction


def trace_lane(feature_map, start, num_lines=16, step_size=None, look_ahead=None):

    h, w = feature_map.shape

    start = np.array(start, dtype=np.float64)
    lane_points = [start]

    if step_size == None:
        length = np.where(feature_map.sum(axis=-1) > 0)[0][0]
        step_size = int(length / num_lines)

    if look_ahead == None:
        look_ahead = step_size

    curr_point = start.copy()
    direction = np.array([1., 0.])
    for l in range(num_lines):
        direction = -get_slope(feature_map > 0, curr_point + direction * look_ahead, 100)
        curr_point += direction * step_size
        #print(direction)
        lane_points.append(curr_point)

    return np.array(lane_points)


def preprocess(img_matrix, final_width=30, final_size=(256, 128)):
    
    img_8bit = (img_matrix[::-1] * 255).astype(np.uint8)
    initial_h, initial_w = img_8bit.shape
    
    left, center, right = h_pinch(img_matrix, (int(initial_w / 2), initial_h - 1))
    width = right - left
    
    scale_factor = final_width / width
    scaled_w = int(initial_w * scale_factor)
    scaled_h = int(initial_h * scale_factor)
    scaled_img = cv2.resize(img_8bit, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    scaled_center = int(center * scale_factor)
    final_h, final_w = final_size
    shift_x = int(final_w / 2) - scaled_center


    scaled_x = max(0, -shift_x)
    final_x = max(0, shift_x)
    shared_end = min(final_w, scaled_w + shift_x)
    shared_w = shared_end - final_x
    
    final_img = np.zeros(final_size, dtype=np.float32)
    final_img[:scaled_h, final_x:scaled_w + final_x] = scaled_img[:final_h, scaled_x:final_w + scaled_x] / 255
    
    return final_img


def pred_path(images, twin, sal):

    lane_seg_probs, _ = twin(images)

    pov_lane_segs = (lane_seg_probs[:, 1] > 0.1) * 1.

    input_imgs = []
    for pov_lane_seg in pov_lane_segs:
        lane_seg = bird_eye(pov_lane_seg[..., None], (400, 800), 195, 0.09, 100)
        lane_map = isolate(lane_seg)
        input_img = preprocess(lane_map)
        input_imgs.append(input_img)
    input_imgs = np.array(input_imgs, dtype=np.float32)[:, None, ...]

    paths = sal(input_imgs, 100)

    return input_imgs, paths


if __name__ == "__main__":
    twin = TwinLiteNet("twin.onnx")
    sal = SAL("sal.onnx")

    image = cv2.imread("road.jpg")

    plt.imshow(image[..., ::-1])
    plt.show()

    start = time()

    dur = time() - start

    plt.imshow(pov_lane_seg)
    plt.show()

    plt.imshow(lane_seg)
    plt.show()

    plt.imshow(lane_map)
    plt.show()

    show_path(input_img[None, ...], paths)
