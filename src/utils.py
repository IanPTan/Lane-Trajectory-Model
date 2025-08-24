import cv2
import numpy as np

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image = cv2.imread('test.jpg')
    projected_mask = bird_eye(image, (400, 800), 390, 0.09, 100)

    plt.imshow(image)
    plt.show()

    plt.imshow(projected_mask)
    plt.show()
