import cv2
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    image = cv2.imread('line.png')[..., 0] / 255
    points = np.array(np.where(image > 0)).T
    seed = np.array([486, 196])
    scope = 40

    distances = np.linalg.norm(points - seed, axis=-1)
    #weights = np.exp(-distances / scope)
    weights = (distances < 100) * 1

    std = np.sqrt(np.sum((points - seed) ** 2 * weights[..., None], axis=0) / (weights.sum() - 1))

    r = np.sum(np.prod((points - seed) / std, axis=-1) * weights) / (weights.sum() - 1)

    direction = std
    direction[1] *= r
    direction /= np.linalg.norm(direction)

    line = np.array([seed, seed+direction*100])

    #plt.imshow(image)
    plt.scatter(*points[weights > 0.1].T)
    plt.plot(*line.T, color="orange")
    plt.show()

    plt.scatter(*points.T)
    #plt.plot(*line.T, color="orange")
    plt.show()
