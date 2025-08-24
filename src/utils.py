import cv2
import numpy as np

def bird_eye_view_projection(image, mask, src_points, dst_points):
    """
    Projects a segmented mask from a camera's perspective to a bird's-eye view.

    Args:
        image (np.array): The original RGB or BGR image from the camera.
        mask (np.array): The segmented binary mask of the road.

    Returns:
        np.array: The projected bird's-eye view mask.
    """

    # Destination points for the bird's-eye view
    # These four points form a rectangle, "flattening" the perspective
    output_width, output_height = 800, 800

    # Calculate the homography matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transform to the mask
    birds_eye_mask = cv2.warpPerspective(
        mask,
        M,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST  # Use nearest-neighbor interpolation for masks
    )

    return birds_eye_mask

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example usage:

    # Manually defined source points from the camera's perspective
    # These four points form a trapezoid on the road
    """
    src_points = np.float32([
        [643, 651],  # Top-left corner of the road segment
        [758, 653],  # Top-right corner
        [773, 691], # Bottom-right corner
        [626, 690]   # Bottom-left corner
    ])
    """

    WIDTH = 1280
    HEIGHT = 720
    FEET = np.float32([WIDTH // 2, HEIGHT])
    dst_width = 20
    dst_height = 80
    src_size = 368
    top_size = 75
    forward = 273


    disp = np.float32([
        [-top_size, -forward],
        [top_size, -forward],
        [src_size, 0],
        [-src_size, 0],
    ])
    src_points = FEET + disp

    dst_points = np.float32([
        [400 - dst_width, 700 - dst_height],
        [400 + dst_width, 700 - dst_height],
        [400 + dst_width, 700 + dst_height],
        [400 - dst_width, 700 + dst_height],
    ])

    # Assuming you have an image and its corresponding binary mask
    original_image = cv2.imread('test.jpg')
    #segmented_mask = your_segmentation_function(original_image)
    segmented_mask = original_image
    projected_mask = bird_eye_view_projection(original_image, segmented_mask, src_points, dst_points)

    # To visualize:
    #cv2.imshow('Original Mask', segmented_mask)
    #cv2.imshow('Bird\'s-Eye View Mask', projected_mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    plt.imshow(segmented_mask)
    plt.plot(*src_points.T)
    plt.show()
    plt.imshow(projected_mask)
    plt.show()
