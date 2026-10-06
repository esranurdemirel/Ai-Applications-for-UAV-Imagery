import cv2
import numpy as np


class Camera:

    def __init__(self):

        # Kalibrasyonun yapıldığı çözünürlük
        self.calib_width = 1920
        self.calib_height = 1080

        self.K = np.array([
            [1389.7,    0.0, 954.007],
            [0.0,    1387.1, 558.896],
            [0.0,       0.0,    1.0]
        ], dtype=np.float64)

        self.dist = np.array([
            0.1378,
            -0.2564,
            0.0,
            0.0,
            0.0
        ], dtype=np.float64)

    def get_camera_matrix(self, width, height):

        sx = width / self.calib_width
        sy = height / self.calib_height

        K = self.K.copy()

        K[0, 0] *= sx      # fx
        K[1, 1] *= sy      # fy

        K[0, 2] *= sx      # cx
        K[1, 2] *= sy      # cy

        return K

    def undistort(self, image):

        h, w = image.shape[:2]

        K = self.get_camera_matrix(w, h)

        return cv2.undistort(
            image,
            K,
            self.dist
        )
    def camera_matrix(self, image):

        h, w = image.shape[:2]

        return self.get_camera_matrix(w, h)