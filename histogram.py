import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils import paths


def histopogram(folder, number):
    imagePath = "dataset/{}/{}.{}.jpg".format(folder, folder, number)

    def compute_hist(img):
        hist = np.zeros((256,), np.uint8)
        h, w = img.shape[:2]
        for i in range(h):
            for j in range(w):
                hist[img[i][j]] += 1
        return hist

    def equal_hist(hist):
        cumulator = np.zeros_like(hist, np.float64)
        for i in range(len(cumulator)):
            cumulator[i] = hist[:i].sum()

        new_hist = (
            (cumulator - cumulator.min()) / (cumulator.max() - cumulator.min()) * 255
        )
        new_hist = np.uint8(new_hist)
        return new_hist

    img = cv2.imread(imagePath)
    hist = compute_hist(img).ravel()
    new_hist = equal_hist(hist)
    print(imagePath)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            img[i, j] = new_hist[img[i, j]]
    cv2.imwrite("dataset/{}/{}h.jpg".format(folder, number), img)
