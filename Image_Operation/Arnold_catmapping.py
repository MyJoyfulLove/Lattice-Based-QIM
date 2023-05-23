import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity


def arnold(original: np.array, mat: np.array) -> np.array:
    rows, cols = original.shape
    processed = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            coefficient = np.dot(mat, np.array([row, col])) % rows
            processed[coefficient[0], coefficient[1]] = original[row, col]
    return processed


def inverse_arnold(arnolded: np.array, mat: np.array) -> np.array:
    rows, cols = arnolded.shape
    mat_inv = np.int32(np.linalg.inv(mat))
    reduction = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            coefficient_inv = np.dot(mat_inv, np.array([row, col])) % rows
            reduction[coefficient_inv[0], coefficient_inv[1]] = arnolded[row, col]
    return reduction


def psnr_gray(target, ref):
    diff = target - ref
    mse = np.mean(np.square(diff))
    psnr_valse = 10 * np.log10(255 * 255 / mse)
    return psnr_valse


def ssim(target, ref):
    ssim_value = structural_similarity(target, ref, channel_axis=True)
    return ssim_value


if __name__ == "__main__":
    cat_matrix = np.array([[1, 1],
                           [1, 2]])
    image = cv2.imread('lena.jpg', 0)
    print(image)
    encrypted = arnold(image, cat_matrix)
    print(encrypted)
    decrypted = inverse_arnold(encrypted, cat_matrix)
    print(decrypted)
    print(ssim(image, decrypted))
    print(psnr_gray(image, decrypted))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(encrypted, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(decrypted, cmap='gray')

    plt.show()
    pass
