import numpy as np
import cv2
from Research_Algorithm.SDCVP import SDCVP
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import random


def standard_qim(c: np.array, basis: np.array, des: int, r: int, representatives: list[np.array]) -> np.array:
    """
    根据秘密消息对载体进行移动
    :param c: 单个载体
    :param basis: 格基
    :param des: 秘密消息所在陪集
    :param r: 码率
    :param representatives: 陪集代表
    :return: 含密载体
    """
    coarse_basis = r * basis  # 粗格
    q = np.dot(coarse_basis, SDCVP(c - np.dot(basis, representatives[des]), coarse_basis, ))  # 量化器
    return q + np.dot(basis, representatives[des])


def qim_decode(y: np.array, basis: np.array, r: int, representatives: list[np.array]) -> int:
    """
    解码
    :param y: 单个含密载体
    :param basis: 格基
    :param r: 码率
    :param representatives: 陪集代表
    :return: 含密载体所属陪集，即解码结果
    """
    x = SDCVP(y, basis, )  # 将量化后的坐标代入得到其最近格点的列向量系数
    for i in range(len(representatives)):
        if (x % r == representatives[i]).all():
            break
    return i


def psnr_gray(target, ref):
    diff = target - ref
    mse = np.mean(np.square(diff))
    psnr_valse = 10 * np.log10(255 * 255 / mse)
    return psnr_valse


def ssim(target, ref):
    ssim_value = structural_similarity(target, ref, channel_axis=True)
    return ssim_value


def cosets(r: int, n: int) -> list[np.array]:
    """
    计算陪集代表
    :param r: 码率
    :param n: 格的维度
    :return: 陪集代表
    """
    vector = []
    for i in range(r):
        vector.append(i)
        pass
    representative = [vector] * n
    from itertools import product  # 笛卡尔积
    representatives = []
    for item in list(product(*representative)):
        representatives.append(np.asarray(item))
    return representatives


if __name__ == "__main__":
    # b = np.array([[1]])
    b = np.array([[1, 1 / 2],
                  [0, pow(3, 0.5) / 2]])
    # b = np.array([[2, 1, 1, 1],
    #               [0, 1, 0, 0],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]])
    # b = np.array([[2, -1, 0, 0, 0, 0, 0, 0.5],
    #               [0, 1, -1, 0, 0, 0, 0, 0.5],
    #               [0, 0, 1, -1, 0, 0, 0, 0.5],
    #               [0, 0, 0, 1, -1, 0, 0, 0.5],
    #               [0, 0, 0, 0, 1, -1, 0, 0.5],
    #               [0, 0, 0, 0, 0, 1, -1, 0.5],
    #               [0, 0, 0, 0, 0, 0, 1, 0.5],
    #               [0, 0, 0, 0, 0, 0, 0, 0.5]])
    N = b.shape[0]
    R = 2
    coset_representatives = cosets(R, N)

    image = cv2.imread('08.png', 0)
    rows, cols = image.shape
    image2 = image.reshape((1, -1))
    embeded_image2 = np.zeros(image2.shape)
    random.seed(10)
    secret = [random.randint(0, 1) for _ in range(rows * cols)]
    # secret = [0 for _ in range(rows * cols)]
    count = 0
    for i in range(0, image2.shape[1], N):
        embeded_image2[0][i:i + N] = standard_qim(image2[0][i:i + N], b, secret[count], R, coset_representatives)
        count += 1
        pass
    embeded_image = np.int32(embeded_image2.reshape((rows, cols)))
    # cv2.imwrite('08_embed.png', embeded_image)
    print(ssim(image, embeded_image))
    print(psnr_gray(image, embeded_image))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(embeded_image, cmap='gray')
    plt.show()

    # 提取
    decode = []
    for i in range(0, image2.shape[1], N):
        decode.append(qim_decode(embeded_image2[0][i:i + N], b, R, coset_representatives))
        pass
    wrong = 0
    for item in range(len(decode)):
        if decode[item] != secret[item]:
            wrong += 1
    print(wrong / len(decode))
