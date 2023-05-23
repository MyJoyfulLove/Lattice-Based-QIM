import numpy as np
import cv2
from Research_Algorithm.SDCVP import SDCVP
import random


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
    b = np.array([[1]])
    # b = np.array([[1, 1 / 2],
    #               [0, pow(3, 0.5) / 2]])
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

    embeded_image = cv2.imread('08_embed_3.png', 0)
    rows, cols = embeded_image.shape
    embeded_image2 = embeded_image.reshape((1, -1))
    random.seed(10)
    secret = [random.randint(0, 1) for _ in range(rows * cols)]
    # secret = [1 for _ in range(rows * cols)]
    decode = []
    for i in range(0, embeded_image2.shape[1], N):
        decode.append(qim_decode(embeded_image2[0][i:i + N], b, R, coset_representatives))
        pass
    wrong = 0
    for item in range(len(decode)):
        if decode[item] != secret[item]:
            wrong += 1
    print(wrong / len(decode))
