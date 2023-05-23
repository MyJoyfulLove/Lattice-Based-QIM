# 《Lossless and Reversible Data Hiding in Encrypted Images With Public-Key Cryptography》中复现图片加密
import numpy as np
import math
import random
from Number_Algorithm import *


def image_encryption(image: np.array):
    """
    一种基于同态加密的图像加密，对图像每个像素分别进行处理
    :return:
    """
    p, q = random_prime(100, 1000), random_prime(100, 1000)
    n = p * q
    labda = math.lcm(p - 1, q - 1)
    while math.gcd(n, (p - 1) * (q - 1)) != 1:
        p, q = random_prime(100, 1000), random_prime(100, 1000)
        n = p * q
        labda = math.lcm(p - 1, q - 1)
        pass
    g = random.randint(0, n ** 2)
    miu = int(ex_euclid((pow(g, labda) % pow(n, 2) - 1) / n, n))
    return miu


if __name__ == '__main__':
    img = np.array([[1, 2, 3],
                    [4, 5, 6]])
    print(image_encryption(img))
    pass
