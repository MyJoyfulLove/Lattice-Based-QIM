"""
Reproduce code of 'Lattice-Based Minimum-Distortion Data Hiding'
"""

import numpy as np
from SDCVP import SDCVP
import Create_Data
import math


def md_qim(c: np.array, basis: np.array, des: int, a: int, representatives: list[np.array]) -> np.array:
    """
    Minimum Distortion QIM
    :param c: single host signal
    :param basis: lattice basis
    :param des: messages
    :param a: magnification
    :param representatives: coset representatives
    :return: host signal with message
    """
    coarse_basis = a * basis
    x = np.dot(coarse_basis, SDCVP(c - np.dot(basis, representatives[des]), coarse_basis, )) \
        + np.dot(basis, representatives[des])
    p = x - c
    p_len = pow(sum(pow(p, 2)), 1 / 2)
    rp = packing_radius(basis)
    if p_len < rp:
        cw = c
        pass
    else:
        epsilon = rp / 2 - 0.05
        cw = x - p / p_len * (rp - epsilon)
        pass
    return cw


def packing_radius(basis: np.array) -> float:
    """
    Calculate the packing radius of given lattice
    :param basis: lattice basis
    :return: packing radius
    """
    return round(min(pow(sum(pow(basis, 2)), 1 / 2)) / 2, 5)


def cosets(a: int, n: int) -> list[np.array]:
    """
    Calculate the coset representatives
    :param a: magnification
    :param n: dimension of the lattice
    :return: coset representatives
    """
    vector = []
    for i in range(a):
        vector.append(i)
        pass
    representative = [vector] * n
    from itertools import product  # Cartesian product
    representatives = []
    for item in list(product(*representative)):
        representatives.append(np.asarray(item))
    return representatives


def qim_decode(y: np.array, basis: np.array, a: int, representatives: list[np.array]) -> int:
    """
    decode
    :param y: single host signal with message
    :param basis: lattice basis
    :param a: magnification
    :param representatives: coset representatives
    :return: the coset where host signal belongs, i.e. secret message
    """
    x = SDCVP(y, basis, )
    for i in range(len(representatives)):
        if (x % a == representatives[i]).all():
            break
    return i


if __name__ == '__main__':
    # b = np.array([[1]])
    b = np.array([[1, 0],
                  [1, 2]])
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

    alpha = 4  # magnification
    num_carrier = 1280  # number of carriers (or host signals), divisible by dimension

    # coset representatives
    coset_representatives = cosets(alpha, N)
    length = round(math.log(len(coset_representatives), alpha))

    # create host signals
    carrier = Create_Data.carrier(0, 1, num_carrier)  # mean and variance of Gaussian distribution
    # create secret messages
    secret = Create_Data.trans(Create_Data.secret_information(0.5, len(carrier)), length)

    carrier_md = []
    mse_md = 0
    count = 0
    for num in range(0, len(carrier), N):
        qim_md = md_qim(carrier[num: num + N], b, secret[count], alpha, coset_representatives)
        carrier_md.append(qim_md)
        mse_md += sum(pow(qim_md - carrier[num: num + N], 2)) / N

        count += 1
        pass

    J = alpha * np.eye(N)
    R = 1 / N * math.log(np.linalg.det(J), 2)
    print('code rate')
    print(R)
    MSE_md = mse_md / len(carrier)
    print('MSE of MD-QIM')
    print(MSE_md)
    PSNR_md = 20 * math.log(255 / math.sqrt(MSE_md), 10)
    print('PSNR of MD-QIM ')
    print(PSNR_md)
