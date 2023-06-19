import numpy as np
from SDCVP import SDCVP
import Create_Data
import math


def standard_qim(c: np.array, base: np.array, des: int, a: int, representatives: list[np.array]) -> np.array:
    """
    Move the host signal based on message
    :param c: single host signal
    :param base: lattice base
    :param des: messages
    :param a: magnification
    :param representatives: coset representatives
    :return: host signal with message
    """
    coarse_base = a * base  # coarse base
    q = np.dot(coarse_base, SDCVP(c - np.dot(base, representatives[des]), coarse_base, ))  # quantizer
    return q + np.dot(base, representatives[des])


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


def qim_decode(y: np.array, base: np.array, a: int, representatives: list[np.array]) -> int:
    """
    decode
    :param y: single host signal with message
    :param base: lattice base
    :param a: magnification
    :param representatives: coset representatives
    :return: the coset where host signal belongs, i.e. secret message
    """
    x = SDCVP(y, base, )  # column vector coefficients of host signals
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

    alpha = 2  # magnification
    num_carrier = 1280  # number of carriers (or host signals), divisible by dimension

    # coset representatives
    coset_representatives = cosets(alpha, N)
    length = round(math.log(len(coset_representatives), alpha))

    # create host signals
    carrier = Create_Data.carrier2(0, 1, num_carrier)  # mean and variance of Gaussian distribution
    # create secret messages
    secret = Create_Data.trans(Create_Data.secret_information(0.7, len(carrier)), length)

    carrier_standard = []  # host signals after standard QIM
    mse_standard = 0  # MSE of single QIM
    count = 0
    for num in range(0, len(carrier), N):
        qim_standard = standard_qim(carrier[num: num + N], b, secret[count], alpha, coset_representatives)
        carrier_standard.append(qim_standard)
        mse_standard += sum(pow(qim_standard - carrier[num: num + N], 2)) / N

        count += 1
        pass

    J = alpha * np.eye(N)
    R = 1 / N * math.log(np.linalg.det(J), 2)
    print('code rate')
    print(R)
    MSE_standard = mse_standard / len(carrier)
    print('MSE of QIM')
    print(MSE_standard)
    PSNR_standard = 20 * math.log(255 / math.sqrt(MSE_standard), 10)
    print('PSNR of QIM')
    print(PSNR_standard)

    """
    # decode
    m_standard_decode = []  # messages after decoding
    for num in range(0, round(len(carrier) / N)):
        m_standard_decode.append(qim_decode(carrier_standard[num], b, alpha, coset_representatives))
        pass
    difference_standard = 0
    for num in range(len(m_standard_decode)):
        if m_standard_decode[num] != secret[num]:
            difference_standard += 1
            pass
        pass
    print('difference after decoding')
    print(difference_standard)
    """

