import numpy as np
from TheSDCVP import SDCVP
import Create_Data
import math


def ca_qim(c: np.array, basis: np.array, des: int, a: int, representatives: list[np.array]) -> np.array:
    """
    Content Aware QIM
    :param c: single host signal
    :param basis: lattice basis
    :param des: messages
    :param a: magnification
    :param representatives: coset representatives
    :return: host signal with message
    """
    coarse_basis = a * basis
    q = np.dot(coarse_basis, SDCVP(c - np.dot(basis, representatives[des]), coarse_basis, ))
    return q + np.dot(basis, representatives[des])


def camd_qim(c: np.array, basis: np.array, des: int, a: int, representatives: list[np.array]) -> np.array:
    """
    Content Aware + Minimum distortion QIM
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
    c_coset = qim_decode(c, basis, a, representatives)
    if c_coset == des:
        cw = c
        pass
    else:
        p = x - c
        p_len = pow(sum(pow(p, 2)), 1 / 2)
        rp = packing_radius(basis)
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
    from itertools import product
    representatives = []
    for item in list(product(*representative)):
        representatives.append(np.asarray(item))
    return representatives


def statis(c: np.array, basis: np.array, s: list[int], a: int, representatives: list[np.array]) -> list[tuple]:
    """
    Statistically count the coset in which the carrier is in
    and the corresponding situation of the secret message representation
    :param c: single host signal
    :param basis: lattice basis
    :param s: single message
    :param a: magnification
    :param representatives: coset representatives
    :return: new remapping
    """
    w = np.zeros((a ** basis.shape[0], a ** basis.shape[0]))  # record by a matrix
    record = []
    for i in range(0, round(len(c) / N)):
        z = SDCVP(np.asarray(c[N * i: N * i + N]), basis, )
        c_coset = get_coset(z, a, representatives)
        w[s[i]][c_coset] += 1
        record.append(c_coset)
        pass

    from scipy.optimize import linear_sum_assignment

    def matrix_max_value(matrix: np.array):
        matrix = -matrix
        row_ind, col_ind = linear_sum_assignment(matrix)
        max_task = [(i, c) for i, c in enumerate(col_ind)]
        return max_task

    order = matrix_max_value(w)
    return order


def update_secret(s: list[int], order: list[tuple]) -> list[int]:
    """
    Update secret messages
    :param s: whole secret
    :param order: new remapping
    :return: updated messages
    """
    update_s = []
    for i in range(len(s)):
        for j in range(len(order)):
            if s[i] == order[j][0]:
                update_s.append(order[j][1])
                pass
    return update_s


def qim_decode(y: np.array, basis: np.array, a: int, representatives: list[np.array]) -> int:
    """
    decode
    :param y: single host signal with message
    :param basis: lattice basis
    :param a: magnification
    :param representatives: coset representatives
    :return: the coset where host signal belongs, i.e. secret message
    """
    x = SDCVP(y, basis, )  # 将量化后的坐标代入得到其最近格点的列向量系数
    for i in range(len(representatives)):
        if (x % a == representatives[i]).all():
            break
    return i


def get_coset(y: np.array, a: int, representatives: list[np.array]) -> int:
    """
    decode in statis
    :param y: single host signal with message
    :param a: magnification
    :param representatives: coset representatives
    :return: the coset where host signal belongs, i.e. secret message
    """
    for i in range(len(representatives)):
        if (y % a == representatives[i]).all():
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

    coset_representatives = cosets(alpha, N)
    length = round(math.log(len(coset_representatives), alpha))

    carrier = Create_Data.carrier(0, 1, num_carrier)
    secret = Create_Data.trans(Create_Data.secret_information(0.9, len(carrier)), length)  # 非均匀分布中0所占比例

    # 通过修改消息，重新调整对应关系
    statis_w = statis(carrier, b, secret, alpha, coset_representatives)
    secret_update = update_secret(secret, statis_w)

    mse_ca = 0
    mse_camd = 0
    count = 0
    carrier_ca = []
    carrier_camd = []
    for num in range(0, len(carrier), N):
        qim_ca = ca_qim(carrier[num: num + N], b, secret_update[count], alpha, coset_representatives)  # CA-QIM
        carrier_ca.append(qim_ca)
        mse_ca += sum(pow(qim_ca - carrier[num: num + N], 2)) / N

        qim_camd = camd_qim(carrier[num: num + N], b, secret_update[count], alpha, coset_representatives)  # CA-QIM
        carrier_camd.append(qim_camd)
        mse_camd += sum(pow(qim_camd - carrier[num: num + N], 2)) / N

        count += 1
        pass

    J = alpha * np.eye(N)
    R = 1 / N * math.log(np.linalg.det(J), 2)
    print('code rate')
    print(R)

    MSE_ca = mse_ca / len(carrier)
    print('MSE of CA-QIM')
    print(MSE_ca)
    PSNR_ca = 20 * math.log(255 / math.sqrt(MSE_ca), 10)  # max(carrier) - min(carrier)
    print('PSNR of CA-QIM')
    print(PSNR_ca)

    MSE_camd = mse_camd / len(carrier)
    print('MSE of CAMD-QIM')
    print(MSE_camd)
    PSNR_camd = 20 * math.log(255 / math.sqrt(MSE_camd), 10)  # max(carrier) - min(carrier)
    print('PSNR of CAMD-QIM')
    print(PSNR_camd)
