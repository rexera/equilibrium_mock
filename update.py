import numpy as np
import math

# 假设超参数已经设定
ETA_G = 0.1
ETA_D = 0.1
LAMBDA_G = 0.1
LAMBDA_D = 0.1
NUM_ITERATIONS = 5000

import numpy as np

def update_policy_G(pi_G_init, Q_G_t, t):
    """
    更新生成器的策略 π_G^(t+1)
    :param pi_G_init: 初始生成器策略
    :param Q_G_t: 生成器的平均值函数 Q_G^(t)
    :param t: 当前迭代次数 t
    :return: 更新后的生成器策略 π_G^(t+1)
    """
    pi_G_new = {}

    # 遍历每个候选答案 y
    for y, pi_G_1 in pi_G_init.items():
        if y in Q_G_t:
            # 增加小的稳定性常数以防止 log(0) 和除以 0 的问题
            stability_constant = 1e-8
            exponent = (Q_G_t[y] + LAMBDA_G * np.log(pi_G_1 + stability_constant)) / (1 / (ETA_G * t) + LAMBDA_G)
            pi_G_new[y] = np.exp(exponent)
        else:
            pi_G_new[y] = 1e-10  # 如果缺少数据，给出一个极小值

    # 归一化概率分布，确保数值稳定性
    max_prob = max(pi_G_new.values())  # 避免数值过大，使用最大值缩放
    for key in pi_G_new:
        pi_G_new[key] = np.exp(np.log(pi_G_new[key]) - np.log(max_prob))  # 减去最大值以防溢出

    total_sum = sum(pi_G_new.values())
    if total_sum == 0:  # 防止除以0的情况
        total_sum = 1e-10

    for key in pi_G_new:
        pi_G_new[key] /= total_sum

    return pi_G_new


def update_policy_D(pi_D_init, Q_D_t, t):
    """
    更新判别器的策略 π_D^(t+1)
    :param pi_D_init: 初始判别器策略
    :param Q_D_t: 判别器的平均值函数 Q_D^(t)
    :param t: 当前迭代次数 t
    :return: 更新后的判别器策略 π_D^(t+1)
    """
    pi_D_new = {}

    # 遍历每个候选答案 y
    for y, pi_D_1 in pi_D_init.items():
        if y in Q_D_t:
            # 增加小的稳定性常数以防止 log(0) 和除以 0 的问题
            stability_constant = 1e-8
            exponent = (Q_D_t[y] + LAMBDA_D * np.log(pi_D_1 + stability_constant)) / (1 / (ETA_D * t) + LAMBDA_D)
            pi_D_new[y] = np.exp(exponent)
        else:
            pi_D_new[y] = 1e-10  # 如果缺少数据，给出一个极小值

    # 归一化概率分布，确保数值稳定性
    max_prob = max(pi_D_new.values())  # 使用最大值缩放，避免数值溢出
    for key in pi_D_new:
        pi_D_new[key] = np.exp(np.log(pi_D_new[key]) - np.log(max_prob))  # 减去最大值

    total_sum = sum(pi_D_new.values())
    if total_sum == 0:  # 防止除以0的情况
        total_sum = 1e-10

    for key in pi_D_new:
        pi_D_new[key] /= total_sum

    return pi_D_new

def compute_Q_G_t(pi_D_history, t):
    """
    计算生成器的平均值函数 Q_G^(t)
    :param pi_D_history: 判别器策略的历史记录
    :param t: 当前迭代次数 t
    :return: Q_G^(t) 的值
    """
    Q_G_t = {}
    for key, values in pi_D_history.items():
        # 确保 pi_D_history 和 Q_G_t 中的键与初始策略一致
        if len(values) >= t:
            Q_G_t[key] = (1 / (2 * t)) * sum(values[:t])
        else:
            Q_G_t[key] = 1e-10  # 如果历史记录不足，初始化为很小的数值
    return Q_G_t


def compute_Q_D_t(pi_G_history, t):
    """
    计算判别器的平均值函数 Q_D^(t)
    :param pi_G_history: 生成器策略的历史记录
    :param t: 当前迭代次数 t
    :return: Q_D^(t) 的值
    """
    Q_D_t = {}
    for key, values in pi_G_history.items():
        if len(values) >= t:
            Q_D_t[key] = (1 / (2 * t)) * sum(values[:t])
        else:
            Q_D_t[key] = 1e-10  # 如果历史记录不足，初始化为很小的数值
    return Q_D_t


def equilibrium_ranking(pi_G_init, pi_D_init):
    """
    执行 5000 次迭代的策略更新
    :param pi_G_init: 生成器初始策略
    :param pi_D_init: 判别器初始策略
    :return: 最终的生成器和判别器策略
    """
    pi_G_history = {key: [pi_G_init[key]] for key in pi_G_init}
    pi_D_history = {key: [pi_D_init[key]] for key in pi_D_init}

    for t in range(1, NUM_ITERATIONS + 1):
        # 计算平均值函数 Q_G^(t) 和 Q_D^(t)
        Q_G_t = compute_Q_G_t(pi_D_history, t)
        Q_D_t = compute_Q_D_t(pi_G_history, t)

        # 更新生成器和判别器的策略
        pi_G_new = update_policy_G(pi_G_init, Q_G_t, t)
        pi_D_new = update_policy_D(pi_D_init, Q_D_t, t)

        # 记录策略历史
        for key in pi_G_new:
            pi_G_history[key].append(pi_G_new[key])
        for key in pi_D_new:
            pi_D_history[key].append(pi_D_new[key])

        # 更新初始策略为新的策略
        pi_G_init = pi_G_new
        pi_D_init = pi_D_new

    # 打印最终的生成器和判别器的完整策略
    print("最终生成器策略:", pi_G_init)
    print("最终判别器策略:", pi_D_init)

    # 找到生成器和判别器策略中概率最大的项
    max_G_key = max(pi_G_init, key=pi_G_init.get)  # 生成器概率最大的候选答案
    max_D_key = max(pi_D_init, key=pi_D_init.get)  # 判别器概率最大的候选答案


    # 返回生成器和判别器策略中概率最大的两个项
    return max_G_key, max_D_key

def main():
    # 示例策略初始化
    pi_G_init = {'Paris': 0.7407182607700358, 'Berlin': 0.008202937026555022, 'Rome': 0.005355415567765109, 'London': 0.18997983004555913}
    pi_D_init = {'Paris': 0.2513788173191846, 'Berlin': 0.2357402271162316, 'Rome': 0.26486686878387566, 'London': 0.24801408678070813}

    # 执行迭代更新
    final_pi_G, final_pi_D = equilibrium_ranking(pi_G_init, pi_D_init)

    print("最终生成器策略:", final_pi_G)
    print("最终判别器策略:", final_pi_D)

if __name__ == "__main__":
    main()