from equilibrium.initial import initial_policy
from equilibrium.update import equilibrium_ranking
from datasets import load_dataset
import os
from tqdm import tqdm

# 禁用并行化避免死锁问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def evaluate_mmlu(model_name, pi_G_init, pi_D_init):
    """
    在MMLU数据集上评估生成器(G)和判别器(D)的效果
    :param model_name: 使用的语言模型名称
    :param pi_G_init: 初始生成器策略
    :param pi_D_init: 初始判别器策略
    :param num_iterations: 最大迭代次数
    :return: G和D的准确率
    """
    # 加载MMLU数据集
    dataset = load_dataset("cais/mmlu", "all", split="test")  # 加载测试集

    # 初始化统计量
    correct_g = 0
    correct_d = 0
    total = 0

    # 遍历MMLU数据集
    for data in tqdm(dataset):
        input_text = data['question']  # MMLU问题文本
        candidates = data['choices']   # 候选答案列表
        correct_answer_index = data['answer']  # 正确答案是索引 (0, 1, 2, 3 等)

        correct_answer = candidates[correct_answer_index]

        # 调用 initial_policy 使用实际的 MMLU 输入
        pi_G_init, pi_D_init = initial_policy(model_name, input_text, candidates)

        # 设置迭代次数上限
        max_G_key, max_D_key = equilibrium_ranking(pi_G_init, pi_D_init)

        # 使用生成器G的策略
        g_answer = max_G_key  # 生成器策略中的最佳候选答案

        # 使用判别器D的策略
        d_answer = max_D_key  # 判别器策略中的最佳候选答案

        # 比较生成器G的答案与正确答案
        if g_answer == correct_answer:
            correct_g += 1

        # 比较判别器D的答案与正确答案
        if d_answer == correct_answer:
            correct_d += 1

        total += 1

    # 计算G和D的准确率
    accuracy_g = correct_g / total
    accuracy_d = correct_d / total

    print(f"生成器 G 的准确率: {accuracy_g * 100:.2f}%")
    print(f"判别器 D 的准确率: {accuracy_d * 100:.2f}%")

    return accuracy_g, accuracy_d


# 示例调用
model_name = "gpt2"

# 调用 initial_policy 和 equilibrium_ranking 获得策略并评估
evaluate_mmlu(model_name, pi_G_init=None, pi_D_init=None)
