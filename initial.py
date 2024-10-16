import random

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_generator_initial_probabilities(model, tokenizer, input_text, candidates, v):
    """
    初始化生成器策略 π_G^(1)，针对给定的正确性参数 v 和候选答案进行归一化
    :param model: 预训练语言模型
    :param tokenizer: 模型的 tokenizer
    :param input_text: 输入的文本
    :param candidates: 候选答案列表
    :param v: 当前的正确性参数 ('correct' 或 'incorrect')
    :return: 生成器策略的归一化概率分布
    """
    pi_G = {}
    v_list = ['correct', 'incorrect']  # 可能的正确性参数，用于归一化

    # 针对传入的 v，构建 prompt，不包含具体的候选答案
    if v == "correct":
        prompt = f"{input_text} Answer:"
    else:
        prompt = f"{input_text} Incorrect Answer:"

    # 编码 prompt
    inputs_with_prompt = tokenizer(prompt, return_tensors="pt")

    # 通过模型获取 logits
    with torch.no_grad():
        outputs = model(**inputs_with_prompt)

    # 获取最后一个 token 的 logits 并计算 softmax 概率
    logits = outputs.logits[:, -1, :]
    probabilities = torch.softmax(logits, dim=-1)

    # 遍历每个候选答案 y，获取它们的概率
    candidate_probs = {}
    for y in candidates:
        # 获取候选答案 y 对应的 token id
        y_token_id = tokenizer.encode(y, add_special_tokens=False)[-1]
        # 获取该 token 的生成概率
        candidate_probs[y] = probabilities[0, y_token_id].item()

    # 为归一化计算所有 v' 的候选答案生成概率总和
    sum_v_probs = 0
    for v_prime in v_list:
        # 重新构建 v' 对应的 prompt
        if v_prime == "correct":
            prompt_prime = f"{input_text} Answer:"
        else:
            prompt_prime = f"{input_text} Incorrect Answer:"

        inputs_prime = tokenizer(prompt_prime, return_tensors="pt")
        with torch.no_grad():
            outputs_prime = model(**inputs_prime)

        logits_prime = outputs_prime.logits[:, -1, :]
        probabilities_prime = torch.softmax(logits_prime, dim=-1)

        # 计算所有 v' 下候选答案的生成概率总和
        for y in candidates:
            y_token_id = tokenizer.encode(y, add_special_tokens=False)[-1]
            y_prob_v_prime = probabilities_prime[0, y_token_id].item()
            sum_v_probs += y_prob_v_prime

    # 归一化当前 v 下的生成器概率
    for y in candidates:
        pi_G[y] = candidate_probs[y] / sum_v_probs

    return pi_G

def get_discriminator_initial_probabilities(model, tokenizer, input_text, candidates, v):
    """
    初始化判别器策略 π_D^(1)，根据输入 x 和候选答案 y 推断出 v 的概率，并进行归一化
    :param model: 预训练语言模型
    :param tokenizer: 模型的 tokenizer
    :param input_text: 输入的文本
    :param candidates: 候选答案列表
    :return: 判别器策略的归一化概率分布
    """
    pi_D = {}
    #v = "correct"  # 判别的目标是 correct 或 incorrect, 我们以 "correct" 为例

    # 计算每个 y 的 P_LM(v | x, y)
    candidate_probs = {}

    # 遍历候选答案 y，计算其 P_LM(v | x, y)
    for y in candidates:
        # 构建 prompt：让模型判断候选答案 y 是不是正确的答案
        prompt = f"{input_text} + {y}, is it correct or incorrect?"

        # 编码 prompt
        inputs_with_prompt = tokenizer(prompt, return_tensors="pt")

        # 获取模型输出
        with torch.no_grad():
            outputs = model(**inputs_with_prompt)

        # 获取最后一个 token 的 logits 并计算 softmax 概率
        logits = outputs.logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)

        # 获取 'correct' 的概率
        v_token_id = tokenizer.encode(v, add_special_tokens=False)[-1]
        v_prob = probabilities[0, v_token_id].item()

        # 保存该候选答案 y 对应的 P_LM(v | x, y)
        candidate_probs[y] = v_prob

    # 计算所有 y' 的概率和，进行归一化
    sum_probs = sum(candidate_probs.values())
    for y in candidates:
        pi_D[y] = candidate_probs[y] / sum_probs

    return pi_D

def initial_policy(model_name, input_text, candidates):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    v = random.choice(["correct", "incorrect"])
    print(f"Random Correctness: {v}")
    # 初始化生成器和判别器的策略
    generator_probs = get_generator_initial_probabilities(model, tokenizer, input_text, candidates, v)
    discriminator_probs = get_discriminator_initial_probabilities(model, tokenizer, input_text, candidates, v)
    return generator_probs, discriminator_probs

def main():
    # 加载预训练模型和 tokenizer
    model_name = "gpt2"  # 或其他预训练模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 输入问题和候选答案
    input_text = "The capital of France is"
    candidates = ["Paris", "Berlin", "Rome", "London"]
    v = random.choice(["correct", "incorrect"])
    print(f"Random Correctness: {v}")
    # 初始化生成器和判别器的策略
    generator_probs = get_generator_initial_probabilities(model, tokenizer, input_text, candidates, v)
    discriminator_probs = get_discriminator_initial_probabilities(model, tokenizer, input_text, candidates, v)

    print("生成器初始策略:", generator_probs)
    print("判别器初始策略:", discriminator_probs)

if __name__ == "__main__":
    main()
