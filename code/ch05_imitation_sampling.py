"""第 5 章 · 动作模仿：参考状态初始化、提前终止，以及 SuperMimic 的按回报重置（50% 均匀 + 40% softmax + 10% 历史）。
讲义：robot-motion-rl/notes/lessons/05-imitation-sampling.md；运行：uv run python ch05_imitation_sampling.py
"""

import numpy as np


def softmax(x):
    """softmax：先减最大值防溢出，再 exp、归一化。"""
    z = np.exp(np.asarray(x, float) - np.max(x))
    return z / z.sum()


def reset_distribution(g_over_gmax, past_weight=None, temp=25.0):
    """SuperMimic 附录 D.1：50% 均匀 + 40% softmax(−temp·G_b/G_max) + 10% 历史权重（不给就用当前 softmax）。"""
    g = np.asarray(g_over_gmax, float)
    hard = softmax(-temp * g)
    past = hard if past_weight is None else np.asarray(past_weight, float)
    uniform = np.full_like(g, 1.0 / len(g))
    return 0.5 * uniform + 0.4 * hard + 0.1 * past, hard


def rsi_reach_probability(start_phase_uniform, clip_seconds, flip_at, survive_seconds):
    """从片头开始时，策略能撑 survive 秒才可能练到 flip_at 处；RSI 均匀抽起点时练到它的概率（编的简化）。"""
    if not start_phase_uniform:
        return 1.0 if survive_seconds >= flip_at else 0.0
    window = min(survive_seconds, clip_seconds)
    return min(1.0, window / clip_seconds)


def main():
    print("\n第一步 参考状态初始化 RSI")
    print("10 s 片段，空翻在第 6 s，策略只能撑 2 s：从片头开始练到空翻的概率 =", rsi_reach_probability(False, 10, 6, 2))
    print("RSI 均匀抽起点，起点落在空翻前 2 s 内的概率 =", rsi_reach_probability(True, 10, 6, 2))

    print("\n第二步 按回报重置")
    g = [0.9, 0.9, 0.2, 0.8, 0.9]
    mix, hard = reset_distribution(g)
    print("各段 G/G_max =", g)
    print("softmax(−25·G/G_max) =", np.round(hard, 3))
    print("最终重置分布 =", np.round(mix, 3))


if __name__ == "__main__":
    main()
