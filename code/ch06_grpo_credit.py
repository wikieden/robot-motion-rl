"""第 6 章 · GRPO 为什么不适合底层运控：整条打分 vs PPO + GAE 逐步打分（理想 critic、λ=0、γ=1 的小例子）。
讲义：robot-motion-rl/notes/lessons/06-grpo-credit.md；运行：uv run python ch06_grpo_credit.py
例子（编的）：4 条轨迹、最长 6 步、每步站着得 1 分；轨迹 A 在第 1 步踩错，第 3 步后摔倒（只活 3 步）。
"""

import numpy as np


def grpo_advantages(returns):
    """GRPO：A_i = (R_i − 组平均) / 组标准差，整条轨迹每一步都用这一个数。"""
    r = np.asarray(returns, float)
    return (r - r.mean()) / r.std()


def td_errors(rewards, values):
    """GAE 取 λ=0、γ=1：A_t = δ_t = r_t + V(s_{t+1}) − V(s_t)。values 比 rewards 多一个（最后是 0）。"""
    return [rewards[t] + values[t + 1] - values[t] for t in range(len(rewards))]


def ideal_values(length, horizon, mistake=None):
    """理想 critic：没出错时 V = 还剩几步；踩错之后 V = 摔倒前还剩几步。"""
    v = []
    for t in range(length + 1):
        if t >= length:
            v.append(0.0)
        elif mistake is not None and t > mistake:
            v.append(float(length - t))
        else:
            v.append(float(horizon - t))
    return v


def main():
    print("\n第一步 GRPO 整条打分")
    returns = [3, 6, 6, 6]
    adv = grpo_advantages(returns)
    print("回报 =", returns, "→ 优势 =", np.round(adv, 3))

    print("\n第二步 PPO + GAE 逐步打分（轨迹 A）")
    v = ideal_values(3, 6, mistake=1)
    print("V =", v)
    print("A_t =", td_errors([1, 1, 1], v))

    print("\n第三步 惩罚落在踩错那一步的比例")
    print(f"GRPO：1/3 = {1 / 3:.0%}；PPO + GAE：3/3 = 100%")


if __name__ == "__main__":
    main()
