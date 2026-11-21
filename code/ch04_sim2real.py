"""第 4 章 · 从仿真到真机：只在一个摩擦上训、域随机化、知道摩擦的 teacher，三种策略在真机摩擦下的表现。
讲义：robot-motion-rl/notes/lessons/04-sim2real.md；运行：uv run python ch04_sim2real.py
玩具任务（编的）：动作 a 推一个箱子，效果 = a·μ（μ 是摩擦系数），想让效果正好等于 1，奖励 = −(a·μ − 1)²。
"""

import numpy as np


def reward(a, mu):
    """玩具奖励：−(a·μ − 1)²，效果正好是 1 时得 0（最高）。"""
    return -((a * mu - 1.0) ** 2)


def best_action_single(mu_train):
    """只在一个摩擦上训：a = 1/μ_train。"""
    return 1.0 / mu_train


def best_action_dr(lo=0.3, hi=1.2):
    """域随机化：μ ~ U[lo, hi]，最大化平均奖励 → a = E[μ] / E[μ²]。"""
    e_mu = (lo + hi) / 2
    e_mu2 = (hi**3 - lo**3) / (3 * (hi - lo))
    return e_mu / e_mu2


def worst_case(a, lo=0.3, hi=1.2, n=91):
    """在 μ ∈ [lo, hi] 上扫一遍，返回最差的奖励。"""
    return float(min(reward(a, m) for m in np.linspace(lo, hi, n)))


def best_action_teacher(mu_true):
    """teacher 能看到真实摩擦（特权信息）：a = 1/μ。"""
    return 1.0 / mu_true


def main():
    print("\n第一步 只在 μ=1.0（仿真默认摩擦）上训")
    a1 = best_action_single(1.0)
    print(f"a = {a1:.3f}")

    print("\n第二步 域随机化 μ~U[0.3, 1.2]")
    a2 = best_action_dr()
    print(f"a = E[μ]/E[μ²] = {a2:.3f}")

    print("\n第二步（续）最坏情况：μ 在 0.3 和 1.2 两端")
    for name, a in [("只训 1.0", a1), ("域随机化", a2)]:
        print(f"{name}: μ=0.3 奖励 {reward(a, 0.3):.3f}，μ=1.2 奖励 {reward(a, 1.2):.3f}，最坏 {worst_case(a):.3f}")

    print("\n第三步 真机 μ=0.4 时三种策略")
    for name, a in [("只训 1.0", a1), ("域随机化", a2), ("teacher", best_action_teacher(0.4))]:
        print(f"{name}: a={a:.3f}，效果 {a * 0.4:.3f}，奖励 {reward(a, 0.4):.3f}")


if __name__ == "__main__":
    main()
