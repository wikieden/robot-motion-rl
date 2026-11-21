"""第 10 章 · 在想象里训练：BPTT 的梯度是雅可比连乘、critic 补上视野外的回报、梯度双重裁剪。
讲义：robot-motion-rl/notes/lessons/10-bptt-critic.md；运行：uv run python ch10_bptt_critic.py
"""

import numpy as np


def jacobian_product(a, horizon):
    """一维时 dx_H/dx_0 = a^H：|a|>1 爆炸，|a|<1 消失。"""
    return a**horizon


def imagined_objective(rewards, gamma, survival, v_end):
    """式 2：Σ_{t<H} γ^t·c_t·r_{t+1} + γ^H·c_H·V(s_H)。survival 长度 H+1。"""
    h = len(rewards)
    j = sum(gamma**t * survival[t] * rewards[t] for t in range(h))
    return j + gamma**h * survival[h] * v_end


def sees_reward(horizon, reward_step, critic):
    """视野 H 内看不到第 reward_step 步的奖励；有 critic 时 V(s_H) 能把它带回来（理想情况）。"""
    return reward_step < horizon or critic


def clip_grad(g, global_max=1.0, elem_max=0.1):
    """SuperMimic 表 A.5：先按全局范数缩到 ≤ 1，再逐元素裁到 ±0.1。"""
    g = np.asarray(g, float)
    norm = np.linalg.norm(g)
    if norm > global_max:
        g = g * (global_max / norm)
    return np.clip(g, -elem_max, elem_max)


def main():
    print("\n第一步 雅可比连乘")
    for a in (1.1, 0.9):
        for h in (8, 32, 128):
            print(f"a={a}, H={h}: a^H = {jacobian_product(a, h):.4g}")

    print("\n第二步 critic 补上视野外")
    for h, critic in [(16, False), (16, True), (64, False)]:
        print(f"H={h}，critic={critic} → 看得到第 40 步的回报：{sees_reward(h, 40, critic)}")
    j = imagined_objective([1, 1, 1], 0.99, [1, 1, 1, 1], 50.0)
    print(f"H=3、每步奖励 1、V(s_H)=50、γ=0.99 → J = {j:.3f}")

    print("\n第三步 梯度裁剪")
    print("g = [3, 0.4, −0.05] →", np.round(clip_grad([3, 0.4, -0.05]), 4))


if __name__ == "__main__":
    main()
