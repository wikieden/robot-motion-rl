"""第 17 章 · 可微仿真与 SHAC：硬接触的导数要么 0 要么很大，平滑接触让梯度能用；整段 BPTT 切成短窗口 + critic。
讲义：robot-motion-rl/notes/lessons/17-shac.md；运行：uv run python ch17_shac.py
"""

import numpy as np


def hard_contact_force(z, k=1000.0):
    """硬接触（罚函数）：脚陷进地面（z < 0）才有力 f = k·(−z)，否则 0。"""
    return k * max(0.0, -z)


def hard_contact_grad(z, k=1000.0):
    """硬接触对高度的导数：离地时 0，触地时 −k，在 z = 0 处跳变。"""
    return -k if z < 0 else 0.0


def soft_contact_force(z, k=1000.0, beta=0.01):
    """平滑接触：把 max(0, −z) 换成 softplus，β 越小越接近硬接触。f = k·β·log(1 + exp(−z/β))。"""
    return float(k * beta * np.log1p(np.exp(-z / beta)))


def soft_contact_grad(z, k=1000.0, beta=0.01):
    """平滑接触的导数：−k·sigmoid(−z/β)，从 0 平滑过渡到 −k。"""
    return float(-k / (1 + np.exp(z / beta)))


def windows(total_horizon, window):
    """SHAC：长度 H 的任务切成多少个长度 h 的短窗口（向上取整）。"""
    return -(-total_horizon // window)


def main():
    print("\n第二步 硬接触 vs 平滑接触")
    for z in (0.02, 0.005, 0.0, -0.005):
        print(f"z = {z:+.3f} m：硬 f = {hard_contact_force(z):.2f}，df/dz = {hard_contact_grad(z):.0f}；"
              f"平滑 f = {soft_contact_force(z):.2f}，df/dz = {soft_contact_grad(z):.1f}")

    print("\n第三步 短窗口")
    print("SHAC 论文：H = 1000 → h = 32 的窗口数 =", windows(1000, 32))


if __name__ == "__main__":
    main()
