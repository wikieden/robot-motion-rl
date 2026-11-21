"""第 12 章 · LQR：线性化 + 二次代价，Bellman 方程有闭式解（Riccati 迭代）。
讲义：robot-motion-rl/notes/lessons/12-lqr.md；运行：uv run python ch12_lqr.py
例子（编的）：一维离散系统 x_{t+1} = a·x_t + b·u_t，代价 Σ (q·x² + r·u²)。
"""

import numpy as np


def linearize_pendulum(g=9.8, length=1.0):
    """倒立摆在竖直附近：θ̈ = (g/l)·sin θ + u ≈ (g/l)·θ + u，返回线性系数 g/l。"""
    return g / length


def riccati_step(p, a, b, q, r):
    """一步 Riccati 迭代：P ← q + a²P − (abP)² / (r + b²P)。"""
    return q + a * a * p - (a * b * p) ** 2 / (r + b * b * p)


def solve_lqr(a, b, q, r, iters=200):
    """反复迭代到收敛，返回 (P, K, 每次迭代的 P)。K = abP / (r + b²P)，控制律 u = −K·x。"""
    p, hist = q, [q]
    for _ in range(iters):
        p = riccati_step(p, a, b, q, r)
        hist.append(p)
    k = a * b * p / (r + b * b * p)
    return p, k, hist


def simulate(a, b, k, x0, steps):
    """闭环 x_{t+1} = (a − bK)·x_t，返回轨迹。"""
    xs = [x0]
    for _ in range(steps):
        xs.append((a - b * k) * xs[-1])
    return np.array(xs)


def main():
    print("\n第一步 线性化")
    print("θ̈ ≈ (g/l)·θ + u，g/l =", linearize_pendulum())

    print("\n第三步 Riccati 迭代（a=1.1, b=1, q=1, r=1）")
    p, k, hist = solve_lqr(1.1, 1.0, 1.0, 1.0)
    print("前 4 次 P：", [round(v, 4) for v in hist[:4]])
    print(f"收敛 P = {p:.4f}，K = {k:.4f}，闭环 a − bK = {1.1 - k:.4f}")

    print("\n第四步 调 q 和 r")
    for q, r in [(1, 1), (1, 10), (10, 1)]:
        _, k, _ = solve_lqr(1.1, 1.0, q, r)
        xs = simulate(1.1, 1.0, k, 1.0, 5)
        print(f"q={q}, r={r}: K = {k:.3f}，闭环系数 {1.1 - k:.3f}，5 步后 x = {xs[-1]:.4f}")


if __name__ == "__main__":
    main()
