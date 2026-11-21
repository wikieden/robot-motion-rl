"""第 8 章 · 两种梯度：似然比（score function）估计 vs 解析（pathwise）梯度，比方差。
讲义：robot-motion-rl/notes/lessons/08-two-gradients.md；运行：uv run python ch08_two_gradients.py
例子（编的）：回报 J(a) = −(a − 2)²，策略 a = θ + σ·ε，ε ~ N(0, 1)，在 θ = 0、σ = 0.5 处求 dE[J]/dθ。真值 = 4。
"""

import numpy as np


def J(a):
    """回报：a 越接近 2 越好。"""
    return -((a - 2.0) ** 2)


def dJ_da(a):
    """回报对动作的导数（解析梯度要用到）：−2(a − 2)。"""
    return -2.0 * (a - 2.0)


def score_estimate(theta, sigma, eps, baseline=True):
    """似然比估计：mean[(J(a) − b) · ε / σ]，b 取这批 J 的平均。"""
    a = theta + sigma * eps
    j = J(a)
    b = j.mean() if baseline else 0.0
    return float(np.mean((j - b) * eps / sigma))


def pathwise_estimate(theta, sigma, eps):
    """解析梯度：a = θ + σε 对 θ 的导数是 1，所以 dJ/dθ = mean[dJ/da(a)]。"""
    return float(np.mean(dJ_da(theta + sigma * eps)))


def spread(estimator, n, repeats=2000, seed=0, **kw):
    """重复估计 repeats 次（每次 n 个样本），返回 (平均, 标准差)。"""
    rng = np.random.default_rng(seed)
    vals = [estimator(0.0, 0.5, rng.standard_normal(n), **kw) for _ in range(repeats)]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    print("\n第一步 真梯度")
    print("dE[J]/dθ 在 θ=0 = −2(0 − 2) =", dJ_da(0.0))

    print("\n第二步 一次估计（4 个样本）")
    eps = np.array([0.5, -1.0, 1.5, 0.2])
    print(f"似然比 = {score_estimate(0.0, 0.5, eps):.3f}；解析 = {pathwise_estimate(0.0, 0.5, eps):.3f}")

    print("\n第三步 重复 2000 次看散布")
    for n in (4, 64):
        ms, ss = spread(score_estimate, n)
        mp, sp = spread(pathwise_estimate, n)
        print(f"N={n}: 似然比 {ms:.2f} ± {ss:.2f}；解析 {mp:.2f} ± {sp:.2f}")


if __name__ == "__main__":
    main()
