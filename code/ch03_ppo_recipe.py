"""第 3 章 · 运控里的 PPO 配方：高斯策略的 log 概率、比率与剪裁、批量账、按 KL 调学习率。
讲义：robot-motion-rl/notes/lessons/03-ppo-recipe.md；运行：uv run python ch03_ppo_recipe.py
"""

import numpy as np


def gauss_logp(a, mu, sigma):
    """一维高斯 log 概率：−(a−μ)²/(2σ²) − log σ − ½ log(2π)；多维时各维相加。"""
    a, mu, sigma = np.asarray(a, float), np.asarray(mu, float), np.asarray(sigma, float)
    return float(np.sum(-((a - mu) ** 2) / (2 * sigma**2) - np.log(sigma) - 0.5 * np.log(2 * np.pi)))


def ratio(a, mu_new, mu_old, sigma):
    """比率 r = π_新(a)/π_旧(a) = exp(log π_新 − log π_旧)。"""
    return float(np.exp(gauss_logp(a, mu_new, sigma) - gauss_logp(a, mu_old, sigma)))


def ppo_term(r, adv, eps=0.2):
    """PPO 剪裁目标的一项：min(r·A, clip(r, 1−ε, 1+ε)·A)。"""
    return min(r * adv, float(np.clip(r, 1 - eps, 1 + eps)) * adv)


def batch_accounting(envs=4096, steps=24, minibatches=4, epochs=5):
    """一次迭代的批量账：总样本、每个 mini-batch 大小、梯度步数。"""
    total = envs * steps
    return total, total // minibatches, minibatches * epochs


def adaptive_lr(lr, kl, kl_target=0.01, factor=1.5):
    """按 KL 调学习率（rsl_rl 的规则）：KL > 2·目标就除以 1.5，KL < 目标/2 就乘 1.5。"""
    if kl > 2 * kl_target:
        return max(lr / factor, 1e-5)
    if kl < kl_target / 2:
        return min(lr * factor, 1e-2)
    return lr


def main():
    print("\n第一步 高斯策略的 log 概率")
    print(f"μ=0.2, σ=0.5, a=0.7 → log π = {gauss_logp(0.7, 0.2, 0.5):.3f}")

    print("\n第二步 比率与剪裁")
    r = ratio(0.7, 0.3, 0.2, 0.5)
    print(f"μ 从 0.2 改到 0.3 → r = {r:.3f}；A=+1 时 PPO 项 = {ppo_term(r, 1.0):.3f}")
    r2 = ratio(0.7, 0.45, 0.2, 0.5)
    print(f"μ 改到 0.45 → r = {r2:.3f}；A=+1 时被剪成 {ppo_term(r2, 1.0):.3f}")

    print("\n第三步 批量账")
    total, mb, steps = batch_accounting()
    print(f"4096 × 24 = {total} 个样本；mini-batch {mb}；每次迭代 {steps} 步梯度")

    print("\n第四步 按 KL 调学习率")
    for kl in (0.03, 0.01, 0.004):
        print(f"lr=1e-3, KL={kl} → {adaptive_lr(1e-3, kl):.2e}")


if __name__ == "__main__":
    main()
