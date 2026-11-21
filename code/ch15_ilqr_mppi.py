"""第 15 章 · 轨迹优化：iLQR（沿轨迹线性化 + LQR 反推）与 MPPI（采样 + 按 exp(−代价/λ) 加权平均）。
讲义：robot-motion-rl/notes/lessons/15-ilqr-mppi.md；运行：uv run python ch15_ilqr_mppi.py
例子（编的）：x_{t+1} = x_t + dt·(sin x_t + u_t)，从 x=0 走到 x=π（像把摆从下面甩到上面），代价 Σ q(x−π)² + r·u²。
"""

import numpy as np

DT, Q, R, T = 0.1, 1.0, 0.1, 30


def f(x, u):
    """非线性动力学。"""
    return x + DT * (np.sin(x) + u)


def total_cost(xs, us):
    """轨迹总代价：Σ q(x−π)² + r·u²（x 取第 1 到 T 步）。"""
    return float(Q * np.sum((np.asarray(xs[1:]) - np.pi) ** 2) + R * np.sum(np.square(us)))


def rollout(x0, us):
    """按控制序列往前滚，返回状态序列。"""
    xs = [x0]
    for u in us:
        xs.append(f(xs[-1], u))
    return np.array(xs)


def ilqr(x0=0.0, iters=8):
    """一维 iLQR：反推（线性化 a = ∂f/∂x，b = ∂f/∂u，标量 Riccati）→ 正推（带线搜索），返回每次迭代的代价。"""
    us = np.zeros(T)
    xs = rollout(x0, us)
    costs = [total_cost(xs, us)]
    for _ in range(iters):
        vx, vxx = 0.0, 0.0
        ks, kk = np.zeros(T), np.zeros(T)
        for t in reversed(range(T)):
            a, b = 1 + DT * np.cos(xs[t]), DT
            lx = 2 * Q * (xs[t + 1] - np.pi)
            # 把 t+1 步的状态代价折进来：Q 函数对 (δx, δu) 的二次展开
            qx = a * (vx + lx)
            qu = 2 * R * us[t] + b * (vx + lx)
            qxx = a * a * (vxx + 2 * Q)
            quu = 2 * R + b * b * (vxx + 2 * Q)
            qux = b * a * (vxx + 2 * Q)
            ks[t], kk[t] = -qu / quu, -qux / quu
            vx = qx + kk[t] * quu * ks[t] + kk[t] * qu + qux * ks[t]
            vxx = qxx + kk[t] * quu * kk[t] + 2 * kk[t] * qux
        best = None
        for alpha in (1.0, 0.5, 0.25, 0.1):
            xn, un = [x0], np.zeros(T)
            for t in range(T):
                un[t] = us[t] + alpha * ks[t] + kk[t] * (xn[-1] - xs[t])
                xn.append(f(xn[-1], un[t]))
            c = total_cost(xn, un)
            if c < costs[-1]:
                best = (np.array(xn), un, c)
                break
        if best is None:
            break
        xs, us, c = best
        costs.append(c)
    return costs, xs


def mppi_weights(costs, lam):
    """MPPI 权重：w_k ∝ exp(−(S_k − min S)/λ)，归一化到和为 1。"""
    s = np.asarray(costs, float)
    w = np.exp(-(s - s.min()) / lam)
    return w / w.sum()


def mppi(x0=0.0, samples=64, sigma=1.0, lam=1.0, iters=8, seed=0):
    """MPPI：每轮在当前控制序列上加噪声采样，按权重平均噪声更新序列，返回每轮代价。"""
    rng = np.random.default_rng(seed)
    us = np.zeros(T)
    costs = [total_cost(rollout(x0, us), us)]
    for _ in range(iters):
        eps = rng.normal(0, sigma, (samples, T))
        s = [total_cost(rollout(x0, us + e), us + e) for e in eps]
        us = us + mppi_weights(s, lam) @ eps
        costs.append(total_cost(rollout(x0, us), us))
    return costs


def main():
    print("\n第二步 iLQR")
    costs, xs = ilqr()
    print("每次迭代的代价：", [round(c, 2) for c in costs])
    print(f"最后一步 x = {xs[-1]:.3f}（目标 π ≈ 3.142）")

    print("\n第三步 MPPI 的权重")
    s = [10.0, 12.0, 20.0, 11.0]
    for lam in (1.0, 5.0):
        print(f"代价 {s}，λ={lam} → 权重 {np.round(mppi_weights(s, lam), 3)}")

    print("\n第三步 MPPI 迭代")
    print("每轮代价：", [round(c, 1) for c in mppi()])


if __name__ == "__main__":
    main()
