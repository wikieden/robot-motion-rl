"""第 9 章 · 可微 world model：误差怎么滚大、k 步自回归损失、生存权重。
讲义：robot-motion-rl/notes/lessons/09-world-model.md；运行：uv run python ch09_world_model.py
玩具系统（编的）：真实 x_{t+1} = 0.95·x_t，模型学成 0.97·x_t，起点 x_0 = 1。
"""

import numpy as np


def rollout(a, x0, steps):
    """线性系统 x_{t+1} = a·x_t 往前滚 steps 步，返回 [x_0, …, x_steps]。"""
    return x0 * a ** np.arange(steps + 1)


def rollout_error(a_true, a_model, x0, k):
    """模型自己滚 k 步（吃自己的预测）和真实值的差。"""
    return float(abs(rollout(a_model, x0, k)[-1] - rollout(a_true, x0, k)[-1]))


def k_step_loss(a_true, a_model, x0, k):
    """k 步自回归损失：Σ_{j=1..k} (x̂_j − x_j)²，x̂ 从真实起点出发、一路用自己的预测。"""
    xh, x = rollout(a_model, x0, k), rollout(a_true, x0, k)
    return float(np.sum((xh[1:] - x[1:]) ** 2))


def survival_weights(d_hat):
    """生存权重：c_0 = 1，c_{t+1} = c_t·(1 − d̂_{t+1})。d_hat 给的是 d̂_1, d̂_2, …"""
    c = [1.0]
    for d in d_hat:
        c.append(c[-1] * (1 - d))
    return c


def main():
    print("\n第一步 误差滚大")
    for k in (1, 8, 32):
        print(f"滚 {k} 步：误差 {rollout_error(0.95, 0.97, 1.0, k):.4f}")

    print("\n第二步 1 步损失 vs 8 步损失")
    print(f"1 步：{k_step_loss(0.95, 0.97, 1.0, 1):.6f}；8 步：{k_step_loss(0.95, 0.97, 1.0, 8):.6f}")

    print("\n第三步 生存权重")
    c = survival_weights([0.0, 0.1, 0.5])
    print("d̂ = [0, 0.1, 0.5] → c =", [round(v, 3) for v in c])
    print("每步奖励都是 1 时，加权回报 = c_0 + c_1 + c_2 =", round(sum(c[:3]), 3))


if __name__ == "__main__":
    main()
