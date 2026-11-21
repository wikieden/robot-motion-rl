"""第 18 章 · 上真机：执行器网络、ASAP 的动作修正量（delta action）、控制延迟。
讲义：robot-motion-rl/notes/lessons/18-deploy.md；运行：uv run python ch18_deploy.py
玩具系统（编的）：仿真 x' = x + a；真机电机弱 20% 还有 0.05 的摩擦：x' = x + 0.8·a − 0.05。
"""

import numpy as np


def sim_step(x, a):
    """仿真里的一步。"""
    return x + a


def real_step(x, a):
    """真机的一步（编的偏差）。"""
    return x + 0.8 * a - 0.05


def fit_delta_action(actions):
    """ASAP 思路：学 Δa(a)，让 sim(a + Δa) 等于 real(a)。这里用线性模型 Δa = c1·a + c0 做最小二乘。"""
    a = np.asarray(actions, float)
    target = np.array([real_step(0.0, v) - sim_step(0.0, v) for v in a])
    design = np.stack([a, np.ones_like(a)], axis=1)
    c1, c0 = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(c1), float(c0)


def delay_steps(latency_ms, control_period_ms=20):
    """延迟相当于晚几个控制周期（向上取整）。"""
    return -(-latency_ms // control_period_ms)


def main():
    print("\n第二步 现实差距")
    for a in (0.0, 0.5, 1.0):
        print(f"a = {a}：仿真走 {sim_step(0, a):.2f}，真机走 {real_step(0, a):.2f}")

    print("\n第三步 学动作修正量")
    c1, c0 = fit_delta_action([0.0, 0.25, 0.5, 0.75, 1.0])
    print(f"Δa = {c1:.3f}·a + {c0:.3f}")
    a = 1.0
    print(f"a = 1：仿真里执行 a + Δa = {a + c1 * a + c0:.2f}，走 {sim_step(0, a + c1 * a + c0):.2f}（真机 {real_step(0, a):.2f}）")

    print("\n第四步 延迟")
    for ms in (5, 20, 30):
        print(f"延迟 {ms} ms，控制周期 20 ms → 晚 {delay_steps(ms)} 个周期")


if __name__ == "__main__":
    main()
