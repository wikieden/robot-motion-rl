"""第 14 章 · 单刚体凸 MPC 与 WBC：地面反力受摩擦锥限制；滚动优化只执行第一步；WBC 用 τ = Jᵀf 把力变成关节力矩。
讲义：robot-motion-rl/notes/lessons/14-convex-mpc.md；运行：uv run python ch14_convex_mpc.py
数字是编的，质量 9 kg 接近 MIT Mini Cheetah。
"""

import numpy as np

G = 9.8


def stance_normal_force(mass, n_feet):
    """匀速站立时每只着地脚分担的竖直力：mg / n。"""
    return mass * G / n_feet


def max_horizontal_force(fz, mu):
    """摩擦锥（线性化成金字塔）：|f_x| ≤ μ·f_z。"""
    return mu * fz


def max_accel(mu):
    """所有脚都用满摩擦时的最大水平加速度：μ·g（与质量无关）。"""
    return mu * G


def mpc_plan(v0, v_ref, mass, horizon, dt, f_max, smooth=1e-6):
    """一维速度 MPC：选 horizon 步水平力，min Σ(v_k − v_ref)² + smooth·Σf²，再把力截到 ±f_max。
    动力学 v_{k+1} = v_k + f_k·dt/m 是线性的，用最小二乘求解。"""
    lower = np.tril(np.ones((horizon, horizon))) * dt / mass
    a = np.vstack([lower, np.sqrt(smooth) * np.eye(horizon)])
    y = np.concatenate([np.full(horizon, v_ref - v0), np.zeros(horizon)])
    f = np.linalg.lstsq(a, y, rcond=None)[0]
    return np.clip(f, -f_max, f_max)


def receding_horizon(v0, v_ref, mass, horizon, dt, f_max, steps):
    """滚动优化：每次规划 horizon 步、只执行第一步，返回速度轨迹。"""
    v, vs = v0, [v0]
    for _ in range(steps):
        f = mpc_plan(v, v_ref, mass, horizon, dt, f_max)[0]
        v += f * dt / mass
        vs.append(v)
    return np.array(vs)


def leg_torque(jacobian, force):
    """WBC 的核心一步：着地脚要出力 f，关节力矩 τ = Jᵀ·f。"""
    return np.asarray(jacobian).T @ np.asarray(force)


def main():
    print("\n第二步 地面反力与摩擦锥")
    fz = stance_normal_force(9.0, 2)
    print(f"9 kg、对角两脚着地：每脚 f_z = {fz:.1f} N；μ=0.6 时 |f_x| ≤ {max_horizontal_force(fz, 0.6):.2f} N")
    print(f"最大水平加速度 μg = {max_accel(0.6):.2f} m/s²")

    print("\n第三步 滚动优化")
    vs = receding_horizon(0.0, 1.0, 9.0, 10, 0.03, 2 * max_horizontal_force(fz, 0.6), 10)
    print("速度（每 0.03 s）：", np.round(vs, 3))

    print("\n第四步 WBC：τ = Jᵀf")
    j = [[0.0, -0.4], [0.3, 0.2]]
    print("J =", j, "，f = [0, 44.1] → τ =", np.round(leg_torque(j, [0.0, 44.1]), 3))


if __name__ == "__main__":
    main()
