"""第 1 章 · 运控的三层与 PD 目标接口：τ = Kp(q* − q) − Kd·q̇，低增益时设定值 ≈ 力矩代理。
讲义：robot-motion-rl/notes/lessons/01-layers-pd.md；运行：uv run python ch01_layers_pd.py
"""

import numpy as np


def pd_torque(kp, kd, q_target, q, qd, tau_max=None):
    """PD 目标接口：τ = Kp(q* − q) − Kd·q̇（目标速度取 0），可选力矩饱和 ±τ_max。"""
    tau = kp * (q_target - q) - kd * qd
    return tau if tau_max is None else float(np.clip(tau, -tau_max, tau_max))


def pd_steps_per_policy_step(pd_hz=1000, policy_hz=50):
    """策略每输出一次设定值，底层 PD 要跑几次。"""
    return pd_hz // policy_hz


def steady_state_error(load, kp):
    """恒定负载（如重力）下 PD 的稳态误差：Kp·e = 负载 → e = 负载 / Kp。"""
    return load / kp


def simulate_joint(kp, kd, q_target, load, inertia=0.1, seconds=2.0, dt=0.001):
    """1 自由度关节：I·q̈ = τ_PD − 负载，1 kHz 积分，返回最后的角度。"""
    q, qd = 0.0, 0.0
    for _ in range(int(seconds / dt)):
        tau = pd_torque(kp, kd, q_target, q, qd)
        qdd = (tau - load) / inertia
        qd += qdd * dt
        q += qd * dt
    return q


def main():
    print("\n第一步 三层频率")
    print("1 kHz PD / 50 Hz 策略：每个策略动作对应", pd_steps_per_policy_step(), "次 PD")

    print("\n第二步 PD 力矩")
    print("Kp=40, Kd=2, q*=0.5, q=0.3, q̇=1.0 → τ =", pd_torque(40, 2, 0.5, 0.3, 1.0), "N·m")
    print("加饱和 τ_max=5 →", pd_torque(40, 2, 0.5, 0.3, 1.0, tau_max=5), "N·m")

    print("\n第三步 低增益：设定值是力矩代理")
    for kp in (40, 200):
        e = steady_state_error(4.0, kp)
        q_end = simulate_joint(kp, 2 * np.sqrt(kp * 0.1), 0.5, 4.0)
        print(f"Kp={kp}: 稳态误差 = 4/{kp} = {e:.3f} rad；仿真最后 q = {q_end:.3f}（目标 0.5）")
    print("想停在 0.5、负载 4 N·m、Kp=40：设定值要给 0.5 + 0.1 =", 0.5 + steady_state_error(4.0, 40))


if __name__ == "__main__":
    main()
