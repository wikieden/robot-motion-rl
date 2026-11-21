"""第 2 章 · 把动作跟踪写成 MDP：观测、目标、奖励核 exp(−e²/σ²)、终止。
讲义：robot-motion-rl/notes/lessons/02-tracking-mdp.md；运行：uv run python ch02_tracking_mdp.py
"""

import numpy as np

OBS_TERMS = {"关节位置 q−q0": 29, "关节速度 q̇": 29, "IMU 角速度": 3, "重力投影": 3, "上一步动作": 29}


def obs_dim(history=10):
    """本体感知观测维度：每帧 5 项相加，再乘历史帧数。"""
    return sum(OBS_TERMS.values()) * history


def kernel(err, sigma):
    """跟踪奖励核：exp(−e²/σ²)，误差 0 时得 1，误差等于 σ 时得 e^−1 ≈ 0.368。"""
    return float(np.exp(-(err**2) / sigma**2))


def tilt_error_from_angle(deg):
    """直立参考下，机器人倾斜 deg 度时的倾斜误差：|1 − cos(角度)|。"""
    return abs(1 - np.cos(np.radians(deg)))


def tracking_reward(pos_err, ori_err, lin_err, ang_err, root_pos_err, root_ori_err, da, joint_violation, dt=0.02):
    """BeyondMimic 式每步奖励（SuperMimic 表 A.2 的权重），每项再乘控制周期 dt。"""
    task = (
        kernel(pos_err, 0.3)
        + kernel(ori_err, 0.4)
        + kernel(lin_err, 1.0)
        + kernel(ang_err, 3.14)
        + 0.5 * kernel(root_pos_err, 0.3)
        + 0.5 * kernel(root_ori_err, 0.4)
    )
    reg = -0.1 * float(np.sum(np.square(da))) - 2.0 * joint_violation
    return (task + reg) * dt, task, reg


def main():
    print("\n第一步 观测维度")
    print("每帧", sum(OBS_TERMS.values()), "维 × 10 帧 =", obs_dim(), "维")

    print("\n第二步 奖励核")
    for e in (0.0, 0.1, 0.3, 0.6):
        print(f"位置误差 {e} m，σ=0.3 → {kernel(e, 0.3):.3f}")

    print("\n第三步 终止")
    print(f"倾斜 78° → 倾斜误差 {tilt_error_from_angle(78):.3f}（阈值 0.8）")

    print("\n第四步 一步的总奖励")
    r, task, reg = tracking_reward(0.1, 0.2, 0.5, 1.0, 0.1, 0.2, np.array([0.2, -0.1]), 0.0)
    print(f"任务项 {task:.3f}，正则项 {reg:.3f}，乘 dt=0.02 后 r = {r:.4f}")


if __name__ == "__main__":
    main()
