"""第 16 章 · AMP：判别器给「像不像人」打分，风格奖励 r_S = max(0, 1 − 0.25(D − 1)²)，和任务奖励按权重相加。
讲义：robot-motion-rl/notes/lessons/16-amp.md；运行：uv run python ch16_amp.py
公式和权重来自 AMP 论文式 4、式 7 与表（amp-2104.02180.txt L277、L302、L517）。
"""

import numpy as np


def style_reward(d):
    """式 7：判别器输出 D 越接近 1（像参考动作）奖励越高，最低截到 0。"""
    return max(0.0, 1.0 - 0.25 * (d - 1.0) ** 2)


def total_reward(r_task, r_style, w_task=0.5, w_style=0.5):
    """式 4：r = w_G·r_G + w_S·r_S，论文里两个权重都是 0.5。"""
    return w_task * r_task + w_style * r_style


def lsgan_disc_loss(d_real, d_fake):
    """最小二乘判别器损失：参考动作的 D 往 1 推，策略动作的 D 往 −1 推。"""
    return float(np.mean((np.asarray(d_real) - 1) ** 2) + np.mean((np.asarray(d_fake) + 1) ** 2))


def main():
    print("\n第三步 风格奖励")
    for d in (1.0, 0.0, -1.0, 2.0, 3.0):
        print(f"D = {d:+.0f} → r_S = {style_reward(d):.2f}")

    print("\n第三步 判别器损失")
    print("参考 D = [0.9, 1.1]，策略 D = [−0.8, −1.2] →", round(lsgan_disc_loss([0.9, 1.1], [-0.8, -1.2]), 3))
    print("参考 D = [0.2, 0.4]，策略 D = [0.3, 0.1] →", round(lsgan_disc_loss([0.2, 0.4], [0.3, 0.1]), 3))

    print("\n第四步 和任务奖励相加")
    print("任务 0.8、D = 0（风格 0.75）→ r =", total_reward(0.8, style_reward(0.0)))
    print("任务 0.2、D = 1（风格 1.00）→ r =", total_reward(0.2, style_reward(1.0)))


if __name__ == "__main__":
    main()
