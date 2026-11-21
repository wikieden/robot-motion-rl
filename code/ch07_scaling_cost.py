"""第 7 章 · 规模化的代价：SONIC 与 SuperMimic 的算力账，以及扣掉数据量、硬件差异后还剩多少。
讲义：robot-motion-rl/notes/lessons/07-scaling-cost.md；运行：uv run python ch07_scaling_cost.py
数字来自 SuperMimic 论文 §4.1 "Compute accounting"（supermimic-2610.28480.txt L329–L337）。
"""


def gpu_hours(gpus, hours):
    """GPU 时 = GPU 数 × 小时数。"""
    return gpus * hours


def ratios(sonic=(128, 168), supermimic=(1, 3)):
    """返回 (SONIC GPU 时, SuperMimic GPU 时, GPU 时比, 墙钟比)。"""
    a, b = gpu_hours(*sonic), gpu_hours(*supermimic)
    return a, b, a / b, sonic[1] / supermimic[1]


def leftover_gap(ratio, data_factor=611 / 264.56, hw_factor=5.5):
    """把数据量差（2.3×）和硬件峰值差（≤ 5.5×）都算给 SONIC 之后，剩下的倍数。"""
    return ratio / (data_factor * hw_factor)


def main():
    print("\n第一步 GPU 时")
    a, b, r, w = ratios()
    print(f"SONIC 128 × 168 = {a} GPU 时；SuperMimic 1 × 3 = {b} GPU 时")
    print(f"GPU 时比 ≈ {r:.0f}×，墙钟比 = {w:.0f}×")

    print("\n第二步 扣掉数据和硬件")
    print(f"数据 611 / 264.56 = {611 / 264.56:.2f}×，硬件峰值最多 5.5× → 合计 {611 / 264.56 * 5.5:.1f}×")
    print(f"剩下的差距 ≈ {leftover_gap(r):.0f}×")

    print("\n第三步 PPO 加卡")
    for g, ret in [(1, 0.60), (4, 0.70), (8, 0.72)]:
        print(f"PPO-Sim {g} GPU，18 h 后回报 {ret}")
    print("SuperMimic 1 GPU，1.2 h 到 0.82")


if __name__ == "__main__":
    main()
