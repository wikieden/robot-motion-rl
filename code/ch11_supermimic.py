"""第 11 章 · SuperMimic 全流程：一次迭代的数据量、想象步数、和 PPO 在同一个 world model 上的比较。
讲义：robot-motion-rl/notes/lessons/11-supermimic.md；运行：uv run python ch11_supermimic.py
超参来自 SuperMimic 附录表 A.5。
"""

HP = {
    "envs_per_gpu": 4096,
    "buffer_steps_per_env": 1000,
    "window_k": 8,
    "wm_batches": 10,
    "wm_windows_per_batch": 2048,
    "horizon_H": 32,
    "imagined_rollouts": 4096,
    "iterations": 320_000,
}


def buffer_size(hp=HP):
    """回放缓冲区能存多少条转移：环境数 × 每环境步数。"""
    return hp["envs_per_gpu"] * hp["buffer_steps_per_env"]


def wm_transitions_per_update(hp=HP):
    """一次 world model 更新看多少个转移：10 批 × 2048 个窗口 × 8 步。"""
    return hp["wm_batches"] * hp["wm_windows_per_batch"] * hp["window_k"]


def imagined_steps_per_update(hp=HP):
    """一次策略更新想象多少步：4096 条 × 32 步。"""
    return hp["imagined_rollouts"] * hp["horizon_H"]


def main():
    print("\n第一步 回放缓冲区")
    print(f"4096 × 1000 = {buffer_size():,} 条转移")

    print("\n第二步 每次迭代的计算量")
    print(f"world model：10 × 2048 × 8 = {wm_transitions_per_update():,} 步预测")
    print(f"策略：4096 × 32 = {imagined_steps_per_update():,} 步想象")
    print(f"32 万次迭代共想象 {imagined_steps_per_update() * HP['iterations']:.2e} 步")

    print("\n第三步 三个提速因素")
    print("想象比仿真快 >10×（64k 并行时 62×）；解析梯度比 PPO-WM 省约 64× 想象步数；误差随数据增长保持 0.007–0.011")


if __name__ == "__main__":
    main()
