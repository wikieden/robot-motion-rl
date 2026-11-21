"""第 13 章 · 线性倒立摆与 ZMP：ẍ = ω²(x − p)，质心离开 ZMP 就按 cosh 发散；捕获点 ξ = x + ẋ/ω。
讲义：robot-motion-rl/notes/lessons/13-lipm-zmp.md；运行：uv run python ch13_lipm_zmp.py
"""

import numpy as np


def omega(z_c, g=9.8):
    """LIPM 的固有频率 ω = √(g / z_c)，z_c 是质心高度。"""
    return float(np.sqrt(g / z_c))


def lipm_state(x0, v0, p, w, t):
    """ẍ = ω²(x − p) 的解析解：x = p + (x0−p)·cosh(ωt) + (v0/ω)·sinh(ωt)，返回 (x, ẋ)。"""
    c, s = np.cosh(w * t), np.sinh(w * t)
    x = p + (x0 - p) * c + v0 / w * s
    v = (x0 - p) * w * s + v0 * c
    return float(x), float(v)


def capture_point(x, v, w):
    """捕获点 ξ = x + ẋ/ω：把 ZMP（脚）放在这里，质心正好停在它正上方。"""
    return x + v / w


def zmp_from_accel(x, xdd, w):
    """由质心加速度反推 ZMP：p = x − ẍ/ω²。"""
    return x - xdd / w**2


def main():
    print("\n第二步 LIPM")
    w = omega(0.8)
    print(f"质心高 0.8 m → ω = √(9.8/0.8) = {w:.3f} 1/s")
    print("想要 ẍ = 1 m/s²、质心在 0 → ZMP 要在", round(zmp_from_accel(0.0, 1.0, w), 4), "m")

    print("\n第三步 发散")
    for t in (0.1, 0.3, 0.5):
        x, v = lipm_state(0.05, 0.0, 0.0, w, t)
        print(f"质心比 ZMP 偏 5 cm、静止起步，t={t}s：x = {x:.4f} m，ẋ = {v:.4f} m/s")

    print("\n第四步 捕获点")
    xi = capture_point(0.0, 0.5, w)
    print(f"质心在 0、速度 0.5 m/s → 捕获点 ξ = {xi:.4f} m")
    x, v = lipm_state(0.0, 0.5, xi, w, 2.0)
    print(f"脚踩在 ξ 上，2 s 后：x = {x:.4f}，ẋ = {v:.4f}（停在脚正上方）")


if __name__ == "__main__":
    main()
