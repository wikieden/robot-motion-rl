# 机器人运控与强化学习

**在线阅读：<https://wikieden.github.io/robot-motion-rl/>** · [讲义目录](https://wikieden.github.io/robot-motion-rl/lessons/index.html) · [全部动画](https://wikieden.github.io/robot-motion-rl/animations/index.html) · [复习卡片](https://wikieden.github.io/robot-motion-rl/review.html) · [全部讲义书](https://wikieden.github.io/ai-books/)

> 在线阅读：<https://wikieden.github.io/robot-motion-rl/> · 全部讲义书：<https://wikieden.github.io/ai-books/>
> 英文名 RL for Robot Motion Control。《从零学强化学习》的续篇。
> 18 章（第一部分 11 章 + 第二部分 7 章）· 18 个动画（GIF + 可交互页面）· 36 张复习卡片 · 覆盖 SuperMimic（arXiv 2610.28480）全文，legged_gym、DeepMimic、BeyondMimic、SONIC、GRPO 相关论文的运控主线，以及 Underactuated Robotics 的 LQR / 人形 / 轨迹优化、凸 MPC + WBIC、MPPI、AMP、SHAC、ASAP

## 这本书讲什么

从「策略到底输出什么」讲起：运控的三层和 PD 目标接口，把模仿人类动作写成 MDP，PPO 在几千个并行机器人上的配方，仿真到真机，动作模仿的采样技巧；再看清两件事：GRPO 为什么不适合底层运控，PPO 为什么「动作越多越要堆算力」；最后用两种梯度、可微 world model、BPTT + critic 三章，一步步拼出 SuperMimic：1 张 GPU 训 3 小时，超过 128 张 GPU 训 7 天的跟踪器。第二部分（第 12–18 章）补上两样东西：经典控制这条路（LQR、线性倒立摆与 ZMP、凸 MPC + WBC、iLQR 与 MPPI），作为理解「学习在省什么」的参照；以及学习路线的三块拼图（AMP 风格奖励、SHAC 可微仿真、上真机）。

- **适合谁**：读完《从零学强化学习》（至少第 20–24 章：策略梯度、baseline、GAE、PPO、GRPO）的人。机器人只要知道「关节、电机、传感器」；PD 控制、高斯分布、重参数化、链式法则、线性化、双曲函数、雅可比都在用到的那一章现场补。
- **读完能做什么**：读懂 SuperMimic、BeyondMimic、legged_gym 的方法和超参表；解释为什么一个人形跟踪器要那样设计奖励、终止、重置和梯度；对一个运控问题判断该用 PPO、GRPO、MPC 还是可微 world model；自己推 LQR 的 Riccati 迭代、ZMP 和捕获点、摩擦锥约束；动手做 学习计划 的 MC-M8（手写 PPO）、MC-M12（GRPO 对照）、MC-M16（小规模 SuperMimic）。
- **怎么读**：按章顺序读。每章先看讲解和动画，再自己做练习，最后用复习卡片隔几天考自己。

## 讲解的特点

- **一次一小步**：每个概念按「白话类比 → 公式（每个符号说清楚怎么读）→ 小数字亲手算一遍」讲。
- **每章一个动画**：PD 跟踪、奖励核、高斯比率、域随机化、重置分布、信用分配、算力账、两种梯度、world model 滚动、BPTT 视野、三步循环、LQR、倒立摆与捕获点、滚动优化、MPPI、AMP、接触导数、动作修正量，都能播放、单步、拖滑块。
- **对照原文**：讲到论文的地方标本地文本行号（如 SuperMimic L278–L281），论文文本在 `cs285-deep-rl/materials/text/papers/`。
- **算账不含糊**：SuperMimic「省 1000 倍」怎么算、哪里不公平、扣掉后还剩多少（第 7 章）；玩具例子出了和讲义说法矛盾的数，就改例子，不改结论。
- **术语不设门槛**：缩写第一次出现就给英文全称并逐词解释，全书缩写见 [缩写与术语表](https://wikieden.github.io/robot-motion-rl/lessons/appendix-glossary.html)。
- **每章都有代码对照**：只用 NumPy，Mac 上直接跑，配测试核对和讲义一致；只算讲过的例子，不提前给练习答案。
- **每章收尾有口诀**：全部口诀汇总在本页末尾。

## 示例动画

| | | |
|---|---|---|
| ![低增益 PD](https://wikieden.github.io/robot-motion-rl/animations/01-pd-layers.gif) | ![GRPO vs PPO 信用分配](https://wikieden.github.io/robot-motion-rl/animations/06-grpo-vs-ppo-credit.gif) | ![两种梯度](https://wikieden.github.io/robot-motion-rl/animations/08-score-vs-pathwise.gif) |
| **第 1 章** 低增益 PD：目标角其实是力矩旋钮 · [讲义](https://wikieden.github.io/robot-motion-rl/lessons/01-layers-pd.html) · [▶ 交互](https://wikieden.github.io/robot-motion-rl/animations/01-pd-layers.html) | **第 6 章** GRPO 整条打分，惩罚摊薄；PPO 逐步打分，惩罚落在踩错那步 · [讲义](https://wikieden.github.io/robot-motion-rl/lessons/06-grpo-credit.html) · [▶ 交互](https://wikieden.github.io/robot-motion-rl/animations/06-grpo-vs-ppo-credit.html) | **第 8 章** 采样梯度弯弯绕绕，解析梯度几乎直线 · [讲义](https://wikieden.github.io/robot-motion-rl/lessons/08-two-gradients.html) · [▶ 交互](https://wikieden.github.io/robot-motion-rl/animations/08-score-vs-pathwise.html) |
| ![算力账](https://wikieden.github.io/robot-motion-rl/animations/07-gpu-hours.gif) | ![视野与 critic](https://wikieden.github.io/robot-motion-rl/animations/10-bptt-horizon.gif) | ![三步循环](https://wikieden.github.io/robot-motion-rl/animations/11-supermimic-loop.gif) |
| **第 7 章** 21504 vs 3 GPU 时，扣掉数据和硬件还剩 564 倍 · [讲义](https://wikieden.github.io/robot-motion-rl/lessons/07-scaling-cost.html) · [▶ 交互](https://wikieden.github.io/robot-motion-rl/animations/07-gpu-hours.html) | **第 10 章** 视野越长梯度越不稳，critic 补上视野外 · [讲义](https://wikieden.github.io/robot-motion-rl/lessons/10-bptt-critic.html) · [▶ 交互](https://wikieden.github.io/robot-motion-rl/animations/10-bptt-horizon.html) | **第 11 章** 采数据 → 拟合模型 → 想象里改策略 · [讲义](https://wikieden.github.io/robot-motion-rl/lessons/11-supermimic.html) · [▶ 交互](https://wikieden.github.io/robot-motion-rl/animations/11-supermimic-loop.html) |

可交互页面要用本地服务打开才能拖滑块：`.claude/launch.json` 里的 `robot-motion-rl-animations`（端口 8354），见 [animations/README.md](https://wikieden.github.io/robot-motion-rl/animations/index.html)。

## 目录

| 章 | 主题 | 一句话精要 | 动画 | 卡片 | 状态 |
|---|---|---|---|---|---|
| **第一部分** | **从 PPO 到 SuperMimic** | | | | |
| 1 | [三层与 PD 目标接口](https://wikieden.github.io/robot-motion-rl/lessons/01-layers-pd.html) | 策略 50 Hz 给设定值，PD 1 kHz 出力矩；低增益时设定值是力矩旋钮 | [01](https://wikieden.github.io/robot-motion-rl/animations/01-pd-layers.html) | M01–M02 | 正文 ✅，练习待作答 |
| 2 | [把动作跟踪写成 MDP](https://wikieden.github.io/robot-motion-rl/lessons/02-tracking-mdp.html) | 930 维本体感知 + 10 帧参考；exp(−e²/σ²) 奖励核；偏高倒斜就终止 | [02](https://wikieden.github.io/robot-motion-rl/animations/02-reward-kernel.html) | M03–M04 | 正文 ✅，练习待作答 |
| 3 | [PPO 运控配方](https://wikieden.github.io/robot-motion-rl/lessons/03-ppo-recipe.html) | 高斯策略的 log 概率与比率；4096 × 24 一批；KL 自适应学习率 | [03](https://wikieden.github.io/robot-motion-rl/animations/03-gauss-ratio.html) | M05–M06 | 正文 ✅，练习待作答 |
| 4 | [从仿真到真机](https://wikieden.github.io/robot-motion-rl/lessons/04-sim2real.html) | 域随机化买的是稳；特权信息给 critic 或 teacher；ξ 喂给模型 | [04](https://wikieden.github.io/robot-motion-rl/animations/04-domain-rand.html) | M07–M08 | 正文 ✅，练习待作答 |
| 5 | [动作模仿与采样](https://wikieden.github.io/robot-motion-rl/lessons/05-imitation-sampling.html) | RSI 从中间开局；摔了就停；50/40/10 按回报重置 | [05](https://wikieden.github.io/robot-motion-rl/animations/05-reset-sampling.html) | M09–M10 | 正文 ✅，练习待作答 |
| 6 | [GRPO 为什么不适合底层运控](https://wikieden.github.io/robot-motion-rl/lessons/06-grpo-credit.html) | 整条打分 vs 逐步打分；惩罚摊成 1/T；GRPO 留给 VLA 微调 | [06](https://wikieden.github.io/robot-motion-rl/animations/06-grpo-vs-ppo-credit.html) | M11–M12 | 正文 ✅，练习待作答 |
| 7 | [规模化的代价](https://wikieden.github.io/robot-motion-rl/lessons/07-scaling-cost.html) | 21504 vs 3 GPU 时；扣掉后仍 564×；PPO 加卡不变聪明 | [07](https://wikieden.github.io/robot-motion-rl/animations/07-gpu-hours.html) | M13–M14 | 正文 ✅，练习待作答 |
| 8 | [两种梯度](https://wikieden.github.io/robot-motion-rl/lessons/08-two-gradients.html) | 似然比只看分，解析沿路求导；同一模型里省 64 倍 | [08](https://wikieden.github.io/robot-motion-rl/animations/08-score-vs-pathwise.html) | M15–M16 | 正文 ✅，练习待作答 |
| 9 | [可微 world model](https://wikieden.github.io/robot-motion-rl/lessons/09-world-model.html) | 已知随机性当输入；8 步自回归纠偏；生存权重让摔倒可导 | [09](https://wikieden.github.io/robot-motion-rl/animations/09-wm-rollout.html) | M17–M18 | 正文 ✅，练习待作答 |
| 10 | [在想象里训练](https://wikieden.github.io/robot-motion-rl/lessons/10-bptt-critic.html) | 雅可比连乘会爆炸；32 步 BPTT + critic；梯度双重裁剪 | [10](https://wikieden.github.io/robot-motion-rl/animations/10-bptt-horizon.html) | M19–M20 | 正文 ✅，练习待作答 |
| 11 | [SuperMimic 全流程](https://wikieden.github.io/robot-motion-rl/lessons/11-supermimic.html) | 三步循环、一次迭代的账、三个提速因素、局限与选路线 | [11](https://wikieden.github.io/robot-motion-rl/animations/11-supermimic-loop.html) | M21–M22 | 正文 ✅，练习待作答 |
| **第二部分** | **经典控制作参照 + 学习路线补全** | | | | |
| 12 | [LQR](https://wikieden.github.io/robot-motion-rl/lessons/12-lqr.html) | 线性 + 二次时 Bellman 有闭式解；Riccati 迭代；q 管偏离、r 管用力 | [12](https://wikieden.github.io/robot-motion-rl/animations/12-lqr.html) | M23–M24 | 正文 ✅，练习待作答 |
| 13 | [线性倒立摆与 ZMP](https://wikieden.github.io/robot-motion-rl/lessons/13-lipm-zmp.html) | ẍ = ω²(x − p)；按 cosh 发散；捕获点 ξ = x + ẋ/ω | [13](https://wikieden.github.io/robot-motion-rl/animations/13-lipm-zmp.html) | M25–M26 | 正文 ✅，练习待作答 |
| 14 | [单刚体凸 MPC 与 WBC](https://wikieden.github.io/robot-motion-rl/lessons/14-convex-mpc.html) | 机身当砖头；摩擦锥锁住 μg；滚动优化；τ = Jᵀf | [14](https://wikieden.github.io/robot-motion-rl/animations/14-convex-mpc.html) | M27–M28 | 正文 ✅，练习待作答 |
| 15 | [iLQR 与 MPPI](https://wikieden.github.io/robot-motion-rl/lessons/15-ilqr-mppi.html) | 沿轨迹线性化 + LQR 反推；撒序列 + exp(−代价/λ) 加权；在线 vs 离线 | [15](https://wikieden.github.io/robot-motion-rl/animations/15-ilqr-mppi.html) | M29–M30 | 正文 ✅，练习待作答 |
| 16 | [AMP](https://wikieden.github.io/robot-motion-rl/lessons/16-amp.html) | 判别器打分当风格奖励；r_S = max(0, 1 − 0.25(D−1)²)；任务风格各一半 | [16](https://wikieden.github.io/robot-motion-rl/animations/16-amp.html) | M31–M32 | 正文 ✅，练习待作答 |
| 17 | [SHAC 与可微仿真](https://wikieden.github.io/robot-motion-rl/lessons/17-shac.html) | 硬接触导数跳变、softplus 磨平；32 步窗口 + critic；梯度从仿真器还是模型来 | [17](https://wikieden.github.io/robot-motion-rl/animations/17-shac.html) | M33–M34 | 正文 ✅，练习待作答 |
| 18 | [上真机](https://wikieden.github.io/robot-motion-rl/lessons/18-deploy.html) | 四类差距；执行器网络；ASAP 动作修正量；sim2sim 与上机清单 | [18](https://wikieden.github.io/robot-motion-rl/animations/18-deploy.html) | M35–M36 | 正文 ✅，练习待作答 |
| 附录 | [缩写与术语表](https://wikieden.github.io/robot-motion-rl/lessons/appendix-glossary.html) | 全书缩写、记号、论文名 | — | — | ✅ |

---

本仓库由私有学习仓库里的脚本 `tools/publish-transformer/build.py` 自动生成，内容摘自讲义目录，不要直接修改这里的文件。
