# SuperMimic 论文与 GitHub 复现调研

调研日期：2026-10-22（America/Los_Angeles）。查询了 GitHub 名称/描述搜索和 README 扩展搜索，筛出 10 个相关实现仓库；未把论文日报、论文列表和普通 fork 重复计入。仓库仍在快速变化，查询快照在 repositories.json，查询时 main 的 SHA 在 revisions.json。

本次完成论文正文、附录、10 个仓库 README 和目录检查，并抽查重点实现的世界模型、BPTT、观测、评估和实验记录。没有安装训练依赖、下载动作数据或执行 GPU 程序。下列训练成绩是仓库作者报告，未经本机独立复测。

## 论文读后结论

论文：[SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/abs/2610.28480)，2026-10-20 发布，Berkeley；[项目页](https://supermimic.github.io/)。截至查询时，主页没有提供官方训练仓库链接，以下均为第三方实现。

核心训练循环是：当前策略在 MuJoCo Warp 中采集带动作噪声的 transition → replay buffer → 用 8 步自回归窗口更新世界模型 → 固定世界模型权重，通过 32 步 imagined rollout 的 BPTT 更新策略，再用 GAE 目标训练 critic。世界模型权重在策略更新期间固定，但对状态与动作的梯度必须保留；并非对真实模拟器求导。世界模型和策略持续交替更新。

重要配置：Unitree G1 29 DoF，50 Hz 控制、4×5 ms 物理子步，10 帧本体观测历史、10 帧未来参考动作，世界模型 MLP 2048×3，4096 仿真环境/卡，4096 imagined trajectories，γ=0.99、λ=0.95、UTD=0.4。论文采用 TF32；多个复现改用 bf16。

适合第一次训练的目标是 Fig. 4a：LAFAN1 40 条动作、2.45 小时，论文报告单 H100 1.2 小时评估回报达到 0.82。主结果则是过滤后的 BONES-SEED 129,514 条、264.56 小时：单 H100 3 小时超过 SONIC 已发布 checkpoint，失败 25 条；两卡 9.8 小时零失败。后者衡量训练动作覆盖率，不是未见动作泛化；不同硬件、数据量和评估口径也影响算力比较。

## 仓库列表

| 仓库 | 后端/定位 | 作者报告的结果或当前证据 | 本次判断 |
|---|---|---|---|
| [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) | Isaac Lab 3 kit-less + Newton/MuJoCo-Warp；独立实现 | 40 LAFAN1；单 RTX PRO 6000，1.47 小时，return 0.829，0/40 失败；有曲线、GIF、MILESTONES | 优先做第一次训练；依赖较重，含论文外的 6 维参考相对朝向观测、bf16、关闭自碰撞，不能称严格同配置复现 |
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) | mjlab/MuJoCo Warp；独立实现 | 单 RTX 5090，增加锚点观测后 1.66 小时 0.824，3.3 小时 0.904（3/40 失败）；论文观测版本约 0.56；有 BONES-SEED 数据管线与多卡 | 最适合阅读代码、核对论文和继续做大数据实验；差异、失败记录透明；BONES 筛选结果是 129,943 条，已公布高分只评估 2000 条子集，不能当完整基准 |
| [dhmbb2/supermimic_vibe](https://github.com/dhmbb2/supermimic_vibe) | mjlab/MuJoCo Warp；独立实现 | 单 RTX A6000，1.34 小时 0.833；12 小时 0.946、0/40 失败；含 PPO-Sim、PPO-WM、H×critic 消融、JSON 审计和 Slurm | 适合系统复现实验；目标输入使用相对机器人根部的位置/旋转，bf16；BONES 全量尚未完成。已有 sbatch 需改成本机 pilot 分区 |
| [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic) | UniLab；mjwarp/MuJoCo/Motrix，参考 vibed 的下游实现 | 12k checkpoint，40 LAFAN1，return 0.827227，0/40 失败；有视频和曲线 | 想使用 UniLab、多后端时可选；含额外参考相对朝向输入，缺少可直接比较的完整训练耗时/硬件信息 |
| [humble02/supermimic_mjlab_vibed](https://github.com/humble02/supermimic_mjlab_vibed) | vibed → mjlab 移植，无 Isaac Lab/Sim/Newton/USD | RTX 5090 上仿真、FK、短训练、续训、评估、视频流程验证；明确未做完整收敛训练 | 依赖路径较直接，适合作为 mjlab 移植起点；不能继承上游训练成绩 |
| [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab) | mjlab/MuJoCo Warp，模块化任务 | 提供 40 LAFAN1、在线训练、续训、全片评估及测试；未在 README 公布与论文同口径的收敛分数 | 工程结构清楚，可备选；机器人资产变更使旧 checkpoint 需要重训；关闭自碰撞 |
| [RaraAlu/Supermimic](https://github.com/RaraAlu/Supermimic) | 参考 vibed 的 mjlab 迁移；HDF5 索引、多卡分片 | RTX 5080/4070 Ti SUPER 小规模回归；CPU/Gloo 与单卡流程验证；H100/NCCL 和完整长训未验收 | 全量数据和多卡工程准备多，但当前是流程证据；README 中 8×H100 是目标资源，不是已完成成果 |
| [ZouYan93/trackerlab-supermimic](https://github.com/ZouYan93/trackerlab-supermimic) | vibed → Isaac Lab 2.3.2 / Isaac Sim 5.1 / PhysX | 母项目单 RTX PRO 6000、8096 环境、2000 iterations，23.71 分钟，0.5416、12/40 失败 | 适合已有 Isaac Sim/PhysX 工作流；尚未达到论文 LAFAN1 指标，含额外观测 |
| [JackMin7/supermimic_whole_body_tracking](https://github.com/JackMin7/supermimic_whole_body_tracking) | BeyondMimic/whole_body_tracking 中加入 MBRL | 静态站立任务；训练约 1 小时，MBRL 完成 32/32 个 60 秒站立测试；与 PPO 同 GPU 对照 | 适合入门对照；只有静态站姿，不能归为 40 动作或全库复现；32 个并行环境不是独立种子 |
| [HighTorque-Dev/supermimic-colab](https://github.com/HighTorque-Dev/supermimic-colab) | vibed 加 Colab notebook | README 主要保留上游结果；描述提到 A100/14k，但所查 notebook 没有保存可核对的运行输出 | 暂不优先；静态检查发现 notebook 搜索 model_*.pt，而训练写 ckpt_*.pt；评估传 --save_traj，而评估脚本接受 --record |

## 重点代码对应

- zita-ch：supermimic/train.py 的 wm_update / policy_update 包含 k=8 拟合、固定 WM 权重、H=32 BPTT、soft survival、critic bootstrap 与 GAE；supermimic/mdp.py 的 OBS_DIM=99（论文列出的本体项为 93），obs() 额外拼接 6D 参考相对朝向。goal() 取参考自身坐标下的未来动作。
- CuboiLeo：src/supermimic/imagination/objective.py 对应论文 Eq. 2/3；world_model/world_model.py 对应自回归世界模型拟合；training/loop.py 串联采样、拟合和策略更新；docs/EXPERIMENTS.md 记录加入/去除 anchor observation 的结果。README 的“进行中”与更详细实验记录存在更新不同步，选实验时应核对后者。
- dhmbb2：code/supermimic/train.py 的 update_policy 在 WM 权重固定时保留 imagined state→action→reward 梯度；notes/IMPLEMENTATION.md 说明目标中的参考根部位姿相对机器人计算，涉及状态估计。notes/RESULTS.md 和 results/audit/ 可核对单个实验。
- Motphys：src/unisupermimic/tasks/motion_tracking/g1/supermimic/mdp.py 中有参考相对旋转特征；训练结果应与原论文观测区分。

上述只是静态阅读和结构核对，没有完成数值正确性审计，也没有独立验证作者的训练耗时。

## 权重和数据

这 10 个仓库查询时的 Git 树没有发现 .pt/.pth/.onnx/.safetensors 文件；重点 4 个仓库的 GitHub Releases 为空。不能把 README 中出现的 checkpoint 文件名视为已提供下载。

LAFAN1 参考数据：[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)，g1/ 下 40 条 CSV，30 Hz，需转 50 Hz 并核对关节顺序与四元数约定。BONES-SEED 的下载入口与许可要求以项目数据说明为准，部分复现报告需要 Hugging Face 许可授权。训练前还应固定过滤后的 motion ID 清单；不同筛选结果不可直接比较失败总数。

本机 /mnt/data/datasets 的顶层目录未发现明显命名为 LAFAN1 或 BONES-SEED 的数据集，本次未进一步扫描其他用户数据。

## 这台服务器的建议

已阅读 /home/SERVER_GPU_RULES.md。sinfo 显示 pilot 分区纳管 4 张 nvidia_h200_nvl；GPU 分配由 Slurm 决定。本次只做 CPU 阅读与下载源码，没有启动 GPU 任务。

建议第一次只做单卡 LAFAN1：

1. 快速得到一个可评估的策略：选 zita-ch/supermimic_vibed，固定 a1508d42c9d1699a52e8233ab16d08c488441ab9。
2. 更关注严格观测口径、后续 BONES-SEED 和可解释实验记录：选 CuboiLeo/SuperMimic-Reproduced，固定 d9be385b3eeb 对应的完整 SHA（见 revisions.json）。把论文观测和加 anchor 观测分为两个实验。
3. 先在 Slurm 内做小规模仿真/FK/梯度/保存-恢复检查，测量显存、主机内存、每迭代耗时和编译耗时，再定正式预算；H200 的训练时间尚未测量，不能承诺直接复制 H100 的 1.2 小时。
4. 正式跑默认 4096 环境、H=32，训练与评估分开计时；对全部 40 条原始 clip 从首帧到末帧评估，报告 macro/frame-weighted return、失败数和逐动作结果。先以 0.82 为目标，再扩 BONES-SEED。
5. 训练输出放 /mnt/data/tianyuan，公共数据复用 /mnt/data/datasets。提交必须显式给出 GPU/CPU/RAM/时限；根据短跑实测配置资源。上游 docker/run_container.sh 0、tmux 启动命令和硬编码 GPU 索引不能直接照搬；应改成 Slurm 分配内的前台进程，继承分配到的设备，不自行绑定物理卡。

## 本地资料

- supermimic.pdf / paper.txt：论文原文及文本提取。
- sources/<owner>__<repo>/README.md：查询时 README。
- sources/<owner>__<repo>/tree.json、files.txt：查询时文件目录。
- sources/<owner>__<repo>/code/：用于静态阅读的部分文本文件。不是可直接训练的完整 checkout。
- repositories.json / revisions.json：搜索结果和查询时版本快照。源码按 main 获取，未构建不可变归档；正式训练请使用版本 SHA 重新 checkout。
