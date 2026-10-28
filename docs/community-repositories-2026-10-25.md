# SuperMimic 公开仓库索引（2026-10-25）

GitHub 名称/描述搜索命中 53 个（包含本项目），另补 MorphMimic。这里记录的是文档声称与静态代码检查，未重新运行第三方 GPU 实验。共取得 50 份 README，26 个项目读取关键源码；其余按目录和文档筛查。完整 SHA、读取文件、证据边界见 [JSON 索引](community-repositories-2026-10-25.json)。深入结论见 [代码与结果复核](community-code-audit.zh-CN.md)。

| 项目（固定提交） | 类型 | 公开结果/状态 | 核对注意点 |
|---|---|---|---|
| [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed/tree/a1508d42c9d1699a52e8233ab16d08c488441ab9) | 重点 | LAFAN1 0.829，0/40；1.47 h，RTX PRO 6000 | 99维观测含6D相对朝向；bf16；关闭自碰撞；完整长度分母。 |
| [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/d9be385b3eeb13a638fb33fb200d3bfd9be9839f) | 重点 | LAFAN1 +9位姿 0.824/0.904；5/3失败；纯参考约0.56 | 基线；评估分母是已执行步数；BONES高分仅2000子集。 |
| [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic/tree/faf7818e8abac3ec6abb8f00e6e904d3577f800b) | 重点 | LAFAN1 0.827227，0/40；12k checkpoint | vibed下游；相对朝向；完整长度分母；更新UniLab依赖。 |
| [Zhengtianjin/go2-supermimic](https://github.com/Zhengtianjin/go2-supermimic/tree/054a13c13285a1c94b2e69dc8ac7331f9a8c71cf) | 扩展 | Go2追逐/躲避任务 | 非G1动作库基准，不纳入LAFAN排名。 |
| [JackMin7/supermimic_whole_body_tracking](https://github.com/JackMin7/supermimic_whole_body_tracking/tree/553a7d32dc5fc00cbe6de5ee46479af2fcceeda8) | 扩展 | 32/32个60秒站立rollout | 静态站立；并行环境不是独立种子。 |
| [btx0424/rethinking-supermimic](https://github.com/btx0424/rethinking-supermimic/tree/ecdaeb6a916a16cf7eb69b19226c17be31a6bc27) | 代码 | 未见README收敛分数 | 有相对朝向/位置、contact特征和actor选项；默认解释需看Hydra配置。 |
| [Baojch/SuperMimic_Dyn](https://github.com/Baojch/SuperMimic_Dyn/tree/28b15acb68ab3871f2ae1479964d53457ac066c6) | 重点 | 验证3939/3966完成；完整片段回报0.86454 | full689；双4090约4.09h；自定义95323训练片段、均衡采样/移除手部几何；非论文全库。 |
| [Shimingyi/allmimic](https://github.com/Shimingyi/allmimic/tree/cb334ff674b2f57248f33f8c21b7fb526fbdf8cc) | 重点 | LAFAN1 0.94，0/40，2h；困难集306/306 | 上身残差、latent128、采样/值裁剪等改变；困难集与论文ID和数量不同。 |
| [Renforce-Dynamics/trackerlab-supermimic](https://github.com/Renforce-Dynamics/trackerlab-supermimic/tree/29d24524f2f731bf4df75b319c3e76544165849e) | 重点 | LAFAN1 0.8694663，0/40；32k checkpoint | 原ZouYan93仓库迁移；PhysX；旧列表只列2k结果已过时。 |
| [YanYuan98/Supermimic_mod](https://github.com/YanYuan98/Supermimic_mod/tree/6932f1cd33532b4e84ff173df926d8195a58c45a) | 重点 | TF32单5090约3h 0.850，0/40；纯参考0.564，5/40 | 6D相对朝向；目标未来轨迹扩展；完整长度分母；有明确纯参考对照。 |
| [panhainan2020/SuperMimic-MuJoCo](https://github.com/panhainan2020/SuperMimic-MuJoCo/tree/db045afa9c52458f213e2c66037387e9c109a03a) | 重点 | LAFAN1 0.871322，40/40；100STYLE 0.804516，3569/3573 | vibed移植；99维观测；98k固定checkpoint；完整分母；全局根误差仍0.643m。 |
| [tianshuwu/supermimic-retargeted](https://github.com/tianshuwu/supermimic-retargeted/tree/ff69a0e5305126eb5b80b323bcd815805bc3df4e) | 重点 | G1 LAFAN基线0.90736，0/40 | G1/A3/X2移植；anchor_obs；同时提供完整片段加权与旧已执行步分数。 |
| [humble02/supermimic_mjlab_vibed](https://github.com/humble02/supermimic_mjlab_vibed/tree/beb18b33ba0e05baaddb9a4c608a7b8b4bc988ce) | 流程 | mjlab迁移/续训/短训验证 | 未完成长训，不能继承vibed成绩。 |
| [RaraAlu/Supermimic](https://github.com/RaraAlu/Supermimic/tree/60f2f9fb31c068ed2ca16209548915f1382a6d50) | 流程 | 多卡/索引HDF5流程及小规模验证 | H100/NCCL/长训未验收，8卡是目标配置。 |
| [KyrieYoung926/supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy/tree/7e63e3bbd5150d55afbdece28278781800f3ed21) | 部署 | G1 MuJoCo/真机部署演示 | 训练基于Cuboi；部署项目，不是独立完整基准。 |
| [kangzejian1896/supermimic-g1-reproduction](https://github.com/kangzejian1896/supermimic-g1-reproduction/tree/5daf8586f26da3ef3bd99bdd875c8f8ce1b4e18e) | 重点 | 128验证片0.92897，0失败；84287训练片0.94424，9失败 | A800约25.86h；验证参与选模；paper_protocol可选；非129514片集。 |
| [wenconggan/SuperMimic_e1](https://github.com/wenconggan/SuperMimic_e1/tree/103adbf754f21e5081077ea9e774c722737d5529) | 异本体 | E1 21DoF，0.8559，40/40；164k | 不同机器人；完整动作与随机8秒演示分开。 |
| [mode1223345/SuperMimic-perceptive](https://github.com/mode1223345/SuperMimic-perceptive/tree/3d38bacbaf5bee7e782bdada6a6da678264d0bfa) | 扩展 | 高程图AE预训练/恢复流程已验证 | 四组控制对照尚在进行；保留vibed成绩非本分支结果。 |
| [fan-ziqi/supermimic-agy](https://github.com/fan-ziqi/supermimic-agy/tree/8513519b1b9b2d18ac61089748ddebbfe1bcf52b) | 需核验 | 公开full_eval 0.730896，15/40失败 | 13条逐片分数超出完成比例理论上界；当前代码与历史JSON口径不一致。 |
| [OMEGA-i/SuperMimicKit](https://github.com/OMEGA-i/SuperMimicKit/tree/6697ff8ca53adc9233ded12a7c8777610ca1154d) | 重点 | G1 train40 0.798961，3失败；val37 0.3103，31失败 | 保留55.5k，100k策略40/40失败；有训练退化证据。 |
| [Breezewrf/supermimic_mjlab](https://github.com/Breezewrf/supermimic_mjlab/tree/aab90addd0e783b007691442907bc8c9ce145970) | 流程 | 历史100轮短训0.001681，0/40完成 | 明确只是闭环验证；修复后尚无论文收敛结果。 |
| [zzhanYoung/supermimic-lite](https://github.com/zzhanYoung/supermimic-lite/tree/e4a67b0f61701e7a47fc96392b4fe8342ce5a672) | 框架 | README仍是mjlab介绍 | 树中以mjlab为主，未见可确认的SuperMimic成绩。 |
| [enffft/supermimic](https://github.com/enffft/supermimic/tree/fa033b8e3b7c5e91bf3ef98a62d40e4d049361be) | 副本 | README与Cuboi相同 | 未找到独立于上游的README结果证据。 |
| [yangkunpeng-coder/supermimic_reproduce](https://github.com/yangkunpeng-coder/supermimic_reproduce/tree/31ad48ff1e4a426dd5c9ede08ad7dbc4f67bce9f) | 占位 | 只有LICENSE | 无训练实现/结果。 |
| [CesarLiu/smimic](https://github.com/CesarLiu/smimic/tree/0c1cda553c667e4fe91dc4f141184938b07db314) | 扩展 | 继承UniLab的0.827227表格，另有Go2迁移/鲁棒训练 | 不能把继承的12k表格算为一次独立复现。 |
| [BlackOtters/supermimicA3](https://github.com/BlackOtters/supermimicA3/tree/ae20f02f2a5391f3243b71f4e9d7ee7cad9062eb) | 异本体 | A3 Ultra v10四卡刚性配方 | 明确未声称收敛或真机迁移；6D相对朝向目标。 |
| [jackzedgar/SuperDex](https://github.com/jackzedgar/SuperDex/tree/f0ae61c64b65d10ef4f80e92394019b913ceee32) | 占位 | 灵巧手世界模型设想 | 目录只有README，无实现或实验。 |
| [PrimebotTech/SuperMimic](https://github.com/PrimebotTech/SuperMimic/tree/118338b19dcd14f768d1e7de5a48d04b78987688) | 异本体 | Q1 V2 22关节，训练质量仍验证 | 非G1，不作为论文LAFAN结果。 |
| [yagami-light7/SuperMimic-codex](https://github.com/yagami-light7/SuperMimic-codex/tree/d2b1875ac4319a56607485109a6e972592b66df1) | 流程 | M0 16/64环境300步通过 | 无训练策略或权重，尚待下一阶段。 |
| [Go-Youngster/supermimic-reproduce](https://github.com/Go-Youngster/supermimic-reproduce/tree/02db3d5ae75a54c76ddec969c8bd63cc8fefc661) | 重点 | A100 3h到0.82；15.7h最佳0.875，0/40 | 93维本体+710维含相对朝向goal；bf16；完整分母；单种子。 |
| [Alchedata/supermimic-locomujoco](https://github.com/Alchedata/supermimic-locomujoco/tree/cc1375d48a38bde062d088ffd132122e0cd7b7da) | 扩展 | H1 40动作，5k多种子，原生return约800 | JAX/LocoMuJoCo；任务奖励和本体不同，不能把原生return与0.82比较。 |
| [KunYing-Lee/SuperMimic](https://github.com/KunYing-Lee/SuperMimic/tree/95eeba6e9a0deed9d5c16623dc3ac2650204bba8) | 代码 | mjlab移植及选片演示 | 有论文对齐和误差对照工具；README未给本分支完整收敛分数。 |
| [dhmbb2/supermimic_vibe](https://github.com/dhmbb2/supermimic_vibe/tree/072f0910dadc9d95c720e625b546120e9ce917c9) | 重点 | A6000 1.34h 0.833；12h 0.9463，0/40 | 73×10相对根位姿goal；bf16；完整分母；逐片JSON可重算。 |
| [joreb/supermimic-yai](https://github.com/joreb/supermimic-yai/tree/7e82875e1c6ffec11a15aa24a61a198cfb718e07) | 流程 | WIP | 部分笔记子模块私有；无公开收敛结果。 |
| [yuntian137/supermimic_lab](https://github.com/yuntian137/supermimic_lab/tree/466340db20fd180851632033e48e80662267458b) | 代码 | 4h34m；36k；最佳0.866784，40/40 | vibed工程适配；复评0.866452；训练库覆盖。 |
| [Tianao-Zhang/supermimic_dualhand](https://github.com/Tianao-Zhang/supermimic_dualhand) | 空仓 | 仓库size=0，无提交可读 | 无法评估实现。 |
| [chu-fa/isaaclab-supermimic](https://github.com/chu-fa/isaaclab-supermimic/tree/d350e4378c6dc08a3f2a1d35d47b45d83db0e081) | 扩展 | IsaacLab通用接口与A3有界验证 | 明确收敛/后端等价/论文性能未证明。 |
| [tianny0078/supermimic-repro](https://github.com/tianny0078/supermimic-repro/tree/5135b7c7a721be6bd9db613ac657d3f72609c918) | 本项目 | H200 3h；最佳0.569920，3/40失败 | reference-only/full-state critic；32000采集步，约12784更新；完整N-1分母。 |
| [xinruilee04/supermimic-test](https://github.com/xinruilee04/supermimic-test) | 空仓 | 仓库size=0，无提交可读 | 无法评估实现。 |
| [samindaa/supermimic_mjlab](https://github.com/samindaa/supermimic_mjlab/tree/9adcad37634058eea7cd53f0d9a02d2e8b944401) | 移植 | 本地4k评测0.732，4/40失败 | README的0.829/1.47h是原IsaacLab结果；不得继承为mjlab结果。 |
| [HighTorque-Dev/supermimic-colab](https://github.com/HighTorque-Dev/supermimic-colab/tree/f9ebd42d0b484b93747918e0ae7306b90f73a172) | 移植 | README完整继承vibed成绩 | notebook入口/权重名不一致在旧审查已记录；无独立结果证据。 |
| [shangchaohu718/supermimic-pera](https://github.com/shangchaohu718/supermimic-pera/tree/0e89e55187a800341e609129b5d5200a607a467c) | 扩展 | 本分支6k 0.715678，34/40完成；另有滑板技能 | 继承UniLab；未完成12k目标，交互技能是另一任务。 |
| [yhx1203/supermimic_mjlab](https://github.com/yhx1203/supermimic_mjlab) | 空仓 | 仓库size=0，无提交可读 | 无法评估实现。 |
| [LaoYang1994/SuperMimic-Rep](https://github.com/LaoYang1994/SuperMimic-Rep/tree/6f8db6b1e9e02062c44aee358d69c6c8852745ee) | 重点 | LAFAN1 18k更新0.821，1失败；BONES 44k更新0.934，210/129511失败 | 默认beyondmimic_seq含每未来帧相对朝向；修正完整分母；未达到全库失败数。 |
| [liuskywalkerjskd/supermimic_oli](https://github.com/liuskywalkerjskd/supermimic_oli/tree/2dc90d90e89fb2e097e985c84c8b99a77708fa5f) | 异本体 | Oli 31DoF 0.6783，36/40；4090约2.20h | 不同本体；继承G1 MILESTONES不是Oli训练记录。 |
| [CumulusAlpha/Dex-Supermimic](https://github.com/CumulusAlpha/Dex-Supermimic/tree/5ca57628af3666a9b2878c00f461f43d6edb5b61) | 扩展 | Wuji Hand2方块重定向，有8k权重Release入口 | 非人形跟踪；README未给与论文可比统计。 |
| [jboyxs/Supermimic_for_transcube](https://github.com/jboyxs/Supermimic_for_transcube/tree/02aa010a5d1e76d8b403f36e7888cf45022f28e5) | 占位 | 空README、少量项目文件 | 无可核对实验总结。 |
| [liudi159/supermimic-backflip-5090](https://github.com/liudi159/supermimic-backflip-5090/tree/5b313d75e804cdd2ff1ce9bdcfa899a98eca856b) | 单动作 | 单5090约93min；完整翻转但严格落地验收未过 | 一个动作一个种子；接触审计值得参考，不能称全库复现。 |
| [sxh-kk/SuperMimic_Recurrence](https://github.com/sxh-kk/SuperMimic_Recurrence/tree/c779ca4346612b907688d42337f293aaf3240b8d) | 流程 | 4090/CUDA12适配 | 明确尚无论文指标复现结论。 |
| [WXyuany/supermimic-m5](https://github.com/WXyuany/supermimic-m5/tree/42451591718e1a9a5ec601366997f47c0ad180a3) | 重点 | M5+CPU；0.7081，25/40完成 | +9位姿；已执行步分母；按公开逐片汇总改用全长度约0.5826。 |
| [Mark054/SuperMimic_TienKung](https://github.com/Mark054/SuperMimic_TienKung/tree/71c9f0e677c498edf387121fea27bf146bfcd423) | 异本体 | 天工3.0训练/数据/演示 | README未提供完整benchmark收敛分数。 |
| [luoshaqi/g1-moonwalk-supermimic](https://github.com/luoshaqi/g1-moonwalk-supermimic/tree/2b445d40e874629d7874f95986f155e229e67924) | 单动作 | 微调3k 0.9453，300/300步完成 | E009预训练后拟合一条CMU动作，非从零/泛化。 |
| [NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real](https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real/tree/99d310d5460333aabe853337b1d55ee27940a3fe) | 重点 | 自报BONES 0.9305，46/129491；发布真机舞蹈模型 | LAFAN e2/e3表沿袭Cuboi；当前eval仍已执行步分母；主结果硬件/耗时未列。 |
| [lzyang2000/morphmimic](https://github.com/lzyang2000/morphmimic/tree/2e35049e9519e511889ef987e4acde55d4f5b017) | 补充 | 多本体共享WM/策略、真实G1部署报告 | 按正文链接补入；LayerNorm/value clip/bf16/避片尾/延迟随机化；非原设置复现。 |

