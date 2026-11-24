# AVerImaTeC 公开对照与评分口径

核对日期 2026-11-18。当前能核实的 FEVER-9 官方盲测最高公开成绩是 **HUMANE／VILLAIN 的 0.5455**。它来自 352 条官方测试及证据达标后才计分的 AVerImaTeC score。在已查的官方结果和一手论文中，未找到后续以同一测试和评分协议超过该分数的公开结果。该检索结论不代表已穷尽所有发表或未发表结果。[FEVER-9 最终榜](https://fever.ai/task.html) · [组织者论文 §2.4、表 3](https://arxiv.org/html/2602.11221v2) · [VILLAIN 论文](https://aclanthology.org/2026.fever-1.9/)

## 本地数据与排行榜

官方数据有 793 条训练、152 条开发、352 条测试，合计 1,297 条。本地人工包中的 945 条对应官方训练和开发。虽然项目把 152 条开发数据留作内部终评，它仍与官方排行榜的 352 条测试分属不同划分。核对了作者发布版本 `7e77fb26a1ba86159ff5ba27b59f373b450d6fd0` 的目录 API 与本地 `git hash-object --no-filters`，两个 JSON 的 Git blob OID 完全一致。本地 `train.json` 为 `ccf4ae863bc2f1214fe86aa4878127fca0128b92`，`val.json` 为 `dc4657b3e1ff209c9ac09bc65b27b70062c64578`。该版本还列出 `test_data.zip`，当前项目的本地数据目录只有训练和开发 JSON、图片和下载记录。取得测试输入后仍须确认官方评分入口和参考答案可用性。[组织者论文表 1](https://arxiv.org/html/2602.11221v2) · [作者数据固定版本](https://huggingface.co/datasets/Rui4416/AVerImaTeC/tree/7e77fb26a1ba86159ff5ba27b59f373b450d6fd0) · [发布目录 API](https://huggingface.co/api/datasets/Rui4416/AVerImaTeC/tree/7e77fb26a1ba86159ff5ba27b59f373b450d6fd0) · [官方测试评测页](https://huggingface.co/spaces/FEVER-IT/AVerImaTeCTest)

本地 152 条中 141 条是 Refuted，恒猜 Refuted 的裸准确率已达 92.76%。官方主分数是在全部测试声明上取平均，只有证据过阈值且四类标签正确的样本才计 1 分，证据门槛设为 0.3。证据文本与图片匹配以人工参考为准，主办方公开脚本默认 `google/gemma-3-27b-it`。官方赛道和原始数据论文采用的评委不同。赛道论文还限定每条最多 10 个证据项，而单项长度在赛道网页与论文分别写为 1,500 字符和 1,500 tokens。正式比较应锁定评分入口、提交格式和裁剪规则。标记了暂缓的样本在官方全量评分中仍有位置，本项目另报同覆盖率风险和使用成本。[本地类别分布](../../outputs/progress/results_audit.json) · [赛道要求与更新](https://fever.ai/task.html) · [组织者论文 §2.4](https://arxiv.org/html/2602.11221v2) · [官方离线评分代码](https://github.com/abril4416/AVerImaTec_Shared_Task/blob/main/prepare_submission/eval_offline.py)

## 先对比谁

| 对照 | 官方 352 条测试得分 | 复现实验中的用法 |
| --- | ---: | --- |
| HUMANE／VILLAIN | 0.5455 | 比公开全系统成绩的主要参照。使用 Gemini 2.5 Pro、扩展的主办方知识库及多阶段证据处理。发布了代码，原系统资源与本地 7B 不同。[论文](https://aclanthology.org/2026.fever-1.9/) · [代码](https://github.com/ssu-humane/VILLAIN) |
| ADA-AGGR | 0.5369 | 检验榜单领先是否稳健。其赛道方案用了 Gemini 3 Pro 及不同检索模型。[组织者论文表 3–4](https://arxiv.org/html/2602.11221v2) · [论文](https://aclanthology.org/2026.fever-1.7/) |
| AIC CTU | 0.3466 | 可移植的双路检索结构。文本检索含 MMR 重排，再结合反向搜图，以一次 GPT-5.1 调用生成结论与证据。换成本地 Qwen 后按同骨干改造版记录。[论文](https://aclanthology.org/2026.fever-1.11/) · [代码](https://github.com/heruberuto/AVerImaTec_Shared_Task) |
| XxP | 0.2557 | Qwen3-VL-8B 的近规模公开参照。其最终参赛方案没有在检索中用视觉证据。[论文](https://aclanthology.org/2026.fever-1.10/) · [组织者论文表 4](https://arxiv.org/html/2602.11221v2) |
| REVEAL | 0.2358 | 优先用于核对同骨干效果。Qwen2.5-VL-7B-Instruct 加迭代证据更新，依赖固定知识库和重排等组件。仍须锁定量化、取证与预算后重跑才有公平比较。[论文](https://aclanthology.org/2026.fever-1.8/) · [组织者论文表 4](https://arxiv.org/html/2602.11221v2) |
| 主办方基线 | 0.1136 | 用于接通提交和评分链路。公开代码支持多骨干，换骨干后的结果要单独记录。[代码](https://github.com/abril4416/AVerImaTec_Shared_Task) · [最终榜](https://fever.ai/task.html) |

FIRE 原文评估文字原子声明二分类。DEFAME 原文表格的 AVeriTeC 是文字基准，其他评估包括 VERITE、MOCHEG 和 ClaimReview2024+。SAFE-MR 评估 NewsCLIPpings、VERITE 二值映射、XFacta。它们可以启发同条件复现或方法对照，论文数字不能填进 AVerImaTeC 官方榜对照列。[FIRE 原文 §4.1](https://arxiv.org/html/2411.00784) · [DEFAME 原文](https://proceedings.mlr.press/v267/braun25b.html) · [SAFE-MR 原文 §4–5](https://arxiv.org/html/2610.06708v1)

## 方法需要回答的问题

SAFE-MR 已讨论同源报道不构成独立佐证，还做了充分性监督、证据删减和污染测试、固定覆盖率风险。AIC CTU 已用 MMR 降低检索冗余。因此应测试更具体的假设：显式估计来源依赖能否在**同一初始证据、同一模型、同等检索上限**下更准确地决定何时停止、何时寻找独立来源，并由此提升证据达标的正确率或节省取证成本。需要同时对比简单 URL/文本去重、按已知来源合并、去除 χ 的同结构策略，并检查误合并会不会伤害真正独立的来源。以上是实验假设，收益须由配对实验验证。[SAFE-MR 原文 §4.2–4.4](https://arxiv.org/html/2610.06708v1) · [AIC CTU 检索实现说明](https://github.com/heruberuto/AVerImaTec_Shared_Task#text-retrieval)
