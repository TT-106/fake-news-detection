# 项目交接

## 先确认做到哪一步

- 已有冻结 Qwen 的本地图文推理、固定训练池先导，以及官方知识库的文本与反向搜图证据检索。先导的评分与官方 FEVER-9 的证据加结论联合评分使用不同口径。
- 事件用途划分经过甲乙原审及署名 AI 裁决。AI 读过两份原审，正式材料仍记录 `human_event_review_complete=false`。
- Sprint 2 已形成阶段标注任务包。ZCode 中 GLM 的返回按机器弱监督处理。来源、质量、充分性三个逻辑回归模型暂无正式拟合成品。开发数据、机器回收、人工复核与正式评测结果分别记录。
- 正式 E1–E4 仍需合规三类监督、实际工具回放、独立来源评测、时空遮罩和引文审计。其中哪些步骤会检查具体材料，见 [数据与模型接口](DATA_CONTRACTS.md) 和 [实验执行器](../src/experiment_runner.py)。

## 接手顺序

1. 按 [环境和资源下载](SETUP.md) 装好 CPU 环境，运行测试。需要核对真实图文入口时，下载固定权重和训练集，按单条 GPU 检查验证。
2. 从项目负责人取得下面清单中与本次工作相关的归档，按原相对路径恢复。沿用已经固定的事件划分、先导样本与请求。样本或材料发生变化时另开一次运行并记下原因。
3. 继续 Sprint 2 时先核对已发阶段包、原始回收和冻结证据。同一题的多个 GLM 会话按机器复标登记。先用 [回收校验工具](../tools/validate_annotation_return.py) 核查任务和证据绑定，再按 [字段契约](DATA_CONTRACTS.md) 组织监督文件。事件审核、机器标注的实施细节见 [Sprint 2 协议](SPRINT2_AI_EVENT_PROTOCOL_20261122.md)。
4. 完成独立监督、正式配置和工具归档后，运行准备检查。确认材料、预算、基线及终评授权后，再执行正式运行。每次实验通过 [实验账本](PROGRESS_TRACKING.md) 记录独立 ID 和输出目录。

```powershell
Copy-Item config/experiment.example.json config/experiment.json
$py = '.\.venv\Scripts\python.exe'
& $py -X utf8 -m src.real_averimatec_pipeline --config config/experiment.json
```

`outputs/preflight/preflight.json` 记录阻塞项。退出码 `2` 表示准备不完整。仅装训练集时，公开检出缺少用于数据核查的留出文件与锁定收据，这属于预期。旧的候选分组入口 `run_all_preliminary_experiments.py` 也写入 `outputs/preflight/`。在恢复了锁定划分和原始收据的工作区，它核验已锁定划分，且保留 `val.json` 封存。需要正式运行时用 GPU 环境，增加 `--run --output-dir outputs/<新运行ID>`。

## 从原工作区恢复的材料

下列路径以原工作区根目录为基准。重建环境时从负责人备份按原路径恢复，含人工原审与盲审映射的材料由负责人通过私下渠道交接。

| 路径 | 用途 |
| --- | --- |
| `data/annotations/experiment_split.json`、`data/annotations/event_review_protocol_ai_v1.json`、`data/annotations/event_review_returns_20261122/` | 锁定事件划分、协议信息和原始双人回收 |
| `outputs/event-review-ai-20261122/`、`outputs/event-review-adjudication-20261122/`、`outputs/method_repair/dataset_audit.json` | AI 裁决、复核与封存前审计原件。签署材料必须按原字节保留 |
| `deliverables/sprint2-fixed-20261122/ZCode阶段标注_sprint2-stage-6f70b225b516.zip`、`outputs/sprint2-fixed-20261122/coordinator/` | 阶段任务、分发收据和协调记录。协调者的盲审映射只交协调者 |
| `outputs/sprint2-fixed-20261122/`、`data/annotations/machine_annotation_protocol_glm_v1.json` | 冻结证据、模型调用、机器标注口径和采集状态 |
| `outputs/pilot/`、`progress/runs/`、`progress/events/`、`progress/attempts/` | 已执行的先导结果、性能指标、运行日志及历史源文件快照 |
| `PROGRESS_FACTS.md`、`研究进度.html`、`backups/` | 核查证据入口、进度页及撤回版本，供回溯 |

公开仓库包含运行脚本及冻结训练先导所依赖的历史候选划分（ID、事件组和哈希）。其中 [候选划分](../outputs/method_repair/provisional_split.json) 用于校验既有先导。它与正式锁定划分是两份不同材料。历史研究文档中的本地报告链接需要恢复上表或相应原始归档后才能打开。第三方基线通过 [上游提交和接入说明](BASELINE_REPRODUCTION.md) 恢复到 `third_party/`。
