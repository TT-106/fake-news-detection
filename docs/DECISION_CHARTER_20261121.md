# 项目决策章程与 Sprint 1–4 施工路线（2026-11-21 锁定）

本文档记录 2026-11-21 团队与代理逐分支对齐（`/grill-me`）后锁定的全部核心架构决策，作为后续 CCF-B 首篇论文研发与大创阶段推进的统一基准。

## 一、8 项已锁定核心设计决策

| 编号 | 决策分支 | 锁定结论 | 关键约束与触发条件 |
| :-: | --- | --- | --- |
| D1 | **阶段主目标与范围** | 稳扎稳打冲**第一篇 CCF-B 论文**（CIKM / WSDM / EMNLP），聚焦本地 945 条 AVerImaTeC 上的 **E1–E4 闭环、60 条双盲引文审计与同骨干对比** | CHEF 中文扩展、352 条 FEVER-9 盲测对标与面向用户的交互原型系统留至第二阶段 |
| D2 | **团队人力带宽** | 完整 3 人小队（甲、乙独立盲审 + 丙第三方仲裁），每人每周稳定投入 **3–6 小时** | 所有人工任务必须拆分为高 ROI 分批离线 HTML 复核包，配套自动化校验脚本 |
| D3 | **事件隔离与划分锁定** | 采用**分级重点复核 + 预注册漏检抽样**协议，并**补齐 15 条温度校准集少数类** | ① 跨训练/留出 10 对全量双人复核；② 训练池内部 314,028 对全量图文/语义筛查 → 9 对跨活跃用途高危对 + 30 对盲混随机未命中抽检（seed=20261121）；③ 审核定锤后从 `reserve` 调入干净少数类事件组使校准集覆盖全部 4 类；审核未完前绝不置 `human_event_review_complete=true` |
| D4 | **取证底座与图像溯源** | **先抽检 AVerImaTeC 官方 Knowledge Store**（覆盖率、URL、日期、同源转载密度、反向搜图网页），若材料充分则锁定为首篇论文主取证轨（同池对标 REVEAL / AIC CTU，预算按调用步数/Token 计量） | 若官方知识库缺乏同源溯源线索，再回退到升级开放网页 + 付费搜索/反向搜图 API + Wayback |
| D5 | **算力与模型精度** | **优先本机 RTX 4060 8GB 全 GPU 跑免费 `Qwen2.5-VL-7B-Instruct` NF4**（原生单 token logits + 夜间断点续跑缓存）；待全流程跑通后再视需要考虑租云端 4090/A100 跑全精度 BF16 | NF4 与 BF16 属于不同冻结模型身份，绝不混用；若切云端 BF16 须统一重做拟合、校准与所有对比基线 |
| D6 | **对比基线梯队** | **第一梯队必做**：内部强消融（零检索 / 固定检索 / MMR 去重 / 去 $\chi$ / 匹配置换 / 诊断基线）+ 同骨干适配 **REVEAL** 与 **AIC CTU** + 已具基础的 **FIRE 四分类适配** | **DEFAME**（34 项依赖与多模态适配）作为第一梯队跑通后的加分项补齐；**SAFE-MR** 因无确认官方代码仅在相关工作与机制对照中如实讨论 |
| D7 | **E2 遮罩与软逻辑规则** | 保留 E2 全部四项消融，但**工具提效 + 务实定界**：为 `no_spatiotemporal` 开发“自动预高亮时空线索 + 双人离线拉框复核”HTML 工具；`no_rules` 仅在 60 条验证集中确有跨原子可靠逻辑关系的样本构建 `PREVERIFIED_SCOPED_RULES_V1` | 缺可靠逻辑关系的样本按协议保持空规则，如实报告真实效应大小，绝不伪造增益 |
| D8 | **科学假设与评估兜底** | 预注册**三维立体评估**：全集 + 预定义子集（同源转载/多源复用、旧图复用、单源/无来源疑点）+ 质量—成本帕累托曲线 | 若全体中可被来源核查翻盘的机会率偏低，则诚实报告真实发生率，突出“同源子集显著降错 + 同质量下更省检索预算”的优势 |

---

## 二、Sprint 1–4 施工路线图

### Sprint 1（第 1–2 周）：双探路 + 事件隔离定锤
1. **人工侧**：甲、乙独立完成跨边界 10 对表（`outputs/event-review-priority-20261121/`）与训练池内部 39 对盲审表（`outputs/event-review-internal-20261121/`），丙完成分歧仲裁 → 锁定正式 `experiment_split.json`（含 `train:466` 隔离补位与 `temperature_calibration` 四类均衡补位）。
2. **技术侧**：完成 AVerImaTeC 官方 Knowledge Store 20 条先导抽检审计（`outputs/knowledge_store_probe_20261121/`），拍板主取证轨，并接入高覆盖图文+反向搜图检索适配层。
3. **基础设施**：将 `src/research_evaluation.py` 中的 `paired_bootstrap` 升级为事件组级（Cluster Bootstrap），并补齐人工回收转规范训练 JSONL 的自动验真转换器。

### Sprint 2（第 3–6 周）：三类监督标注 + 模块拟合冻结
1. 在锁定划分后的 `pilot`（20 条）与 `module_fit`（45 条）上生成真实归档证据、固定证据编号、原子声明与冻结 Qwen 输出。
2. 先做小批试标验证双人一致性，再由甲、乙独立完成来源/质量/充分性三类监督标注与丙仲裁，同步完成 60 条 `mechanism_validation` 的独立来源评测标注。
3. 拟合冻结来源/质量/充分性三个独立 LR 与独立四类共享温度 $T$，锁定充分性阈值与预算点。

### Sprint 3（第 7–11 周）：E1 / E2 / E3 与第一梯队基线
1. 用离线预高亮 + 拉框 HTML 工具完成 60 条 `no_spatiotemporal` 双人遮罩与 `PREVERIFIED_SCOPED_RULES_V1` 规则整理，跑通 **E1 + E2**。
2. 在 180 条 `action_train` + 60 条 `mechanism_validation` 上展开真实共同历史分支（夜间分批 + 断点缓存），训练岭回归收益估计器并跑通 **E3**（`full / no_chi / p(1-p) / permutation` + 预注册子集拆分）。
3. 跑通第一梯队同骨干对照（固定检索、MMR 去重、REVEAL、AIC CTU、FIRE 四分类适配），在机制验证集上**提前锁定最强基线与预算工作点**。

### Sprint 4（第 12–16 周）：E4 终局闭环 + 双盲审计 + 首篇 CCF-B 投稿
1. 解锁 152 条官方留出集执行一次性规范 **E4** 评测，输出多预算工作点质量—成本曲线。
2. 导出 60 条（至多 120 份）盲审去敏引文包，由团队完成双盲配对人工引文审计与仲裁，计算“有据正确回答率”及 95% 事件组级配对区间。
3. 视进度补齐 DEFAME 适配及/或云端 BF16 复核；完成首篇 CCF-B 论文初稿与投稿。

---

## 三、2026-11-21 探路实证记录

- **训练池内部筛查与盲审包**（已完成）：`outputs/event-review-internal-20261121/` —— 全量 314,028 对筛出 9 对跨活跃用途高危关联（含 4 对 `action_train <-> mechanism_validation`、3 对 `module_fit <-> action_train`、1 对 `module_fit <-> temperature_calibration`、1 对 `mechanism_validation <-> pilot`），与 30 对预注册未命中抽检盲混为 39 对离线 HTML 复核表，零泄漏校验通过；并在 `reserve` 确认 7 个完全干净的单条少数类事件组供校准集补位。
- **AVerImaTeC 官方 Knowledge Store 20 条抽检（已完成，裁决：GO）**：详见 `outputs/knowledge_store_probe_20261121/KNOWLEDGE_STORE_VERDICT.md` 与 `pilot20_knowledge_store_audit.json` —— 仅通过 HTTP Range 拉取 160 MB 审计全部 20 条先导，**文本网页覆盖 20/20 (100%，20,041 篇)**、**反向搜图网页覆盖 20/20 (100%，均值 13.9 篇/条)**、**通讯社/归因线索 20/20 (100%，618 篇)**、**跨域近重复对 13/20 (65%)**，完胜 11-18 开放网页先导（50% 证据覆盖、0% 反向搜图）。同时检出两项入库前必加清洗防线：① 剔除 Twitter/X 等 JS 拦截壳页面；② 预注册核查站/目标 `article` 过滤防泄漏规则。正式锁定以官方 Knowledge Store 作为首篇 CCF-B 论文主取证轨（`train/text_related.zip` 6.25 GB + `val.zip` 8.8 GB）。
- **官方 Knowledge Store 20 条三策略真实 Qwen 7B NF4 基准（已完成，非正式先导）**：详见 `outputs/pilot/ks-pilot20-v1/report.json`（使用 `src/knowledge_store.py` 清洗 20,041 → 11,289 篇洁净可回放文档，屏蔽 152 篇核查站与 152 篇事后日期页，全 GPU 60 次单 token logits 前向平均仅 **1.75 秒/次**）：
  - `ks_text_only`（纯文本 top-6）：Macro-F1 `0.1964` / Acc `35.0%`（Refuted F1 `0.500`）
  - `ks_dual_text_ris`（文本+反向搜图 3+3）：Macro-F1 `0.2381` / Acc `50.0%`（Refuted F1 `0.667`）
  - `ks_dual_mmr`（双路 + MMR $\lambda=0.65$）：Macro-F1 **`0.2946`** / Acc **`60.0%`**（Refuted F1 `0.750`，NEI F1 `0.429`）—— 且在 MMR 去冗余之后，**仍有 12/20 (60%) 样本在 top-6 证据池中存在跨域高相似/同源核查机会、8/20 (40%) 存在同域多篇报道**，证实式(1)与式(3)在强基线之上具备充裕实证发挥空间。
- **Sprint 1 基础设施补齐（已完成，离线测试 `268 passed, 3 skipped`）**：① `src/research_evaluation.py::paired_bootstrap` 升级支持事件组级 Cluster Bootstrap 并接入 `src/experiment_runner.py` 的 E3/E4；② 补齐双人回收转规范训练/评测 JSONL 验真工具 `tools/convert_annotation_return.py`；③ 完成 `src/knowledge_store.py` 清洗/检索/MMR/快照验真模块、`OfficialKSEvidenceTools`（完整实现 `EvidenceTools` 协议并支持导出 `ArchivedEvidenceTools` 回放清单）及 `tools/download_knowledge_store.py`。
- **官方 Knowledge Store 训练集全包落地锁定（已完成，`val.zip` 严格封存未触碰）**：`data/averimatec/knowledge_store/train/text_related.zip`（`6,716,122,829` 字节，全包 SHA-256 `6c4e8e2c92e3c352e61c0ae447da0da619e18c8e9c33f269ddfee7310eba6eb9`，`793` 条文本库 + `793` 条反向搜图库共 `1,588` 项 ZIP 条目 CRC32 全部校验通过，校验清单见 `data/averimatec/knowledge_store/train/manifest.json`）。
