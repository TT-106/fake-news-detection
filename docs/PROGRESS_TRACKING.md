# 研究进度页与追加式实验账本

原工作区的 `研究进度.html` 可直接双击打开，生成时嵌入数据。公开检出可从模板生成空账本；恢复历史记录见[交接说明](HANDOFF.md)。真实先导、受控仿真、工程测试与正式评测分别登记。

## 每次继续工作的入口

```powershell
python -X utf8 tools/progress.py build
python -X utf8 tools/progress.py serve --port 8765
```

浏览器打开 `http://127.0.0.1:8765/`。服务器只读、仅绑定本机，只公开进度 HTML 与 `/progress/data.json`，不会公开整个项目文件夹。页面在 HTTP 模式可检测数据指纹变化；双击文件模式需在重新生成后刷新页面。关闭服务器不会丢失账本。

源数据保存在：

- `outputs/progress/module_audit.json`：按申请书核对的模块验收项、证据和阻塞；进度取验收项的完成比例，不是主观工时百分比。
- `outputs/progress/results_audit.json`：数据、环境、历史结果有效性、基线与核查记录。
- `progress/runs/<id>.json`：每次实验的独立记录，使用排他创建，拒绝相同 id 覆盖。
- `progress/events/<uuid>.json`：复核、完成和撤回事件，追加写入。
- `progress/attempts/<id>/`：通过包装器运行的执行前源码副本、配置/划分及结果副本、stdout、stderr；中断的实验仍保留 running 轨迹。
- `progress/attempts/<id>/environment.json`：本次 Python、平台和安装包版本快照；不导入重模型，也不证明指定模型或 GPU 已验收。
- `progress/dashboard.template.html`：HTML 模板；`progress/data.json` 和根目录 HTML 是可重新生成的视图。

该目录未依赖 Git 提交来证明版本。工具对根目录 Python、requirements、src/tools Python 和 config JSON 计算源码内容 SHA-256；包装执行将对应文件按原相对路径复制到 `progress/attempts/<id>/source/`，清单记录每个副本路径与 SHA，并在生成页面时重新核验。以后源码更新不会覆盖旧运行使用的源码副本。数据集与模型权重不在源码副本中，仍须保留与清单一致的原件。审计资料会随人工重新核查更新，单纯运行 build 不会重新执行科学审查。

## 每个版本跑完自动归档

先复制 `progress/run.example.json` 为本次唯一清单，填好数据版本、经过人工事件复核的 split、冻结模型（含量化版本）、评测协议、预算与样本量。示例故意含无效占位符，不能被误当成已完成实验。每次版本、seed、失败重跑都使用新的 id，并为产物使用独立目录，避免旧证据被下一次运行覆盖。

```powershell
# 获取 split 文件的真实 SHA-256，填入 split_sha256
(Get-FileHash -LiteralPath 'outputs/your-split.json' -Algorithm SHA256).Hash.ToLower()

# 实际入口需生成下面约定的结果格式；包装器不会擅自运行正式实验
python -X utf8 tools/progress.py run --file progress/my-run.json -- python -X utf8 your_experiment.py --config config/my-run.json
```

包装器用 `shell=False` 执行 `--` 后的参数数组。执行前保存源码指纹、config 与 split 副本，执行后记录退出码和日志，并归档报告与预测副本。报告或预测路径执行前已经存在会被拒绝，防止误接收旧结果。退出码为零且产物、样本量、指标通过检查才记录 `completed`，否则保存 `failed`。运行中源码或配置变化也会使本次记录失败。完成结果默认 `unreviewed`，不会立即进入已核验性能曲线。

预测文件规范：JSON 文件须为对象数组，JSONL 须每个非空行一个对象；扩展名不明确时可在清单中设置 `prediction_format: json` 或 `jsonl`。每条至少包含唯一的 `sample_id`（也接受 `claim_id`、`id`；字符串或整数）和非空 `prediction`（标签字符串或实际预测对象）。条数必须恰等于 `n_samples`；暂缓样本也要留一条，明确写暂缓预测，不得丢弃。示例：`{"sample_id":"claim-001","prediction":"abstain"}`。空文件、缺行、重复 id 和空预测不能完成入库。当前研究入口若使用嵌套多方法输出，应提取对应方法的逐样本预测形成规范文件，并保留原始产物；工具不猜测其格式，也不自动证明预测 ID 与锁定 split 的科学含义相符。

`artifacts.report` 必须指向规范 JSON，例如：

```json
{
  "n_samples": 152,
  "metrics": {
    "macro_f1": 0.52,
    "accuracy": 0.57,
    "coverage": 0.68,
    "selective_risk": 0.23,
    "mean_cost_usd": 0.01
  }
}
```

以上数字只解释文件格式，不是本项目成绩；未写入账本或页面。现有研究入口的报告若不符合此格式，须根据真实评测产物另写规范汇总 JSON，不得从控制台猜测或补分。`n_samples` 必须是锁定评测集的样本数。暂缓如何计入 F1、风险分母、成本单位等必须写入 `protocol_id` 对应的协议文档和配置中。

支持的比例指标为 `macro_f1`、`accuracy`、`coverage`、`risk`、`selective_risk`、`forced_macro_f1`、`evidence_score`、`averitec_score`，一律 0—1；成本 `cost_rmb`、`mean_cost_rmb` 单位人民币，`cost_usd`、`mean_cost_usd` 单位美元，前端不可混用单位；时延 `latency_seconds` 单位秒。工具拒绝未知指标、负值、NaN、Infinity，以及比例超过 1。未来增加指标需同时定义单位、有效范围和图表解释。

## 复核后与复现基线对比

实验清单新增 `experiment_stage`：`formal` 为正式实验（省略时默认此值，兼容旧记录），`pilot` 为先导，`engineering` 为工程核验。阶段写入比较指纹，先导与正式结果即使使用相同数据/模型/预算也分组展示。工程记录不进入性能曲线。已完成但待核验的真实先导指标可在档案与详情中查看，并明确标记“待核验”；不会进入已核验曲线。

复核事件新增 `reviewer_kind`（命令行 `--reviewer-kind human|agent`，默认 human）。正式实验只有具名 human 复核才有资格进入正式曲线；agent 可核对其运行材料，但不能因此成为正式有效实验。先导允许 agent 核对真实逐样本预测、模型调用日志和指标重算后追加机器核验事件，页面标记“先导·机器核验”。这类小样本运行不替代双人来源/质量/充分性标注、人工事件分组审核或人工引文审计；这些科学人工监督材料仍然必须另行完成。核验命令自身仅检查文件与结构，不能代替实际核验动作。

```powershell
# 真实先导清单中设置 experiment_stage: pilot，数据集使用独立 id（例如 averimatec-pilot20）
python -X utf8 tools/progress.py review --id pilot-run-id --reviewer '执行核验代理名称' --reviewer-kind agent --note '已核对真实预测与模型调用日志，并从逐样本预测重算指标'
```

正式实验与先导分别计数，页面不会因机器核验的先导记录就显示正式实验已建立。新增直接 VLM 图文/纯文本对照可以按 `kind: baseline`、独立 `method_id` 动态登记，无需修改静态论文基线清单；不得因此改称 FIRE 或论文复现。

```powershell
python -X utf8 tools/progress.py review --id your-run-id --reviewer '复核者姓名' --note '已核对逐样本预测、完整评测集、泄漏隔离、指标口径与账单'
```

复核命令只验证记录结构及证据未变，不证明科学结论为真。`reviewed` 代表复核者已实际检查：预测确实来自所存模型/配置和代码、评测集完整无泄漏、标签未进入推理、指标能从逐样本预测重算、成本账单可追溯。不要仅因命令成功就声称实验可信。

复现基线使用同一流程，设 `kind` 为 `baseline`。消融设 `ablation`。`series_id` 用于同方法跨版本连线：完整方法固定为 `full_method`，本地对照与论文适配各自使用稳定标识；每次改进仅更新 `version` 与唯一 `id`。仅同时满足以下条件的记录可进入性能曲线：`completed`、对应阶段核验通过（正式需 human、先导可 agent）、全部证据存在且 SHA-256 未变、未撤回。基线与方法必须有相同 `comparison_key`；这个指纹包含实验阶段、数据集版本、split id/hash、协议、模型、预算及评测样本量，防止把不同设定的分数直接相减。模型或预算不同的外部基线应单独展示；不能冒充同条件对比。seed 单独记录，曲线中的每一点仍对应一次运行，不冒充多 seed 均值或置信区间。

协议、模型、预算的 id 是人工锁定的口径声明；相同 id 不自动证明两份不同配置在科学上相同。复核者仍须检查输入处理、标签集合、暂缓计分方式、预算单位、模型与量化版本等。基线结果通过本账本的结构与本地证据核验，不等同于论文方法已忠实复现；论文复现还需独立核对方法、参数、数据与原文协议，进度页中这些状态应分别表述。

## 已有规范结果手动入库

```powershell
python -X utf8 tools/progress.py fingerprint
python -X utf8 tools/progress.py record --file progress/my-completed-run.json
```

手动清单需补齐 `status: completed`、`review_status: unreviewed`、含时区的开始/结束时间、`code_fingerprint`、指标与四项证据路径；报告中的 metrics 和 n_samples 必须完全一致。不能在清单里自行声明 reviewed，必须另用具名 review 事件复核。代码指纹应为真正执行前已保存的指纹；现在得到的指纹不能证明历史运行的源码。手动声明默认标记 `code_fingerprint_origin: declared`。没有可追溯历史证据的结果应保留为诊断或历史审计，不能补写成正式完成。

```powershell
python -X utf8 tools/progress.py withdraw --id your-run-id --reason '发现评测事件与训练事件重叠，成绩无效'
```

撤回仅追加事件，不删除原始记录。每次生成页面重新检查证据 SHA；缺文件或文件改变的记录自动退出曲线，原始记录仍可查。不要直接编辑 `progress/runs`、`events` 或 `attempts/start.json` 来改变历史。

## 本轮边界

这个入口让后续每次实验可留档和复查；它不会自动启动 GPU、下载模型、申请标注、调用付费 API 或补齐尚未复现的基线。正式实验准备与研究方法问题仍按进度页阻塞清单处理。

## 先导分阶段展示验证

2026-11-17 新增阶段与核验类型后，账本34项测试通过；独立浏览器内存夹具16项通过，覆盖机器先导不增加正式数、跨阶段隔离、图文/纯文本对照动态追加、待核验先导原始指标展示、移动端和无脚本错误。最新完整套件为179通过、1跳过（`outputs/iterations/repair-20261117-01/full_tests_with_stages.xml`）。这些数字是工程验证，不是数据集模型表现；浏览器夹具没有写入真实账本。后续真实先导结果只从真实产物另行登记。


## 2026-11-17 看板初版验证（此前阶段）

本次既有研究约束测试为 51 通过、1 跳过；新增账本测试 23 通过，合计 74 通过、1 跳过。浏览器 20 项交互检查通过，覆盖档案/模板/CSV 导出、筛选、对话框、375/768/1024/1440px 响应式及内存夹具中的同组曲线/基线差值。独立对抗复查发现的四项账本缺陷已修复，修复后 5 项复验通过。具体证据位于 outputs/progress/verification。

测试夹具不会写入正式账本；交付页面仍显示 0 次有效正式实验和 0 项论文复现。页面视觉预览在 output/playwright/progress-desktop.png 和 progress-mobile.png。双击根目录 `更新并打开进度.cmd` 会重建并打开离线页。


## 修复与真实运行阶段（2026-11-17）

九项遗留问题已关闭；当前代码验收43/67。完整Qwen7B NF4在4060 8GB上全GPU实际跑通，单图冒烟峰值分配5.78GiB/保留5.90GiB。训练先导的图文与纯文本各20条已完成，四类Macro-F1均0.100、准确率25%，全部输出证据不足；34项独立产物核验通过。官方152条没有运行，不能用先导结果宣称完整方法有效。

FIRE原生二分类控制器另有真实工程v1失败、v2完成。其True不等于图文声明已证实；无原图输入、事后标签不一致与相似度分支未覆盖均保留在报告中。33项独立工程轨迹核验通过；完整论文复现仍未完成。

最终GPU环境套件194通过、1跳过。源码、日志、配置、源数据版本、预测与复核事件均留档；本轮汇总入口为 outputs/iterations/repair-20261117-01/ITERATION.json。
