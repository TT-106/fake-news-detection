# 基线源码、协议与接入准备

核查日期：2026-11-17。本次只读检查官方源码、包元数据与论文，不导入会下载模型或询问密钥的第三方入口，不调用收费 API，不运行 GPU 基线。

**当前结论：FIRE 与 DEFAME 官方源码已取得并锁定提交；FIRE 外置注入层通过11项控制流测试，并已跑通1条真实Qwen—搜索—原生二分类工程闭环，但没有正式基准成绩。SAFE-MR 论文来源已验证，尚未定位到经作者或论文链接确认的官方实现。三者均不能登记为“论文已复现”。**

## 证据与当前状态

| 方法 | 官方来源与本地锁定 | 当前可确认内容 | 尚未完成 |
| --- | --- | --- | --- |
| FIRE | [作者仓库](https://github.com/mbzuai-nlp/fire)、[论文](https://aclanthology.org/2025.findings-naacl.158/)；`third_party/fire` 的 `main`，提交 `cd7cb4af8dab00b1ab4bd78197a52e052203873d` | 原生文本原子声明、True/False 输出；1条真实模型/取证工程闭环 | 完整图文/四分类适配、科学多样本评测、证据充分性及费用公平验收 |
| DEFAME | [作者 ICML 分支](https://github.com/multimodal-ai-lab/DEFAME/tree/icml)、[ICML 论文](https://proceedings.mlr.press/v267/braun25b.html)；`third_party/defame-icml` 的 `icml`，提交 `9c6664d5a856ae3eefc601df8b04892ba07f4786` | 保留论文对应分支，含计划、工具执行、证据发展与判定流程 | 可用依赖、Qwen 模型适配、AVerImaTeC 数据适配、同预算工具边界和真实端到端运行 |
| SAFE-MR | [申请书[22]对应 arXiv v1](https://arxiv.org/abs/2610.06708v1)、[作者论文正文](https://arxiv.org/html/2610.06708v1) | 文献确实存在；包含训练的证据关系/充分性组件 | 经确认的官方代码、权重与标注；不能凭论文分数或自有 LR 替代实现宣称复现 |

两份本地仓库在检查时均无工作区修改。第三方原文件保持原样；适配必须放独立层，记录相对官方提交的差异。

公开检出按 [README](../README.md#快速上手) 恢复 FIRE。其适配器校验的是原 Windows 检出时的 CRLF 字节。需要查看 DEFAME 的论文版本时可执行：

```powershell
git clone --branch icml --no-checkout https://github.com/multimodal-ai-lab/DEFAME.git third_party/defame-icml
git -C third_party/defame-icml checkout --detach 9c6664d5a856ae3eefc601df8b04892ba07f4786
```

## FIRE 的真实运行入口与依赖

README 的 quickstart 写了 `python run_fire.py ...`，但锁定提交的文件树没有该文件。实际核心入口是 [`eval/fire/verify_atomic_claim.py::verify_atomic_claim`](https://github.com/mbzuai-nlp/fire/blob/cd7cb4af8dab00b1ab4bd78197a52e052203873d/eval/fire/verify_atomic_claim.py#L213)。其 `__main__` 只执行 `pass`，不能把 `python -m eval.fire.verify_atomic_claim` 退出成功当作跑完实验。官方批处理范例位于 `evaluator.ipynb`，其中示例还截取 `readlines()[:1]`，并非完整 benchmark 运行器。[源码树](https://github.com/mbzuai-nlp/fire/tree/cd7cb4af8dab00b1ab4bd78197a52e052203873d)、[范例](https://github.com/mbzuai-nlp/fire/blob/cd7cb4af8dab00b1ab4bd78197a52e052203873d/evaluator.ipynb)。

核心调用形态为：

```python
result, searches, usage = verify_atomic_claim(claim_text, rater)
# rater.generate(prompt) -> (response_text, usage_or_none)
# result.answer 原生只有 'True' / 'False'
```

核查的运行依赖与副作用：

- [`requirements.txt`](https://github.com/mbzuai-nlp/fire/blob/cd7cb4af8dab00b1ab4bd78197a52e052203873d/requirements.txt) 固定 `torch==2.4.0`、`transformers==4.44.2`、`sentence_transformers==3.1.1` 与旧版 LangChain。当前全局 Python 的 10/10 声明依赖均不满足原始版本要求；“不满足”包括缺包与版本不同，不等于全部无法安装。见本次 `baseline_dependency_snapshot.json`。
- 导入核心文件就构造 `SentenceTransformer('all-MiniLM-L6-v2')`；本地缺该模型时可能下载。重复查询检测还显式把 embedding 转到 CUDA，不是纯 CPU 安全入口。[核对位置](https://github.com/mbzuai-nlp/fire/blob/cd7cb4af8dab00b1ab4bd78197a52e052203873d/eval/fire/verify_atomic_claim.py#L14)。
- 原生检索经 `SerperAPI`，当前不得执行该收费路径；已有本地 NLI 权重不能冒充官方的句向量模型。[检索实现](https://github.com/mbzuai-nlp/fire/blob/cd7cb4af8dab00b1ab4bd78197a52e052203873d/eval/fire/query_serper.py)。
- 官方 HF 路径使用 `text-generation`，未实现 Qwen2.5-VL 图像消息适配；原生控制器只有文本输入。[模型实现](https://github.com/mbzuai-nlp/fire/blob/cd7cb4af8dab00b1ab4bd78197a52e052203873d/common/modeling.py#L88)。

原项目未经适配的默认API路径尚未执行；已执行的是下节明确披露依赖替换的一条真实工程冒烟，不等于原论文复现实验。

### 已实现的 FIRE 外置注入层

`src/baseline_reproduction.py` 提供 `OfficialFIREController`，运行名称固定为 **FIRE original-controller local-rater adapted**。它从上述锁定源码读入原提示词、数据类和三个原始控制函数；逐函数核对 AST 完全一致，并核验三份上游文件 SHA-256 与提交。不导入上游的模型/API初始化，不伪造第三方模块或改写 `sys.modules`。

移除无关导入、全局 SBERT 初始化及替换依赖边界的可审阅差异保存在 `docs/baselines/fire_dependency_injection.patch`；运行时重新生成差异并逐字核对，patch不一致就拒绝加载。`call_search` 和 `get_sentence_similarity` 由具名回调提供，原始 `strip_string` 与 JSON 解析函数直接取自锁定上游源码。**模型、搜索和相似度提供者均须记录身份；没有免费的默认“替代相似度”来暗改算法。**

```python
from src.baseline_reproduction import OfficialFIREController, ArchivedFIRESearch

# 以下对象均须由真实运行准备，不是本文件内已运行的模型或网络服务。
search = ArchivedFIRESearch("outputs/locked-real-searches.json")
controller = OfficialFIREController(
    search, actual_sbert_similarity_count,
    rater_id="实际模型与量化指纹",
    search_provider_id=search.manifest_sha256,
    similarity_provider_id="实际句向量模型与阈值协议指纹",
)
record = controller.verify(original_claim_text, local_qwen_rater)
```

`local_qwen_rater.generate(prompt)` 返回文本及实际 token usage，或明确 `None`；回放搜索按精确查询查找已归档片段并验证 SHA-256、URL、正文位置。缺查询、快照变化或模型未输出合法二分类都会记录失败，不填造材料。回放层没有验证原始账单，因此该记录本身不证明费用公平；费用与外部预算控制须在正式运行前另行锁定。

`tests/test_baseline_reproduction.py` 的11项测试实际执行了这三个上游控制函数，覆盖直接定论、检索后定论、重复查询早停、步数上限强制二分类、非法标签、禁止隐式模型依赖导入、源码/patch变更拒绝、归档检索失败及完整失败调用记录。模型/搜索/相似度是明确的合成测试依赖；**没有真实基线分数**。上游原始 usage 可能漏计格式失败尝试，适配层保留原计数并另报全调用 `rater_recorded_usage`，不把缺失 usage 填成真实零费用。原始测试日志为 `outputs/iterations/repair-20261117-01/baseline_adapter_tests.log`，最新完整套件见同目录 `full_tests_with_stages.log`。

### 免费本地依赖已准备

独立 `.venv-research` 只以 `--no-deps` 新增 `sentence-transformers==3.1.1`，保持原有 `torch==2.14.1+cu130`、`transformers==4.57.6`。从[官方 all-MiniLM-L6-v2 仓库](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)下载必要的 safetensors、tokenizer 与配置共10个文件；未下载 ONNX/OpenVINO。模型仓库 revision 为 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，下载时对照官方 LFS SHA-256 或 Git blob 标识核验，本地逐文件 SHA-256 记录于 `models/all-MiniLM-L6-v2.manifest.json`。

`src/fire_local_dependencies.py` 的 `LocalSbertSimilarity` 强制只读本地且 `trust_remote_code=False`，使用真实 encoder 与 `util.cos_sim`，在 CPU 按官方严格 `score > threshold` 统计。实际小测试已通过：相同句与无关句的列表在0.9阈值下计1，空历史计0；见 `fire_sbert_smoke.json` 与 `fire_sbert_pytest.log`。CPU设备和较新Torch/Transformers均为已声明适配差异。

`QwenFIRERater` 只接受外部已经加载的 `LocalQwenVLM`，不加载第二份Qwen，不更改原 `PROMPTS` 或基础 `model.identity`。它读取官方 `SYS_PROMPT`，保留原始FIRE提示文本，system/user两条消息的content均为多模态text列表，以 `do_sample=False` 生成；独立rater身份额外冻结系统提示、输入/输出上限与消息格式。逐次保存原始prompt、response、真实token数、耗时与失败记录。超输入上限不静默截断，达到输出上限不把截断JSON当有效结果。接口测试为6项通过，后续真实工程运行见下节。

```python
from src.fire_local_dependencies import LocalSbertSimilarity, QwenFIRERater

similarity = LocalSbertSimilarity()
rater = QwenFIRERater(loaded_vlm, "outputs/your-run/fire-rater-calls")
controller = OfficialFIREController(
    real_archived_search, similarity,
    rater_id=rater.identity,
    search_provider_id="实际查询归档版本",
    similarity_provider_id=similarity.identity,
)
# 完整真实运行由主流程另外显式执行，仍需真实查询归档与独立二分类评估协议。
```

### 一条真实工程闭环：不等同于真实性验证

`fire-native-smoke-v2` 已经 `progress run` 包装完成：预注册 `train:111` 仅原配文、2次真实Qwen调用、1次模型原样查询、2份真实网页reader正文片段，原生输出 `True`。工程耗时87.27秒包含初始化和取证桥等待；没有计算正确率/F1，也未进入科学曲线。详细材料在 `outputs/pilot/fire-native-smoke-v2/ENGINEERING_REPORT.md`。

**原生 True 仅是模型输出，不能当作声明已证实；FIRE 未读取原图，未适配完整图文/四类任务，检索正文存在不等于充分性通过。** 事后离线诊断发现输出与数据集 Refuted 标注不一致：数据集解释指向西班牙球队人群，而本次查询只取得Dhoni普通公众活动材料。标注解释未作为模型输入，也不被本报告冒充独立外部事实核验；该单例只揭示缺失图像与具体语境的边界，不产生paper准确率。

v1因Windows CRLF/LF片段偏移校验失败而原样留档；v2采用独立ID和字节校验后的LF快照重跑。两版driver相同，v1driver/报告及v2桥构建器哈希均与原账本一致，核对结果在 `fire_real_smoke_verification.json`。

### 独立审查后的阈值边界小修复

独立审查发现本地初版误用 `>=`，官方 helper 实际严格使用 `>`。新版本 `fire-local-similarity-strict-gt-v1` 已改为严格大于，且比较语义进入新的相似度身份。真实CPU encoder 的精准测试取本次实际cosine `0.9247545003890991` 作为阈值：修复前计1（RED），修复后计0（GREEN）。记录及日志在 `outputs/iterations/repair-20261117-01/fire_similarity_boundary_fix.json`。

旧FIRE v2的源码快照仍保留原 `>=` 版本，其 `similarity_trace` 为空，未执行该边界分支，因此工程闭环记录保留，不能冒称它验证过新严格比较。未重跑Qwen、未更改旧driver、报告、模型调用或桥产物。

## DEFAME 的真实运行入口与依赖

原作者 README 明确区分 ICML 复现分支与后续 `main`。本项目锁定 `icml`；其论文实验配置保存在 `config/icml_experiments/`。[分支说明](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/README.md)。

公开入口是 `scripts/run.py` 的 `FactChecker(...).verify_claim(...)`，以及 `scripts/run_config.py` 的 `run_experiment(config_file_path=...)`。后者当前硬编码 ClaimReview 的 Llama 配置，不是可直接传命令行参数的通用 runner；`scripts/run_batch.py` 对应批量执行会删除读过的配置，不应对唯一的复现清单目录直接操作。[最小运行示例](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/scripts/run.py)、[配置入口](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/scripts/run_config.py)、[批量行为](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/defame/eval/utils.py)。

```python
fact_checker = FactChecker(llm=local_model_adapter, tools=archived_tools,
                           classes=locked_classes,
                           class_definitions=locked_definitions)
report, stats = fact_checker.verify_claim(image_text_claim)
```

上面是**待实现的注入形态**，不是已运行示例。官方 [`FactChecker`](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/defame/fact_checker.py#L28) 接受模型对象和自定义工具，可保留其规划/发展/判定控制流。其 [`Model`](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/defame/common/modeling.py#L149) 还需实现加载、任意 Prompt 多模态生成、token 统计、重试与统计重置；现有本项目固定操作提示词的 `LocalQwenVLM._call` 不能原样替代该接口。

需要先处理的具体差异：

- `make_model` 的开源分支只分派 Llava/Llama，没有 Qwen2.5-VL 分派，HF 加载也未设置本地文件限制与本项目的 4bit 方案。[工厂实现](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/defame/common/modeling.py#L752)。
- 直接导入 `config/globals.py` 会在没有配置密钥时进入交互式索取，并可能写服务账户文件；本次没有导入或读取实际密钥。[配置初始化](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/config/globals.py#L62)。
- 官方 benchmark registry 无 **AVerImaTeC**。已有 **AVeriTeC** 被显式标成 `is_multimodal=False`，读取 `claim`、`claim_date` 等不同字段；不能仅重命名文件就当图文任务适配完成。[registry](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/defame/eval/__init__.py)、[AVeriTeC benchmark](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/defame/eval/averitec/benchmark.py#L12)。
- 论文默认配置可能需要 GPT、Google 搜索/反向图片搜索等外部服务；本地免费替换必须共享真实工具边界并记录改动。当前全局 Python 的 45 项声明依赖中 34 项不满足；该统计不是对主代理新建 GPU 虚拟环境的验证。[依赖](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/requirements.txt)。

## SAFE-MR 的来源核验边界

申请书 [22] 与 [arXiv:2610.06708v1](https://arxiv.org/abs/2610.06708v1) 的题名、作者 Shiwen Ni 对应。论文 [Appendix A.3](https://arxiv.org/html/2610.06708v1#A3) 描述冻结 CLIP 表征、训练的图模块与预测头，并用 Qwen2-VL 辅助声明提取；它不是简单静态路由或本项目充分性 LR。检查论文链接并以完整题名、作者和 SAFE-MR 检索后，未定位到经作者/论文确认的官方代码。这里只报告本次检索未确认，不能据此断言代码绝不存在。没有下载不明仓库、导入论文分数或自行补造实现。

## 同模型比较与原论文复现的登记规则

| 运行类别 | 可做的声称 | 与四类主实验的关系 |
| --- | --- | --- |
| 原提交 + 原模型/数据/工具/评估协议 | 达到复现验收后，才称原论文复现 | 通常不同任务，另表报告 |
| 官方控制器 + 本地同一 Qwen + 同一真实检索环境 | `official-code adapted`，逐项披露替换 | 任务与终端/预算锁定一致后，才作同口径对照 |
| 重写的 FIRE-style 或简化工具路由 | 自有启发式/机制对照 | 可比较，但不能标作 FIRE/DEFAME 论文复现 |
| Qwen 原图+配文直接判定 | 无外部检索的直接 VLM 基线 | 可先做真实先导，不能替代动态方法验收 |

FIRE 原生二分类没有 NEI/Conflicting，且最大步数后会强制给出 True/False。禁止把“NEI/冲突”硬塞成 False 后与四类 macro-F1 混用。可选择：

1. 保持官方二分类，在预先锁定的 Supported/Refuted 子协议上单独评价，保留不同任务/分母/指标，不能加入四类主曲线。
2. 保留其迭代检索控制，另接共享四类终端与暂缓门控；命名为 `FIRE-style Qwen four-class adaptation`，明确已改变原生输出与停止规则，不能称原论文复现。

DEFAME 有 NEI/冲突相关标签，但原生标签定义、内部 NEI 继续检索、最终错误/拒答与本项目系统暂缓并不自动等价。需要先写清映射和独立“暂缓”状态，不能把 NEI 当作系统拒答。[官方标签定义](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/defame/common/label.py)、[动态流程](https://github.com/multimodal-ai-lab/DEFAME/blob/9c6664d5a856ae3eefc601df8b04892ba07f4786/defame/procedure/variants/summary_based/dynamic.py)。

## 最小接入验收顺序

1. 保留官方仓库，记录提交与无修改状态；独立适配层记录补丁/包环境。不要把旧依赖一键覆盖到正在使用的项目环境。
2. 先接任意文本/图像 Prompt 的同一冻结 Qwen 与逐调用日志，做无收费 API 的模型接口检查；图片消息和输出解析缺失时停止。
3. 接真实归档工具，查询、返回、URL/快照、失败及计量全部可追溯；缺某查询返回就明确未完成，不能从 QA 或其他样本补足。
4. 单声明先导检查每个官方控制流确实被调用、标签映射有效、无隐藏下载/API 路径、预算统一；再运行20条相同先导声明。
5. 保存逐样本预测、请求与费用、源码/适配层/模型/划分哈希，复核后登记。模型适配测试通过不等于完整基线跑完。

本次审计机器可读记录：`outputs/iterations/repair-20261117-01/baseline_readiness.json`；依赖元数据快照：同目录 `baseline_dependency_snapshot.json`。
