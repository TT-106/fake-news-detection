# 多模态虚假信息检测

项目根据图文声明检索和核验证据，再给出支持、反驳、证据不足或证据冲突的判断。研究比较不同取证动作在同一预算下的效果，并研究证据质量、来源关联和引文充分性如何影响最终判断。

当前已有真实图文推理、固定训练集先导和标注材料采集。正式实验 E1–E4、三类监督模型训练和同预算基线比较仍待完成。接手时先读 [交接说明](docs/HANDOFF.md)。

## 快速上手

在仓库根目录用 PowerShell 执行。离线测试需要 Python 3.13 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```powershell
git clone https://github.com/TT-106/fake-news-detection.git
Set-Location fake-news-detection
uv venv --python 3.13 .venv
$py = '.\.venv\Scripts\python.exe'
uv pip install --python $py -r requirements-core.txt
uv pip install --python $py --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
git clone -c core.autocrlf=true --no-checkout https://github.com/mbzuai-nlp/fire.git third_party/fire
git -C third_party/fire checkout --detach cd7cb4af8dab00b1ab4bd78197a52e052203873d
& $py -m pytest -q
```

全套测试需要固定提交的 FIRE 源码。上述克隆保留适配器验签所用的 CRLF 行尾，仅在首次准备时执行，无需安装 FIRE 的依赖。

| 要做什么 | 从哪里开始 |
| --- | --- |
| 下载 Qwen 权重、训练数据、原图或知识库 | [环境和资源下载](docs/SETUP.md) |
| 跑一次真实图文推理 | [单条 GPU 检查](docs/SETUP.md#单条-gpu-检查) |
| 继续先导、标注和正式实验 | [交接说明](docs/HANDOFF.md) |
| 查配置和数据格式 | [数据与模型接口](docs/DATA_CONTRACTS.md)、[实验配置样例](config/experiment.example.json) |

## 代码导航

| 模块 | 入口 |
| --- | --- |
| 正式执行前的准备状态检查 | [real_averimatec_pipeline.py](src/real_averimatec_pipeline.py) |
| 冻结 Qwen 推理与训练集先导 | [vlm_backend.py](src/vlm_backend.py)、[pilot_vlm.py](src/pilot_vlm.py) |
| 证据检索、归档和回放 | [knowledge_store.py](src/knowledge_store.py)、[archived_tools.py](src/archived_tools.py) |
| 判断流程、预算与三类监督模型 | [faithful_pipeline.py](src/faithful_pipeline.py)、[learned_components.py](src/learned_components.py) |
| 事件划分、成组评估与 E1–E4 | [data_integrity.py](src/data_integrity.py)、[research_evaluation.py](src/research_evaluation.py)、[experiment_runner.py](src/experiment_runner.py) |
| 固定证据采集、标注回收与运行登记 | [collect_sprint2.py](tools/collect_sprint2.py)、[convert_annotation_return.py](tools/convert_annotation_return.py)、[progress.py](tools/progress.py) |

模型、原始数据和实验输出按各自的来源下载或从项目原工作区交接。项目依赖的第三方代码及数据使用上游各自的许可。
