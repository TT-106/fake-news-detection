# 环境和资源下载

命令均在仓库根目录运行。先按 [README](../README.md) 安装 CPU 测试环境。

## AVerImaTeC 训练数据与原图

```powershell
$py = '.\.venv\Scripts\python.exe'
uv pip install --python $py -r requirements-assets.txt
& $py -X utf8 tools/download_research_assets.py dataset
```

脚本使用[作者发布的固定版本](https://huggingface.co/datasets/Rui4416/AVerImaTeC/tree/7e77fb26a1ba86159ff5ba27b59f373b450d6fd0)，校验训练 JSON 和图片 ZIP 的 SHA-256，再逐图校验并解码。

| 内容 | 项目内路径 |
| --- | --- |
| 标签、参考问答及训练样本 | `data/averimatec/train.json` |
| 原图 ZIP 和下载清单 | `data/averimatec/upstream/images.zip`、`data/averimatec/upstream/UPSTREAM_LOCK.json` |
| 解包后的原图与逐图校验清单 | `data/averimatec/images/`、`data/averimatec/images/image_manifest.json` |

开发先导只使用 `train.json` 和指定原图。开展正式终评时，由负责人从同一固定版本下载 `val.json`，保存到 `data/averimatec/val.json`。其 SHA-256 为 `e9c61d6538acc8e8312cf3137ddf8e06efe2a8a2a5ab60583c56344ae8e81166`。下载的 `images.zip` 含上游训练和开发图像，先导代码按固定训练 ID 加载。作者另有独立的 [FEVER-9 测试发布和评测页](https://huggingface.co/spaces/FEVER-IT/AVerImaTeCTest)，其中的测试集和项目留作内部终评的开发集分开管理。

## Qwen2.5-VL-7B-Instruct

```powershell
& $py -X utf8 tools/download_research_assets.py model
```

源文件为 [Qwen 官方固定提交](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/cc594898137f460bfe9f0759e9844b3ce807cfb5)，落在 `models/Qwen2.5-VL-7B-Instruct/`，包含处理器、分词器、权重索引和五个权重分片，合计约 16.6 GB。完整下载后，`UPSTREAM_LOCK.json` 记录 `HASH_VERIFIED` 和各文件摘要。项目加载标准权重并在本地用 bitsandbytes 做 NF4 量化。

网络不稳定时可用分块续传。临时分块和最终权重会同时占一部分磁盘空间。每次续传沿用相同目录。

```powershell
$model = 'models/Qwen2.5-VL-7B-Instruct'
New-Item -ItemType Directory -Path $model -Force | Out-Null
if (-not (Test-Path "$model/UPSTREAM_LOCK.json")) {
    Copy-Item 'docs/assets/qwen-2.5-vl-7b.json' "$model/UPSTREAM_LOCK.json"
}
& $py -X utf8 tools/ranged_download.py --manifest "$model/UPSTREAM_LOCK.json" --destination $model
```

## 知识库与检索模型

双路检索和 Sprint 2 采集需要训练阶段的官方 Knowledge Store 和 MiniLM。下载与部署位置如下。

```powershell
& $py -X utf8 tools/download_knowledge_store.py --split train
$mini = Get-Content 'docs/assets/all-MiniLM-L6-v2.manifest.json' -Raw | ConvertFrom-Json
$miniFiles = @($mini.files.path)
$env:HF_HUB_DISABLE_XET = '1'
& .\.venv\Scripts\hf.exe download $mini.repository @miniFiles --revision $mini.revision --local-dir $mini.local_directory
if ($LASTEXITCODE -ne 0) { throw 'MiniLM 下载失败' }
foreach ($file in $mini.files) {
    $path = Join-Path $mini.local_directory $file.path
    if ((Get-FileHash -LiteralPath $path).Hash -ne $file.sha256) { throw "校验失败：$path" }
}
Copy-Item 'docs/assets/all-MiniLM-L6-v2.manifest.json' 'models/all-MiniLM-L6-v2.manifest.json'
```

知识库保存为 `data/averimatec/knowledge_store/train/text_related.zip`，约 6.72 GB，保持 ZIP 供程序读取。下载器会校验归档结构并生成 `manifest.json`。当前 Sprint 2 配置要求 ZIP 的 SHA-256 为 `6c4e8e2c92e3c352e61c0ae447da0da619e18c8e9c33f269ddfee7310eba6eb9`。终评知识库由负责人在评测阶段授权取用。来源分别是[官方共享任务](https://fever.ai/task.html)及 [MiniLM 固定提交](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)。

历史图文及 NLI 诊断还可从 [TinyCLIP](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M)和 [NLI MiniLM](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768)下载各自的 safetensors、配置以及处理器或分词器，分别放在 `models/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M/`、`models/nli-MiniLM2-L6-H768/`。新环境做历史对照时另行记录对应 revision。

## 单条 GPU 检查

原运行环境使用 Windows x64、Python 3.13、CUDA 13.0 版 PyTorch，NVIDIA 驱动须支持该 CUDA 版本。另建环境并按快照安装。项目原先用 `tools/setup_research_env.ps1` 从另一份已有 Windows 环境复制 PyTorch，新机器可直接从 wheel 索引安装。

```powershell
uv venv --python 3.13 .venv-research
$gpu = '.\.venv-research\Scripts\python.exe'
uv pip install --python $gpu --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match -r requirements-gpu.lock.txt
& $gpu -X utf8 -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
& $gpu -X utf8 tools/smoke_qwen.py --config config/model_smoke_qwen_v1.json
```

单条检查使用固定训练先导中的一条声明，在 `outputs/model_smoke/qwen-bnb4-v1/` 记录预测、真实调用和 SHA 校验结果。原工作区重新执行时，先复制配置并把 `output_dir` 改为新目录。进一步运行完整训练先导用 `python -X utf8 -m src.pilot_vlm --config config/pilot_qwen_direct_image_text_v1.json`，具体比较口径见 [训练池先导协议](PILOT_PROTOCOL.md)。
