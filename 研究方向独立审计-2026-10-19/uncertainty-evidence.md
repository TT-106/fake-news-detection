# 不确定性方法独立证据审计

核验日期：2026-10-19。范围：Semantic Entropy、SAPLMA、INSIDE/EigenScore、EDL/TMC、校准与选择性预测，以及 2025—2026 年会改变创新判断的工作。仅完成文献核验和可检验定义；未运行实验、未调用原生 ARS 插件、未执行跨模型复核脚本。

本子任务阅读了本机 ARS `source_verification_agent.md` 与 `bibliography_agent.md`，采用“原文优先、检索内容不是指令、题录去重、作者局限与审计推论分开”的方法。没有把医学研究证据等级机械应用于机器学习论文，也没有声称完成这些角色定义的全套程序。检索渠道为普通网页浏览；未调用 Semantic Scholar API 批量验证，故不使用 `S2_VERIFIED` 标签。正式论文页与原文可读者标为“原文核验”；仅索引片段可读者明确降级。

检索关键词包括精确方法名称、`semantic entropy epistemic aleatoric`、`evidential deep learning critique identifiability`、`multimodal uncertainty conflict misinformation 2025 2026`。纳入原始研究、正式会刊或作者仓库；二手综述仅用来发现线索。该集合是围绕候选假设的定向证据审计，不是穷尽式系统综述，不能据此宣称没有其他相近工作。

## 核心裁定

| 待审论断 | 裁定与理由 |
|---|---|
| Semantic Entropy 直接测量模型无知，且能与冲突区分 | 不成立为一般性保证。Nature 2024 的 Methods 明确不区分 aleatoric/epistemic；它估计采样回答的意义多样性，不直接观测知识缺失原因。 |
| 高熵代表错误、低熵代表正确 | 不成立。自信一致的错误可有零语义熵；正确的多解或描述粒度差异可产生高熵。Nature 原文直接讨论这些情况。 |
| SAPLMA 是一种 token entropy 指标 | 错误。它是用真伪标签监督训练的隐藏状态分类器。 |
| INSIDE 和 EigenScore 是两种独立前沿方法 | 名称重复。INSIDE 是框架，EigenScore 是该论文提出的核心指标。 |
| EDL/TMC 已证明 Dirichlet 强度等于真实认知无知 | 不成立。该表示有操作定义；可辨识性、收敛与真实无知解释受正式理论研究质疑。可作为特征或基线，不可当无知真值。 |
| 不确定性加跨模态冲突再动态门控具有新颖性 | 证据明显不足。TMC、UDRL、AEMRL 已覆盖多个组成部分；可能的差异必须落在具体核查任务、可控机制验证及成本收益上。 |
| 每个相关方法都适合作为同任务竞争基线 | 错误。多数是文本 QA 的模型输出错误检测，或一般多视图分类；迁移前必须明确输入、标签、访问权限和算力条件。 |

## 逐篇证据记录

### U01 Semantic Uncertainty：原始语义熵论文

**Lorenz Kuhn, Yarin Gal, Sebastian Farquhar. (2023). _Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation_. ICLR 2023.** [原文 PDF](https://arxiv.org/pdf/2302.09664)；[arXiv](https://arxiv.org/abs/2302.09664)；[作者代码](https://github.com/lorenzkuhn/semantic_uncertainty)。未添加未经核验的会刊 DOI。

- 任务/数据/模态：文本 QA 回答错误检测，CoQA、TriviaQA；OPT 系列模型，不是新闻真假分类。
- 方法：同一问题多次采样，用双向蕴含聚类语义等价回答，再估计意义簇上的熵；评价错误识别 AUROC。真正测量的是生成分布的语义分散程度。
- 证据：原文 §6 在两类 QA 条件下优于所比较的序列熵、词面相似度等；不是所有任务或所有模型的普遍定理。
- 开放性：代码与实验数据入口公开，原仓库现已注明 deprecated；后续版本见 U02。
- 限制来源：原文实验设定仅验证 QA；将其当作新闻事实真值或跨模态冲突度量属于审计方不支持的外推。与本研究关联为不确定性工具，非同任务直接方法。

### U02 Nature 2024 Semantic Entropy：不可与 U01 当成无关发明

**Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal. (2024). _Detecting hallucinations in large language models using semantic entropy_. Nature, 630, 625–630.** [正式全文与 DOI 10.1038/s41586-024-07421-0](https://www.nature.com/articles/s41586-024-07421-0)；[作者代码](https://github.com/jlko/semantic_uncertainty)。

- 数据/架构：TriviaQA、SQuAD 1.1、BioASQ、NQ-Open、SVAMP，及 FactualBio；LLaMA 2 Chat、Falcon Instruct、Mistral Instruct；另以 GPT-4 做传记实验。文本输入，多次生成加蕴含聚类；离散变体可不依赖 logits。
- 指标/对象：意义簇熵；错误识别 AUROC 与拒答后的 AURAC。针对 confabulation，即随机种子变化下任意且错误的回答，不涵盖全部事实错误。
- 原文限制：Main/Discussion 明确无法保证 factuality，系统性自信错误不适用；Table 1 说明日期粒度使正确答案被过细聚类；Methods 明确没有区分 aleatoric 与 epistemic；长度归一化缺乏充分理论依据。
- 开放性/适用性：代码与 FactualBio 资料有公开入口；部分复现使用收费闭源评估器，BioASQ 需单独获取。与本课题是可迁移的错误风险信号，非外部新闻证据核查保证。

### U03 SAPLMA

**Amos Azaria, Tom Mitchell. (2023). _The Internal State of an LLM Knows When It’s Lying_. Findings of EMNLP 2023, 967–976.** [正式页](https://aclanthology.org/2023.findings-emnlp.68/)；[全文](https://aclanthology.org/2023.findings-emnlp.68.pdf)；[arXiv](https://arxiv.org/abs/2304.13734)。

- 任务/数据：英文事实陈述真伪分类，自建 6,084 条 true-false statements，六个主题；每次留一主题测试，含模型生成陈述测试；并非依赖外部证据的新闻核查。
- 模型/指标：冻结 OPT-6.7B、Llama2-7B，取若干层隐藏激活训练三隐藏层 MLP，以真伪监督；主要报告 accuracy。测量的是训练探针估计的标签正确性，不是无监督熵，也不意味着模型有主观“撒谎意识”。
- 结论条件：主题留出下可解码事实性信息，优于论文中的提示与概率基线。
- 作者限制（§8）：仅英文、阈值需校准；单句激活不等于长段落混合真伪事实的局部标签。数据公开链接在 §3；本次未核到作者独立完整代码仓库，不能写“代码已验证可复现”。
- 直接关联：隐藏状态轻量探针基线；若移植至新闻，必须用本任务训练/校准并做事件留出。

### U04 INSIDE / EigenScore（同一论文）

**Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, Jieping Ye. (2024). _INSIDE: LLMs’ Internal States Retain the Power of Hallucination Detection_. ICLR 2024.** [原文 HTML](https://arxiv.org/html/2402.03744v2)；[arXiv](https://arxiv.org/abs/2402.03744)；[作者代码，旧 alibaba 链接现转向 D2I-ai](https://github.com/D2I-ai/eigenscore)。

- 数据/模型：CoQA、SQuAD、NQ、TriviaQA，另附 TruthfulQA；LLaMA-7B/13B、OPT-6.7B 等冻结文本模型。主设定十次生成，中间层最后输出 token 表征。
- 方法/测量：以生成表征矩阵构造 K×K 正则化协方差，`EigenScore=logdet(Σ+αI)/K`，评估表征自一致性；另用 feature clipping 缓解过度一致。它不是单次隐藏状态能量，也不等于语义熵原方法。
- 评价/条件：AUROC、与答案相似度相关性；主标签用 ROUGE-L/embedding 相似度阈值，附录补 exact match。对照解释依赖这些正确性代理。
- 作者限制（Appendix G）：需访问内部状态，不适用于无隐藏状态的黑盒；多次采样有成本；未完成幻觉缓解。Appendix H 展示所有采样重复同一错误仍漏检的反例。
- 开放性/关联：MIT 代码与数据入口公开但未运行；是文本 QA 信号，可作迁移基线，不能直接证明视觉接地。

### U05 EDL 原始分类论文

**Murat Sensoy, Lance Kaplan, Melih Kandemir. (2018). _Evidential Deep Learning to Quantify Classification Uncertainty_. NeurIPS 2018.** [正式页](https://proceedings.neurips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html)；[全文](https://proceedings.neurips.cc/paper_files/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf)；[作者实现页面](https://muratsensoy.github.io/uncertainty.html)。

- 输入/数据/模型：图像分类，LeNet，MNIST、CIFAR 子集；含 notMNIST OOD 与对抗扰动验证。
- 定义：网络非负 evidence `e_k` 给出 `α_k=e_k+1`，Dirichlet 总强度 `S=Σα_k`，类别均值 `α_k/S`，vacuity `u=K/S`。真正测量的是该训练目标产生的意见强度与均值；不能仅从公式推出真实世界知识量。
- 结论：论文所测图像分类/OOD/扰动条件下提供单次前向不确定性机制；不是对所有 OOD 或新闻有效的保证。
- 开放性：作者页面给实现与演示；本次未执行。关联为便宜的表示选择及对照，非模型无知真值。
- 局限来源：U07/U08 正式理论工作限制其不确定性解释；这不是原作者承认“方法完全无效”，也不否定某些下游有效性。

### U06 TMC 与期刊扩展：年份须区分

**Zongbo Han, Changqing Zhang, Huazhu Fu, Joey Tianyi Zhou. (2021). _Trusted Multi-View Classification_. ICLR 2021.** [全文](https://arxiv.org/pdf/2102.02051)；[作者代码](https://github.com/Han-Zongbo/TMC)。

**同四位作者. (2023). _Trusted Multi-View Classification With Dynamic Evidential Fusion_. IEEE TPAMI, 45(2), 2551–2566.** [PubMed 正式题录，DOI 10.1109/TPAMI.2022.3171983](https://pubmed.ncbi.nlm.nih.gov/35503823/)；[作者预印本](https://arxiv.org/abs/2204.11423)。DOI/预印本中 2022 不等于卷期年份；不得标成“ICLR 2022 TMC”。

- ICLR 原文：一般多视图分类，Handwritten、CUB、PIE、Caltech101、Scene15、HMDB；各视图 evidence heads 加 Dempster–Shafer 融合，根据 Dirichlet 意见强度动态整合。
- 指标/对象：分类 accuracy；噪声视图鲁棒性与不确定性可视化。实验把高斯噪声样本当 OOD，并非语义新事件。
- 结论/局限：提供已有的不确定性驱动多视图融合机制；“处理视图噪声有效”不等于“已区分文本图像时空冲突与无知”。后一句为审计推论，依据任务/干预不相同。
- 开放性：作者代码仓库可读；未运行。期刊扩展本次核验题录/预印本入口，未将其每项新增实验视为核过。

### U07 EDL 可辨识性批评

**Mira Juergens, Nis Meinert, Viktor Bengs, Eyke Hüllermeier, Willem Waegeman. (2024). _Is Epistemic Uncertainty Faithfully Represented by Evidential Deep Learning Methods?_ ICML 2024, PMLR 235, 22624–22642.** [正式页与全文入口](https://proceedings.mlr.press/v235/juergens24a.html)；[arXiv](https://arxiv.org/abs/2402.09056)。

- 任务/对象：二阶损失最小化的分类、回归、计数理论分析；并非新闻基准。分析可辨识性、收敛及 epistemic 指标的相对性。
- 结论条件：说明给概率分布再套分布，并不自动从一阶观察中可辨识真实知识不确定性；不能以一项 OOD 成绩代替该解释的证明。
- 核验范围：正式题录及作者摘要可读；正式 PDF 的 raw GitHub 下载被浏览器当作 octet-stream，故本条理论细节未逐定理复核。不补写定理号或具体数值。
- 代码/数据：正式页有 software 入口；本次未核 repo 内容。与 A 的关联是反证与概念边界，不是新闻分类基线。

### U08 EDL Mirage：下游有效不等于不确定性解释成立

**Maohao Shen, J. Jon Ryu, Soumya Ghosh, Yuheng Bu, Prasanna Sattigeri, Subhro Das, Gregory W. Wornell. (2024). _Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?_ NeurIPS 2024.** [正式页与 DOI 10.52202/079017-3424](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3177be226ee12e34d6ba3b5e6fe6a5b-Abstract-Conference.html)。

- 任务/机制：统一一类 EDL 损失的渐近行为，并从 energy-based OOD 角度重新解释下游有效性；研究对象是 EDL 的 UQ 解释。
- 作者结论：即便部分下游任务有效，也可能不具有所声称的不确定性量化能力；纳入模型不确定性可改善但增加计算。
- 核验边界：正式页全文链接读取失败，仅题录、摘要、DOI 已核；不填写本次未读的具体数据集和性能数字。代码发布状况本次未核。
- 与课题关联：支持“EDL 浓度不可未经验证就叫无知”的审计规则；不支持“所有 EDL 模型都无用”。

### U09 大规模解耦基准

**Bálint Mucsányi, Michael Kirchhof, Seong Joon Oh. (2024). _Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for Specialized Tasks_. NeurIPS 2024 Datasets and Benchmarks.** [正式页，DOI 10.52202/079017-1614](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5afa9cb1e917b898ad418216dc726fbd-Abstract.html)；[全文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5afa9cb1e917b898ad418216dc726fbd-Paper-Datasets_and_Benchmarks_Track.pdf)；[代码](https://github.com/bmucsanyi/untangle)。

- 任务/数据：ImageNet-1k、CIFAR-10 上的拒判、OOD、aleatoric 等专门任务；重实现 Bayesian/evidential/deterministic 估计方法。
- 对象/结论：检验不同估计器是否真的选择性响应不同来源；在其测试集合中，声称解耦的估计对仍高度相关，未实现期望解耦。
- 适用边界：这是所测图像任务的实证结论，不是任何未来方法不可能解耦的定理，也不直接测试图文冲突。
- 开放性：重实现、运行日志公开。直接关联 A 的实验方法：必须控制来源并检测交叉敏感性，不能凭两条公式命名完成机制认定。

### U10 校准基础

**Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. (2017). _On Calibration of Modern Neural Networks_. ICML 2017, PMLR 70, 1321–1330.** [正式全文](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)。

- 输入/任务：图像与文档的封闭标签分类；图像 CNN 与文档分类模型。研究概率置信度与正确率是否匹配；ECE、NLL、reliability diagrams，提出 temperature scaling。
- 结论：所测多数数据上单参数温度校准有效；温度不改变 argmax 类别，但会改变熵与分布距离。
- 作者条件（§4）：训练、校准、测试默认同分布；这不是事件/时间 OOD 下自动校准的保证。
- 开放性：论文公开；作者公开实现可用于方法复现，但本子任务未核代码链接或执行。关联是 A/B/C 共同必须补齐的校准基线；不解决标签可辨识性或因果归因。

### U11 选择性预测基础

**Yonatan Geifman, Ran El-Yaniv. (2019). _SelectiveNet: A Deep Neural Network with an Integrated Reject Option_. ICML 2019, PMLR 97, 2151–2159.** [正式页](https://proceedings.mlr.press/v97/geifman19a.html)；[全文](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf)。

- 任务：分类/回归时允许拒判；图像分类和回归数据。通过预测、选择、辅助三头联合优化指定覆盖率下风险。
- 真正测量：在接受预测子集上的损失与覆盖率权衡，非真假检测或无知来源。
- 结论条件：所测任务中联合优化优于简单后处理拒判；后处理 MSP 阈值仍是重要廉价对照。
- 限制（审计推论）：拒答减少不可靠输出，与额外检索纠错是不同决策，不能把“检索率”称为“预测覆盖率”。
- 开放性：正式全文公开；本次未核作者代码仓库。关联是评价工具，应画风险–覆盖率，不必复制端到端训练。

### U12 Semantic Entropy Probes

**Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal. (2024). _Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs_.** [arXiv 2406.15927](https://arxiv.org/abs/2406.15927)；[全文](https://arxiv.org/html/2406.15927v1)；[作者代码](https://github.com/OATML/semantic-entropy-probes)。本次看到的[Workshop 版本原文](https://openreview.net/pdf/22a7d04c051215b6f9bc0ff577916053349661ff.pdf)标注 ICML 2024 Workshop on Foundation Models in the Wild；未核到正式 ICLR 2025 接收记录，不应把投稿版本直接写成 ICLR 2025 正式论文。

- 数据/模型：TriviaQA、SQuAD、BioASQ、NQ Open；Llama 2、Mistral、Phi-3 Mini，长回答另含 Llama 3。文本 QA。
- 指标/对象：用多次采样得到的高/低语义熵监督线性 logistic probe；测试时单个隐藏状态预测“高语义熵”的概率。不是直接预测真伪标签，也不是零训练成本。
- 原文结论/限制：比准确性监督探针更能跨任务推广，但弱于昂贵的多次采样方法；需内部状态，训练期仍生成多次样本。作者公开仓库可读，未运行。
- 关联：在“按需检索要省钱”的论证中必须考虑的低成本替代，不能只与昂贵十采样熵相比。

### U13 UDRL：直接虚假新闻相关，全文核验不足

**Zihao Li, Jiaxin Yang, Xianghan Wang, Jun Lei, Shuohao Li, Jun Zhang. (2025). _Uncertainty-aware disentangled representation learning for multimodal fake news detection_. Information Processing & Management, 62(5), 104190.** [正式页/DOI 10.1016/j.ipm.2025.104190](https://www.sciencedirect.com/science/article/pii/S0306457325001311)。

- 已核内容：出版社搜索索引正文片段显示图像 Swin Transformer、文本 BERT；Gaussian probabilistic representation、shared/private 解耦与 uncertainty-aware fusion；摘要称三个基准。
- 未核内容：本次正式页直开 403，故具体数据集、划分、概率量定义、代码、数据许可与性能数值一律待全文。作者名由出版社题录检索与原论文书目交叉发现；不依赖 ResearchGate 的异常作者映射。
- 新颖性冲击：不能宣称“首次将不确定性与模态私有/共享信息融合用于假新闻”。但仅这些可读证据不能认定它已实现锚点级时空核查或按需外部检索。
- 直接关联度高，复现资格待代码及全文核查；目前只能纳入相关工作边界，不能冒充已跑通竞争基线。

### U14 AEMRL：2026 年一般多模态冲突/不确定性工作

**Yupeng Han, Kai Zhang, Xianquan Wang, Zhihong Pan, Ze Liu, Jiyuan He. (2026). _Trust, but Verify: Uncertainty-Driven Evidential Multimodal Representation Learning_. IJCAI 2026, 1143–1151.** [正式页及 DOI 10.24963/ijcai.2026/128](https://www.ijcai.org/proceedings/2026/128)；[正式全文](https://www.ijcai.org/proceedings/2026/0128.pdf)。

- 数据/架构：MIMIC-CXR、Drive&Act、VGGSound、Hateful Memes、MM-IMDb；模态特定编码器和 Dirichlet evidence heads，超网络控制分解，冲突感知融合。原文训练配置四张 A100、端到端；不适合作为本科资源下的首个全量复现目标。
- 指标/对象：各任务主指标加 ECE；以各模态预测的对称 KL 测冲突，再缩放 evidence；`C/Σα` 操作化其单模态 uncertainty。
- 结论范围：原文报告噪声、语义冲突、缺模态鲁棒性；这些不是新闻时空核查实验。将所有 learned strength 解释为真实 epistemic 仍需面对 U07–U09。
- 开放性：正式 PDF 公开，正文没有代码仓库链接，本次未核到官方代码。与 A/C 的一般机制直接相关，与新闻核查任务间接相关。

### U15 2026 年 EDL 指标新标准

**Linye Li, Yufei Chen, Xiaodong Yue, Xujing Zhou, Qunjie Chen. (2026). _Stop Guessing: Choosing the Optimization-Consistent Uncertainty Measurement for Evidential Deep Learning_. ICLR 2026.** [正式页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/55aeba84b402008d3ed10440d906b4e1-Abstract-Conference.html)；[正式全文](https://proceedings.iclr.cc/paper_files/paper/2026/file/55aeba84b402008d3ed10440d906b4e1-Paper-Conference.pdf)。

- 任务/数据：图像分类/OOD/拒判；CIFAR-10/100，OOD 包括 SVHN、GTSRB、Places365、Food101；附视频开集。模型含 VGG16。
- 对象/方法：从 expected cross-entropy 优化动态提出 optimization-consistency，并构造 Margin-aware Predictive Uncertainty（MPU）；比较不同 EDL 指标。
- 原文 §5 限制：UCE+MPU 的优势非所有损失、所有类别数成立；CIFAR-10 上不及 Brier 训练的 Re-EDL。它不是证明某指标等于真实无知。
- 开放性：论文及 supplemental 入口公开，代码状态本次未核。关联是“选择 EDL 不确定性量不能凭名字”的新证据，非新闻任务基线。

### U16 2026 年 epistemic 估计的不完整性

**Sebastian Jimenez, Mira Juergens, Willem Waegeman. (2026). _Position: Epistemic Uncertainty Estimation Methods are Fundamentally Incomplete_. ICML 2026, PMLR 306, 170691–170717.** [正式页](https://proceedings.mlr.press/v306/jimenez26a.html)。

- 类型：Position/理论论证，非新闻系统；讨论二阶分解遗漏 bias 和仅捕获部分 variance 来源。
- 已核结论：不能把一种 uncertainty estimator 解释成全部模型知识缺失；作者主张应明确告知适用限制。
- 核验层级：正式题录、摘要、software 入口已核；raw PDF 因浏览读取错误未展开，故未补写定理、数据与数值。此条用于 2026 时效性及概念边界，不能当实验效果证据。

### U17 Semantic Energy：保留但不能照单全收

**Huan Ma, Jiadong Pan, Jing Liu, Yan Chen. (2025). _Semantic Energy: Detecting LLM Hallucination Beyond Entropy_. arXiv:2508.14496.** [arXiv 当前题录](https://arxiv.org/abs/2508.14496)；[本次精读 v1](https://arxiv.org/html/2508.14496v1)。当前题录修订到 2025-12-01 v3，未核到正式发表会刊，不将其当已同行评审结论。

- 数据/方法：中英 QA，CSQA、TriviaQA；Qwen3-8B、ERNIE-21B-A3B，语义聚类后利用 logits/energy；指标 AUROC、AUPR、FPR95。
- 测量：logit 导出的分数与错误关联，不是已可辨识的真实知识量。原文给出一致错误使 SE=0 的反例，与 U02 已承认边界一致。
- 作者限制（§6）：logits 并非严格物理 energy。原文称“SE only reflects aleatoric”是该论文论断，不是领域共识；U02 明言不进行这种分解。
- 开放性：原文链接 [SemanticEnergy](https://github.com/SemanticEnergy) 为组织级入口，本次未验证具体仓库完整性。关联是替代分数与负例提醒；不能以其标题保证解决一致错误。

### U18 Semantic Reformulation Entropy：2025 预印本已出现 2026 接收证据

**Chaodong Tong, Qi Zhang, Lei Jiang, Yanbing Liu, Nannan Sun, Wei Li. _Semantic Reformulation Entropy for Robust Hallucination Detection in QA Tasks_. ICASSP 2026.** [作者原文](https://arxiv.org/html/2509.17445v1)；[arXiv](https://arxiv.org/abs/2509.17445)；[会议正式程序](https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=1916&bare=1)。IEEE 索引给出 [DOI 10.1109/ICASSP55912.2026.11460452](https://doi.org/10.1109/ICASSP55912.2026.11460452)，直开 IEEE 受限，但会议程序核验了接收与作者，不能仅因 arXiv comments 仍写 submitted 就当未发表。

- 方法/任务：输入释义改写加 exact match/embedding/NLI 混合语义聚类，文本 QA；SQuAD-v2、TriviaQA，各抽取 500 validation 实例；Llama3-8B、Qwen3-14B。
- 测量：释义及采样下语义输出分散，评价错误识别 AUROC/AUPR 等；作者声称更稳定，不等于已独立解耦 epistemic。
- 局限（审计推论）：核验只覆盖两 QA 数据，缺少图像与新闻外部证据实验。输入改写须确认不改变语义，否则把改写失败当不确定性。
- 开放性：论文公开；未核到作者代码。它削弱“输入扰动 + 熵即可作为新方法”的宽泛新颖性主张。

## A 方案数学审计：熵与 JS 只支持操作性区分

下面是审计团队的推导与研究设计建议，**不是前述论文作者已经论证的新闻结论**。

### 共同标签空间必须先成立

对锚点类型 `a∈{entity,event,time,location}` 定义有限候选集合 `C_a={c1,…,cK}`；三种条件使用相同候选、同序列化、同温度定义：

\[
q_T^a(c)=p_\theta(c\mid T,Q_a,C_a),\quad q_V^a(c)=p_\theta(c\mid V,Q_a,C_a),\quad q_{TV}^a(c)=p_\theta(c\mid T,V,Q_a,C_a).
\]

这在工程上可做 constrained decoding 或候选序列 log-likelihood，但不是天然可比较：

1. 图像单模态仍需要一个不泄露文本所声称时间/地点的锚点问题 `Q_a`。若问“图片是否是 2024 年巴黎事件”，就不是纯图像知识预测。
2. 三个输出必须是同一随机变量（如图片真实地点），不能分别是“文本提及地点”“图片可见地点”“新闻是否假”；否则 JS 无意义。
3. 对无视觉线索的确切日期，应标注“图像无法确定”作为单独的 evidence availability 标签。拒答/不可见与候选实体值应分开记录；“未知”输出 token 的高概率不能直接替代 epistemic 熵。
4. 候选集须在训练/协议阶段固定或用不读取金标的程序生成；错误候选截断可制造高置信度。报告真实候选召回率与 `other`。不同样本 K、候选难度影响归一化熵 `H/logK`，归一化不能消除候选构造偏差。
5. 同一 VLM 去掉一种模态可能落入模型未训练过的缺模态分布；对照独立文本/图像探针及原生缺模态模式。不能把 missing modality artifacts 当冲突。

### 定义与不可辨识反例

\[
H(q)=-\sum_cq(c)\log q(c),\quad
JS(q_T,q_V)=\tfrac12 KL(q_T\Vert m)+\tfrac12 KL(q_V\Vert m),\quad m=(q_T+q_V)/2.
\]

- **同熵，机制不同**：`qT=qV=qTV=(0.5,0.5)` 可来自两个模态都没有线索，也可来自两个低质量或冲突证据源使模型保守；所有熵相同、JS=0，无法从这些数值还原成因。
- **自信冲突与自信错误混淆**：`qT=(.99,.01), qV=(.01,.99)` 的 JS 大。一种现实是图文确实指向不同事件；另一种现实是图文完全一致但视觉模型识错、OCR 失败、候选标签映射反转。观测完全相同，不能据此断言输入语义冲突。
- **共同自信错误漏检**：`qT=qV=qTV=(.99,.01)`、真实值为第二类；H 低、JS=0，却完全错误。模型共享偏差和事件记忆可导致此情形。
- **互补证据不是冲突**：T 与 V 分别提供不足线索，联合后确定；联合熵下降可以是互补，不代表某模态原先无知或原输入冲突。
- **EDL 同均值不同浓度**：Dirichlet(1,1) 与 Dirichlet(100,100) 都预测 `(0.5,0.5)`，同预测熵；其 vacuity 不同。增加一个浓度输出增加表达能力，却不解决从标签监督中可辨识真实成因的证明义务。

所以可以把 `H` 命名为“预测分散分数”，`JS` 命名为“模态预测分歧分数”。若要改称“无知检测器/冲突检测器”，必须使用独立的来源标签与干预验证其选择性响应，而不是以自身阈值生成真值再验证自身。

### 什么时候 JS 能称为互信息

设人工变量 `M` 等概率选择文本或图像预测分布，再按该分布抽取候选 `Y`，则 `I(M;Y)=JS(qT,qV)`。这是该**人工混合分布**下的恒等式；它不等于自然数据中的 `I(V;Y|T)`、真实冲突量或检索信息增益。

真正的条件信息增益应在联合数据模型下定义：

\[
I(Y;E\mid T,V)=\mathbb E[\log p(Y\mid T,V,E)-\log p(Y\mid T,V)].
\]

实际模型两次输出熵相减也不等于该量。反例：检索前 `(0.5,0.5)`，假证据后对错误类别变 `(0.99,0.01)`，熵下降但真实性恶化。评估必须看金标上的损失/准确性变化与证据质量。

### 最小校准与验证要求

| 必要检查 | 它能排除什么 | 仍不能声称什么 |
|---|---|---|
| 固定候选与提示，多标签 verbalizer/顺序置换 | token先验、标签顺序造成的分布差异 | 不能证明跨模态认知同质 |
| 文本、图像、联合三条件分别在独立开发集做温度校准 | softmax锐度差异 | 不能保证语义冲突识别或 OOD 校准 |
| NLL/Brier + reliability图；ECE报告分箱规则 | 单一ECE或AUROC掩盖错误置信度 | ECE小不代表每个样本可靠 |
| 同一事件的正确配对、错地点、错时间、不可见时间、模糊/降质版本 | 冲突 vs 可用信息减少的交叉混淆 | 仅合成成功不代表真实谣言成功 |
| 人工/权威来源标注“是否有足够图像线索”“是否图文矛盾” | 用模型自己定义无知与冲突的循环论证 | 仍非直接观测神经模型知识本体 |
| 事件/时间隔离；提示/种子/量化精度复验 | 事件记忆与不稳定代理 | 不能宣称通用无知分解 |

可行目标改为：**在来源和可判定性有独立标注的锚点样本上，预测分歧是否为“外部证据检索能纠错”提供超越单次联合 MSP/熵的增量预测价值。** 这是可证伪实证问题，允许答案为否。

### 按需检索必须计入所有代价

令便宜前置模型为 `f0`、检索后为 `f1`，`r(x)∈{0,1}` 决定检索：

\[
\hat y=(1-r)f_0(x)+rf_1(x,E),\quad
Cost=C_{pre}+r(C_{ret}+C_{post}).
\]

分类输出应按分支选择，上式表示系统路径而非混合离散类标。预算比较至少包括：never retrieve、always retrieve、同检索率 random、联合 MSP/entropy gate、候选 A gate；oracle benefit gate 仅作为不可部署上界。若 A 先做三模态多次采样，`Cpre` 可能大于一次检索，不能只统计 API 次数。

在固定证据池和相同检索器下，对每个样本离线记录检索前/后是否正确，形成“纠错/伤害/均对/均错”四格；验证 gate 是否主要选中纠错而非仅选难例。匹配同预算后比较 Macro-F1 或错误风险，开发集锁阈值，事件级 bootstrap。正文不能把“检索后总体分数提高”单独解释成证据因果有效。

证据对照应有 gold/相关证据、主题相关但不支持、随机、同源重复、删去时间地点句、矛盾或失实证据，并控制证据长度与条数；严禁把 fact-check 文章标题、金标词或测试标签注入检索。若错误证据与正确证据同样提升，优先判为泄漏/提示效应；若检索只是使模型更自信却未减少错误，应推翻“有效信息增益”解释。

## 给总审计的直接建议

1. 将“无知—冲突已可分”降为待检验；保留预测分散/模态分歧的操作性名字。
2. A 的可辩护空白应是特定新闻锚点上、固定预算且事件外测试的“增量检索效用预测”，不是再次设计 entropy+JS 门控。
3. B 的扰动 JS 应称分布敏感性；只有干预对照及金标改善才能谈有益视觉依赖。
4. C 至少面对 UDRL/AEMRL 的融合已有性，以及昂贵采样可由 SEP 类探针替代的成本基线；不能仅列技术组件讲新故事。
5. U07/U08/U16 部分正文未能展开；UDRL 全文受限。最终交付应保留这些核验边界，不能将“未读”变成具体局限或“不存在”。
