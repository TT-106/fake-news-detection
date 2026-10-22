# 数据集、OOC 与外部证据核查证据审计

检索截止：2026-10-19。此文件是证据调查附件，不是系统综述或实验结果。本轮打开并核对了正式论文页面、部分论文正文与作者仓库 README；没有下载全量数据、申请受限数据、调用收费 API 或运行任何模型。页面可达、仓库声明开源，不等于数据已完整取得或代码已经复现。

## 1. 需要优先修正的判断

1. **真实图片配真实文本形成的合成错配，不等于经过核验的真实时空谣言。** NewsCLIPpings 的标注是 pristine/generated mismatch；不能由其中的高分直接推出真实新闻的时间、地点核查能力。[正式原文](https://aclanthology.org/2021.emnlp-main.545/)
2. **“时空锚点恢复 + 外证据”已经有直接近邻。** 5Pils、COVE 和 NewsRECON 分别研究图像原始元语境、先恢复语境再核查、无反向图搜证据时的新闻检索。研究空白必须落在校准、检索收益预测或受控诊断上，不能写成此前没人恢复图像时间地点。[5Pils](https://aclanthology.org/2024.emnlp-main.448/)、[COVE](https://aclanthology.org/2025.naacl-long.102/)、[NewsRECON 最新版本](https://arxiv.org/abs/2601.14121)
3. **最便宜的强证据基线不必是微调大模型。** MUSE 用图文与外证据的相似度特征配 RF/MLP，AITR 加强融合；其正式版本是 WACV 2025，2024 是预印本年份。[正式论文](https://openaccess.thecvf.com/content/WACV2025/html/Papadopoulos_Similarity_over_Factuality_Are_we_Making_Progress_on_Multimodal_Out-of-Context_WACV_2025_paper.html)
4. **AVerImaTeC 是本项目必须纳入的同任务新数据，但不适合无条件当四分类训练主集。** 原始训练集 95.3% 是 refuted；真实核查和证据诊断价值高，少数类校准统计则很弱。[正文 §5、Table 2、§8](https://arxiv.org/html/2505.17978v2)
5. **VERITE 和 VeriTaS 是不同数据集。** 前者 1,000 个图文对，后者是 ACL 2026 的动态、多语、多模态事实核查资源；也不要把 ICLR 的同名 Veritas 深伪检测器混入。[VERITE](https://link.springer.com/article/10.1007/s13735-023-00312-6)、[VeriTaS](https://aclanthology.org/2026.acl-long.1948/)

## 2. 检索过程和核验等级

关键词组包括 `out-of-context misinformation evidence`、论文准确标题、`NewsCLIPpings VERITE MUSE AITR`、`COVE 5Pils image contextualization`、`AVerImaTeC`、`VeriTaS`、`multimodal fact checking 2025 2026`、`MOCHEG`、`Fakeddit`、`MDFEND Weibo21`。先查 ACL/CVF/AAAI/NeurIPS 正式记录，再回溯作者仓库，最后补看 arXiv 最新修订与正文。研究区间以 2022—2026-10-19 为主，2020/2021 数据工作作为必要基础。

这不是穷尽检索：不报告虚假的 PRISMA 总命中数，没有查遍全部索引，也没有进行完整撤稿数据库核验。出版存在性以正式论文集记录为强证据；代码以作者链接的仓库为证据。并未调用跨模型核验脚本或原生 ARS 插件。参考了本地 ARS 的 bibliography/source-verification 角色约束，将作者自承局限和审计推论分开；未机械套用医学 RCT 等级来评定计算机基准论文。

## 3. 逐项文献核验

表中“指标”指方法实际计算或输出的对象，不能读作已建立可信的不确定性估计。未阅读全文的条目，不做作者方法局限的断言。

| 编号、准确文献信息 | 任务、数据、方法和真正测量对象 | 已核实结论、开放资源、与本研究关系、局限出处 |
|---|---|---|
| P01 Grace Luo, Trevor Darrell, Anna Rohrbach. **NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media**. EMNLP 2021, 6801–6817. DOI [10.18653/v1/2021.emnlp-main.545](https://aclanthology.org/2021.emnlp-main.545/) | VisualNews 图文对，通过 CLIP/SBERT/场景相似度构造错配，评估 CLIP 等多模态基线。测量配对能否区分原始/生成错配，不测外部世界真值或无知。 | 有[代码、配对索引、部分预提取特征](https://github.com/g-luo/news_clippings)。直接关联高；审计推论：生成机制可能形成捷径，因此需要真实外部测试，不能只依据合成测试。阅读：正式摘要、数据表、构造说明、README；未做全面方法质量评议。 |
| P02 Sahar Abdelnabi, Rakibul Hasan, Mario Fritz. **Open-Domain, Content-based, Multi-modal Fact-checking of Out-of-Context Images via Online Resources**. CVPR 2022, 14940–14949. [正式 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Abdelnabi_Open-Domain_Content-Based_Multi-Modal_Fact-Checking_of_Out-of-Context_Images_via_Online_Resources_CVPR_2022_paper.pdf)、[arXiv:2112.00061](https://arxiv.org/abs/2112.00061) | CCN：反向图搜获取文本、文字搜图获取视觉证据，比较文—文、图—图、图—文一致性；NewsCLIPpings merged-balanced。attention/memory 汇聚是证据选择，不是不确定性分解。 | 直接同任务外证据基线。[官方页](https://s-abdelnabi.github.io/OoC-multi-modal-fc/)提供证据 URL；已爬取内容需要申请。[代码](https://github.com/S-Abdelnabi/OoC-multi-modal-fc)。论文报告优于无外证据基线；审计推论：固定历史搜索缓存和在线搜索不是同一实验条件。作者补充材料强调攻击风险、人类应保留在环；不可据此声称所有检索有因果增益。 |
| P03 Shivangi Aneja, Chris Bregler, Matthias Niessner. **COSMOS: Catching Out-of-Context Image Misuse Using Self-Supervised Learning**. AAAI 2023, 37(12), 14084–14092. DOI [10.1609/aaai.v37i12.26648](https://ojs.aaai.org/index.php/AAAI/article/view/26648) | 图像对象—caption 对齐；测试给一张图和两条caption，判断是否对齐相同对象却语义冲突。其输入条件不同于只给一个新闻图文对。 | [代码](https://github.com/shivangi-aneja/COSMOS)，数据经表单申请。正式标题与 2021 预印本标题不同。作者正式页报告约 200K 图、450K captions；正文表为160K训练图、40K验证图、1,700测试三元组。VERITE 作者检验了 COSMOS 单模态偏差；这是后续论文的实证结果，不是本审计凭空指控。 |
| P04 Stefanos-Iordanis Papadopoulos, Christos Koutlis, Symeon Papadopoulos, Panagiotis C. Petrantonakis. **VERITE: a Robust benchmark for multimodal misinformation detection accounting for unimodal bias**. International Journal of Multimedia Information Retrieval 13, article 4, 2024. DOI [10.1007/s13735-023-00312-6](https://link.springer.com/article/10.1007/s13735-023-00312-6) | Snopes/Reuters 核查材料构造 truthful/miscaptioned/OOC 配对，模态平衡；DT-Transformer、CHASMA合成训练。测跨模态真假配对。 | [官方仓库](https://github.com/stevejpapad/image-text-verification)。作者识别 COSMOS/VMU-Twitter 单模态偏差；2026-07 README明确原图片URL大量失效，增加[受限HF图像](https://huggingface.co/datasets/stefpapad/VERITE)。直接真实迁移评估，不能拆开同一原始实例的成组样本随机训练测试。最后一点是审计推论。 |
| P05 Peng Qi, Zehong Yan, Wynne Hsu, Mong Li Lee. **SNIFFER: Multimodal Large Language Model for Explainable Out-of-Context Misinformation Detection**. CVPR 2024, 13052–13062. [官方项目](https://yanzehong.github.io/Sniffer/)、[arXiv:2403.03170](https://arxiv.org/abs/2403.03170) | InstructBLIP两阶段指令调优；内部图文核查+外部图像上下文核查，融合判断与解释；NewsCLIPpings，额外真实测试。测判别与解释，不测严格校准。 | 项目和论文有代码链接。作者正文明确某些新闻事件仅凭图像无法识别，需外部上下文；这是A的必要性依据，不能推出A的熵足以定位这种无知。全量重新指令调优不是4060首轮实验首选；该资源建议属审计判断。 |
| P06 Stefanos-Iordanis Papadopoulos, Christos Koutlis, Symeon Papadopoulos, Panagiotis C. Petrantonakis. **Similarity over Factuality: Are we Making Progress on Multimodal Out-of-Context Misinformation Detection?**. WACV 2025, 5570–5579. [正式页](https://openaccess.thecvf.com/content/WACV2025/html/Papadopoulos_Similarity_over_Factuality_Are_we_Making_Progress_on_Multimodal_Out-of-Context_WACV_2025_paper.html) | MUSE：图文/查询—证据相似度特征 + DT/RF/MLP；AITR：attentive intermediate Transformer；NewsCLIPpings和VERITE。不是semantic entropy，也不测证据真实性。 | 作者报告简单相似度可匹敌复杂方法，并明确警示surface shortcuts与factuality脱节。是必须先击败的轻量同任务基线。[官方代码](https://github.com/stevejpapad/outcontext-misinfo-progress)，需要VisualNews、NewsCLIPpings、CCN/RED-DOT证据。 |
| P07 同上四位作者. **RED-DOT: Multimodal Fact-checking via Relevant Evidence Detection**. 作者官方仓库注明 IEEE Transactions on Computational Social Systems, 2025；[arXiv:2311.09939](https://arxiv.org/abs/2311.09939) | CLIP特征、相关证据检测、重排与guided attention，NewsCLIPpings+与VERITE。相关性/支持反驳证据筛选不等于获取证据前的按需检索。 | [官方代码与证据入口](https://github.com/stevejpapad/relevant-evidence-detection)。期刊年份由官方README核实，未独立核到IEEE最终DOI/卷页，故不填写。预印本和README提升数字不同，本审计不混用。阅读：摘要/README；方法局限不作完整评议。 |
| P08 Jonathan Tonglet, Marie-Francine Moens, Iryna Gurevych. **“Image, Tell me your story!” Predicting the original meta-context of visual misinformation**. EMNLP 2024, 7845–7864. DOI [10.18653/v1/2024.emnlp-main.448](https://aclanthology.org/2024.emnlp-main.448/) | 5Pils：1,676核查图像，来源/沿用历史/日期/地点/动机QA，RIS文本与图像回答。任务是context恢复，不直接等于二分类假新闻。 | [代码/数据](https://github.com/UKPLab/5pils)。原文§8作者自承只收英文核查文章导致地区分布偏斜、过滤核查文章不能保证完美召回；为潜在标签泄漏提供真实依据。直接关联高，适合锚点QA诊断。 |
| P09 Jonathan Tonglet, Gabriel Thiem, Iryna Gurevych. **COVE: COntext and VEracity prediction for out-of-context images**. NAACL 2025, 2029–2049. DOI [10.18653/v1/2025.naacl-long.102](https://aclanthology.org/2025.naacl-long.102/) | RIS网页caption、Wikipedia视觉实体、自动caption→恢复语境/填补知识缺口→规则与Llama3核真；NewsCLIPpings、5Pils-OOC；比较语境项质量及OOC判别。 | [官方实现](https://github.com/UKPLab/naacl2025-cove)，部分中间证据已缓存。作者§6承认多步LLM调用代价；§4.1说明5Pils-OOC真caption由GPT-4基于gold context生成。其先语境后核真已直接覆盖泛化版“锚点化外证据”；新颖性必须另证。 |
| P10 Fanxiao Li, Jiaying Wu, Canyuan He, Wei Zhou. **CMIE: Combining MLLM Insights with External Evidence for Explainable Out-of-Context Misinformation Detection**. Findings of ACL 2025, 9342–9354. DOI [10.18653/v1/2025.findings-acl.487](https://aclanthology.org/2025.findings-acl.487/) | GPT-4o等先生成图文可共存关系，给证据打association分数，再核查；NewsCLIPpings等；分数来自LLM-as-judge，没有校准概率含义。 | [代码](https://github.com/fanxiao15/CMIE)。正文§2展示添加外证据并非所有指标都变好；§3.2明载共存分数没有显式定义；这限制把这些score当概率解释。这里是原文设计事实，非声称作者实证证明分数“不可靠”。 |
| P11 Uku Kangur, Krish Agrawal, Yashashvi Singh, Ahmed Sabir, Rajesh Sharma. **MultiReflect: Multimodal Self-Reflective RAG-based Automated Fact-Checking**. MAGMaR workshop 2025, 1–17. DOI [10.18653/v1/2025.magmar-1.1](https://aclanthology.org/2025.magmar-1.1/) | 多模态RAG、一致性检查、外证据检索过滤排序与反思，VERITE二分类。 | [官方代码](https://github.com/ukangur/MultiReflect)，默认gpt分支调用商业模型。确有动态外证据与反思，不等于已证明“熵门控+无知/冲突可辨识”。阅读：正式摘要、正文开头、README；局限不作完整评议。 |
| P12 Jonathan Tonglet, Iryna Gurevych, Tinne Tuytelaars, Marie-Francine Moens. **NewsRECON: News Article Retrieval for Image Contextualization**. 2026；arXiv v2注明已接收 EMNLP 2026 Findings，未核到正式卷页。 [arXiv:2601.14121v2](https://arxiv.org/abs/2601.14121) | 无RIS证据时，图像检索新闻语料、按地点/事件cross-encoder重排，恢复时间地点；TARA、5Pils-OOC。v2语料超过85K（不要抄v1约91K）。 | [作者代码链接由arXiv可达](https://arxiv.org/abs/2601.14121)。直接同锚点/证据缺失问题；正式摘要声称无RIS条件胜过此前方法并跨地域/时间迁移。阅读：最新摘要、HTML方法；尚未逐项复核所有实验局限。 |
| P13 Barry Menglong Yao, Aditya Shah, Lichao Sun, Jin-Hee Cho, Lifu Huang. **End-to-End Multimodal Fact-Checking and Explanation Generation: A Challenging Dataset and Models**. SIGIR 2023, 2733–2743. DOI [10.1145/3539618.3591879](https://doi.org/10.1145/3539618.3591879) | MOCHEG文本声明→多模态证据检索→supported/refuted/NEI→解释。图像主要是证据模态，不保证每条claim有查询图。 | [官方资源](https://github.com/PLUM-Lab/Mocheg)、[大学正式存档](https://vtechworks.lib.vt.edu/items/d702f18a-9471-44e1-b580-6a13355df127)。15,601 claims/33,880文本段/12,112 evidence images。适合claim核查，不宜当图文配对OOC检测同任务基线。阅读：正式摘要/README，不推断具体架构局限。 |
| P14 Michael Sejr Schlichtkrull, Zhijiang Guo, Andreas Vlachos. **AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web**. NeurIPS 2023 Datasets and Benchmarks. [正式页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd86a30526cd1aff61d6f89f107634e4-Abstract-Datasets_and_Benchmarks.html) | 4,568真实文本claims，QA分解与web证据，4类verdict，强调时间约束。 | [官方代码/数据](https://github.com/MichSchli/AVeriTeC)。方法可做A的文本任务对照，不是OOC同任务系统。开放网络基线依赖搜索API；静态knowledge-store与在线搜索成本不可混算。阅读：正式记录和README；不作完整方法局限评议。 |
| P15 Rui Cao, Zifeng Ding, Zhijiang Guo, Michael Schlichtkrull, Andreas Vlachos. **AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web**. NeurIPS 2025 Datasets and Benchmarks. [正式页](https://proceedings.nips.cc/paper_files/paper/2025/hash/0695cbd1f85f54780e2b351db4252a97-Abstract-Datasets_and_Benchmarks_Track.html) | 1,297真实图文claims，QA/外证据、4类verdict、metadata；模型提问→选VQA/RIS/web等工具→回答与verdict。 | [正文](https://arxiv.org/html/2505.17978v2)、[代码](https://github.com/abril4416/AVerImaTeC)、[数据](https://huggingface.co/datasets/Rui4416/AVerImaTeC)。§8作者自承小规模、类别偏斜、约5%claim日期用核查文章日替代且网页时间粗估、reference-based评估漏掉合理新证据。是主任务强候选，不能称完全杜绝泄漏。 |
| P16 Mark Rothermel, Marcus Kornmann, Marcus Rohrbach, Anna Rohrbach. **VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking**. ACL 2026, 42066–42100. DOI [10.18653/v1/2026.acl-long.1948](https://aclanthology.org/2026.acl-long.1948/) | 每季度更新、多语文本图像视频、自动标准化多维verdict；正式论文25K/54语言，官网最新快照27K。测跨模态真实claim核查，动态更新降低记忆污染。 | [官方站](https://veritas.mai.informatik.tu-darmstadt.de/)明确需申请access。不能保证“无泄漏”，只能说设计针对这一风险。paper与网站统计不同属版本变化，记录snapshot。阅读：正式摘要、官网；方法局限 not assessed（abstract only）。 |
| P17 Weilin Zhou, Zonghao Ying, Chunlei Meng, Jiahui Liu, Hengyang Zhou, Quanchen Zou, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang. **DIVER: Dynamic Iterative Visual Evidence Reasoning for Multimodal Fake News Detection**. 2026-01-12预印本。[arXiv:2601.07178](https://arxiv.org/abs/2601.07178) | 文本先行→必要时引入图像/细粒度OCR与dense caption→uncertainty-aware融合；Weibo/Weibo21/GossipCop。 | 同A/C的流程近邻；OCR与caption工具抽取的内部视觉信息不能称原始事件外部溯源。作者摘要报告效果和延迟，但本轮未复核计时口径；不据此设性能承诺。代码开放性未完成核验。 |

**待核名目**：精确检索 “Detecting Out-of-Context Images with Evidence” 未得到可确认同名正式记录，不能编造引用。用户可能指 P02，但标题不同，应由原始引用确认；不能把搜索无精确命中写成该论文不存在。

## 4. 数据集可用性与适配

所有“可达”均指2026-10-19本轮打开的页面/文件清单，**没有完成原图下载和解压完整性验证**。代码许可和新闻图片原作者版权不是同一层。

| 数据集 | 任务、规模/语言/类别 | 图文/声明/外证据 | 入口、许可、当前状态 | 本任务适配、拆分与偏差风险 |
|---|---|---|---|---|
| NewsCLIPpings merged-balanced | 英文；71,072/7,024/7,264 train/val/test；每split pristine/falsified各半。整个库 unique约988K，不能把整个库与merged-balanced混报。 | 查询图+caption齐；原生无gold核查依据；CCN扩展提供检索证据。 | [官网代码](https://github.com/g-luo/news_clippings)，须另取VisualNews；README列metadata timestamps与location。原图和证据下载未验证；不可从GitHub开源自动推导新闻图片许可。 | 合成错配训练/受控实验合适；真实时空错位论证不足。按原始image/caption/事件簇分组；有时间metadata才可重建时间分割。CLIP生成负样本、CLIP评测可能有构造依赖，是审计风险假设。 |
| VERITE | 英文；1,000图文对，338 truthful、338 miscaptioned、324 OOC。 | 图文齐；核查文章URL可追踪；原生不是完整QA证据库，RED-DOT另供检索证据。 | [官方](https://github.com/stevejpapad/image-text-verification)、[HF](https://huggingface.co/datasets/stefpapad/VERITE)。仓库Apache2；数据research-only，HF机构邮箱申请、禁止再分发。2026-07图片失效公告；未申请或下载。 | 小型真实迁移评估，不能用它反复调阈值。原始文章/图片衍生组必须共同分割；模态平衡降低特定单模态偏差，不保证所有偏差消失。README样本数提338篇又称CSV337行，后续需以实际文件清点为准。 |
| 5Pils / 5Pils-OOC | 5Pils：1,676核查图像及context QA；OOC：624图片、1,248二元图文对各半。英文claims，证据可能多语。 | 有时间/地点/source等标签、RIS证据URL；OOC假caption真实，正确caption由GPT-4基于gold生成。 | [5Pils](https://github.com/UKPLab/5pils)、[COVE](https://github.com/UKPLab/naacl2025-cove)。数据CC-BY-SA4、代码Apache2；只给原图URL需自行取。仓库有中间context输入；本轮未下载图。 | **最接近可解释时空锚点诊断**；作为独立test，不把同图真假caption拆开。需控制GPT真caption文风捷径；不能称全人写真实正负。核查来源地区分布与可回收图片缺失可能选择偏差。 |
| AVerImaTeC | 英文真实图文claims 1,297；train793/dev152/test352。S/R/C/N比例：train1.6/95.3/0.8/2.3%；dev2.6/92.8/0.7/3.9%；test13.9/78.1/2.0/6.0%。 | 有claim图、文本、QA/外证据URL、声明日期等；有些claim多图。 | [HF](https://huggingface.co/datasets/Rui4416/AVerImaTeC/tree/main)、[代码](https://github.com/abril4416/AVerImaTeC)。HF card无许可metadata；论文checklist称资产CC-BY-NC4（arXiv正文许可证CC-BY-NC-SA是文稿，不应混淆）。文件清单确有train.json、val.json、images.zip(221.5MB)、test_data.zip(31.7MB)。卡仍称test将发；**没解包确认test labels**。 | 任务最贴合外证据真实核查；少数类过少，不宜直接做稳健四类校准研究。已有时间划分，截止2023-05-31/2023-07-31/2025-03-21。可按claim type筛图像事件溯源子集，但不能先看test结果再筛。 |
| VeriTaS | 正式版25K，54语言；官网快照27K，图像28.1%、视频20.7%、二者1.4%、纯文本49.8%；是多维核查标签，非直接二元OOC。 | 真实claim与媒体、justification；是否对选定子集有完整可复用原始外证据，须获准后查schema。 | [官网](https://veritas.mai.informatik.tu-darmstadt.de/)需申请；本轮未获访问，不报下载成功/具体许可。 | 适合后续严格近期/时间外测试；初期不宜承诺依赖受限资源。自动规范化和多语跨域会引入不同测量难度，不能与VERITE裸分横比。 |
| COSMOS | 英文为主；训练160K图、验证40K图、测试1,700三元组；约450K captions，不是450K独立图片。 | 测试给图和两条caption；非每样本已配独立外证据。 | [代码/数据申请](https://github.com/shivangi-aneja/COSMOS)。表单访问要求，未取得；许可未完成核查。 | 有两caption额外输入信息，不能直接同单caption OOC系统比；VERITE论文已展示单模态偏差。重建group/event划分需metadata，不能假定官方random split是event split。 |
| DGM4 | 英文human-centric新闻，230,000 =77,426 pristine+152,574 manipulated；4种manipulation可重叠。 | 图文+face bbox/token manipulation位置；主要FS/FA/TS/TA局部编辑，非原生图文外证据。 | [官方](https://github.com/hsrr/DGM4)，CVPR2023 Rui Shao/Tianxing Wu/Ziwei Liu **Detecting and Grounding Multi-Modal Media Manipulation**；仓库提供数据和checkpoint。下载及数据许可未验。 | 用来做篡改/局部定位辅测，不能代替真实图片OOC。32,693 mixed操作使类型数不可直接相加为独立样本；VisualNews基础与NC可能同源，必须跨集去重。 |
| Fakeddit | 英文Reddit，1,063,106总条目；2/3/6级标签，真、讽刺、误导、伪装、假关联、操纵等；总数不等于图文齐全数。 | 图像/标题/metadata/comments；无统一声明级gold外证据；标签根据subreddit等弱监督。 | [论文](https://aclanthology.org/2020.lrec-1.755/)、[数据/图片包](https://github.com/entitize/Fakeddit)。官方要求可比论文结果仅用multimodal_only_samples；许可未核明，issues存在license询问，不能写CC许可确定。 | 通用多模态内容分类背景或预训练可用，主任务不优先；来源/风格可预测标签与事件真假不同。timestamp可支持重划分但本轮未清点字段完整率。 |
| Weibo经典2017 | 中文 rumor/non-rumor；文献常见4,749/4,779原始口径，各后处理版本人数不同。 | post+images，可有社交上下文；无统一事件时空错误/外证据标签。 | 原始Jin等2017资料与当前可用图像镜像未独立完整核查；不能把后继论文处理版本当唯一官方版。 | 不宜作为本项目真实OOC主数据；须固定具体版本/hash、label映射、缺图过滤和event metadata，避免“都叫Weibo”导致复现不一致。本行规模仍需原论文/发布包交叉核对。 |
| Weibo21 | 中文9域；原MDFEND正文4,488 fake+4,640 real=9,128。网上9,127、7,610等可能是后处理集，不能混用。 | 原文描述text/images/timestamp/comments、fake判断信息；**MDFEND公开split用于文本模型，不能保证图像可用**。 | [官方库](https://github.com/kennqiang/MDFEND-Weibo21)、[论文](https://arxiv.org/abs/2201.00987)。原始数据需申请表；访问许可未验。 | multi-domain rumor classification，不是OOC gold；domain划分不等于event/time holdout。可用于DIVER等近邻任务复现，但不验证锚点冲突。 |
| MOCHEG | 英文15,601声明，supported/refuted/NEI；本轮未独立清点各类/拆分，不填猜测。33,880 evidence paragraphs+12,112 images。 | 文本claim+多模态证据+rule statement；不保证输入claim有新闻图像。 | [官方](https://github.com/PLUM-Lab/Mocheg)含数据/预训练模型入口与dataset statement；数据CC-BY4、代码Apache2。本轮仅入口可达。 | 外证据检索和verdict评价适合；图文冲突任务不直接适配。训练/验证之间claim-source、事件和核查文章复用需检查；这是待检风险，不是已证实泄漏。 |
| AVeriTeC | 英文4,568真实文本声明，4类supported/refuted/NEI/conflicting-cherrypicking；无固定查询图模态。 | QA/URL/日期/justification完备，重视时间约束。 | [官方](https://github.com/MichSchli/AVeriTeC)CC-BY-NC4；JSON及baseline公开，在线搜索需API；官方协议knowledge-store另核。 | 可验证文本侧按需检索，但无法独立验证视觉创新。不能将conflicting evidence标签直接视为两模态分布冲突。 |

## 5. 供主审计采用的实验边界建议

这些是研究设计建议，不是实验结果，也不是作者原文局限断言。

### 任务和数据的最小组合

研究任务先收窄为：**对“这张真实照片描绘某事件/时间/地点”的图文声明，何时需要获取外部上下文，如何在既定检索预算内改善错误判断？** 泛化版真假新闻、编辑篡改检测与文字claim核查保留为不同任务。

最小资源路线：NewsCLIPpings merged-balanced用于训练轻量头和验证流程；VERITE、5Pils-OOC为锁定真实迁移集；AVerImaTeC图像溯源子集作QA/证据链有效性诊断，保持官方时间顺序。不能靠NC一个数据集支持真实时空核查结论。若原图回收、授权或证据缓存拿不到，先以公开CLIP特征做MUSE/AITR的**成本与校准可行性检查**，暂缓涉及遮挡或VLM token概率的部分。

### 公平基线

1. 文本CLIP frozen feature + logistic regression/MLP；图像同规模头；图文concat或相似度+MLP；相同split和训练预算。
2. MUSE-RF/MLP（最近、轻量、证据同任务），资源许可后加AITR/RED-DOT；不得只比没有检索能力的弱基线。
3. 不检索、全部检索、随机相同比例检索、max-softmax/entropy阈值检索、CLIP相似度阈值检索、候选门控、离线上界oracle门控。
4. 所有门控共用同一冻结证据pool和同一核查器。比较真实联网版本时另统计搜索费用与失败，不能把缓存lookup毫秒写成线上检索延迟。
5. 外证据消融：gold/检索证据/随机同域证据/同事件错误日期地点证据/去除核查网站及label词/仅实体不含日期地点/替换为相似但矛盾证据；测证据究竟改变哪些判定。
6. 如果低confidence样本检索后反而错误更多，或随机门控同样省钱，A的门控机制未成立；如果检索只在核查网站/答案泄漏存在时有益，不能声称模型完成原始事件证据核查。

### 选择指标与分割

- OOC二分类：Macro-F1 + 每类recall，门控重点为固定预算下错误率/风险及搜索率；校准用NLL/Brier，ECE仅辅报，样本少给bootstrap区间。
- abstain：risk–coverage/AURC；预算检索与拒答是两种行动，分别绘制曲线。
- 锚点context：日期/地点匹配或容差命中，必有证据URL与人工核验；不能用生成流畅度衡量真实性。
- 事件分组必须把同图、近重复图、相同caption、相同来源文章衍生样本整体归组；NC原始caption/image id disjoint不自动等于事件不重叠。
- 时间外测试保留AVerImaTeC原顺序，若新增VeriTaS需锁季度release和模型cutoff。无法确认模型训练数据时诚实标“污染风险未排除”，不能称真正完全未见。

### 对方案创新性最直接的反例

- COVE已经恢复context，提出“抽人/时/地锚点，再检索”本身不足为创新。
- MUSE已证明简单表层相似度强；增加不确定性头而只多一点分数可能只是更强classifier，必须证明在相同质量/相同成本的Pareto边界上有增量。
- RED-DOT、CMIE已有证据相关性筛选；“筛证据”与“在检索之前预测是否值得检索”可有区别，但要明确行动时点。
- 图片内容可能同一人物同一讲台，在两个世界中拍摄日期不同、像素完全相同；文本也同样可信。无图像来源信息时任何内部熵/JS都不保证分辨世界真值。这是信息可辨识性反例，不是新算法特性。

## 6. 未完成核验与应保留的不确定性

未下载任意全量图像/证据包；未申请VERITE/VeriTaS/Weibo21；未解包AVerImaTeC test_data.zip判断标签；未实际跑MUSE/CCN/COVE；未对全部论文DOI逐项调用Crossref/S2/撤稿API；未完成Weibo原始发布/所有许可核验。正文没有出现的方法缺点均明确限定为审计风险或未评议。以上不应在主报告被压缩成“所有数据公开且可立即复现”。
