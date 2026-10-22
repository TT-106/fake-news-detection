# B/C 视觉干预与注意力证据独立审计

核查日期与截止日：2026-10-19。仅检索和设计，未运行模型、未下载完整数据、未验证论文数值可复现。阅读本地 ARS 的 bibliography_agent 与 devils_advocate_agent 方法约束，借用来源分层、读范围声明、反例和可证伪设计；不声称调用原生插件。下文“我们的推论”不是作者结论。此文件是主审计的专题证据附件。

## 1. 检索范围与关键纠错

以 CVF、ACL Anthology、arXiv、作者 GitHub、Hugging Face 官方模型卡与 Transformers 固定版本源码为主。关键词组合：visual contrastive decoding / counterfactual image masking / object hallucination / attention entropy / visual grounding / hallucination detection / visual attention ratio / Qwen2.5-VL output_attentions。重点覆盖 2023—2026-10-19，补充 2018—2019 经典。页面搜索非数据库系统综述，没有完整可复核命中总量或 PRISMA 排除数；不宣称穷尽。部分 CVF HTML 返回 403，改读同作者 arXiv 全文或 CVF PDF；未使用浏览器自动化绕过限制。

核心纠错：

1. **VCD 是对象幻觉缓解的解码方法**。原图/加噪图 logits 对比已有明确先例，但不能据此推出新闻真假、原始拍摄时间地点或外部证据可靠性。
2. **空间注意力熵不是空白**：SAE（2026，作者自报 ACM MM 已接收）将语义分割聚合的注意力熵与视觉注意力总量结合；VISAGE（2026 预印本）使用空间注意力熵与头间共识。C 的“文本不确定性 + 视觉空间不确定性”若仅做两路融合，创新依据很弱。
3. **扰动敏感性不等于真实性**：VCD、AGLA、VHD、EnAR、ACFT 均在视觉变化或视觉依赖附近已有研究。B 可成为严谨诊断实验，但“反事实遮挡”本身不足以成为论文贡献。
4. **低注意力熵并不等于正确视觉接地**：集中盯住错误地点标志、注意力汇聚到伪影、文本 token 已吸收图像信息等都可反驳简单单调关系。2019 年关于 attention/explanation 的争论不能简单推广成“所有现代 VLM 注意力无用”；必须做本模型、本任务的预测增量及干预验证。

## 2. 文献核验记录

每条均列原始来源；未单独验证的 DOI 不补造。关联指本项目方法关联，不等于同任务基线。对只读摘要的记录，不评判作者方法局限。

### V01 VCD

- **准确书目**：Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, Lidong Bing. *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*. CVPR 2024，13872–13882；2023 年首发预印本。[CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.pdf)，[arXiv 2311.16922 全文](https://arxiv.org/html/2311.16922)，[官方代码](https://github.com/DAMO-NLP-SG/VCD)。注意作者是 **Chunyan Miao**，部分检索解析出现 Chunyuan，应以论文可视 PDF 为准。
- **任务/架构/数据**：图像+问题→回答；LLaVA-1.5、InstructBLIP、Qwen-VL；POPE（COCO/A-OKVQA/GQA）、MME、LLaVA-Bench。
- **真正测量/改变对象**：对相同前缀，融合原图和高斯扩散加噪图的下一 token logits：softmax((1+α)z原−αz扰动)，并有 plausibility constraint。它是解码纠偏，不是经过校准的新闻真实概率，也不独立给出无知/冲突分解。
- **实验结论与条件**：§4 报告所测模型与采样条件下对象/部分属性幻觉指标改善；不能外推新模型、新任务。作者 §4.2 指出位置推理提升较小；不是证明可恢复拍摄地点或时间。
- **公开性/关联**：代码公开 Apache-2.0；使用公开基准；B 最直接概念近邻，C 是方法背景。阅读：§3–4 和官方 README。

### V02 POPE

- **准确书目**：Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, Ji-Rong Wen. *Evaluating Object Hallucination in Large Vision-Language Models*. EMNLP 2023，292–305。DOI [10.18653/v1/2023.emnlp-main.20](https://aclanthology.org/2023.emnlp-main.20/)，[代码与构造数据](https://github.com/RUCAIBox/POPE)。
- **任务与测度**：图像+对象存在性 Yes/No 问题；随机、流行、共现对抗式负对象采样。衡量模型是否虚构不存在对象；不是图文错配或事件核查。
- **实验结论/条件**：论文观察训练视觉指令中的对象频率和共现与幻觉有关，提出比自由生成更稳定的 probing 评估。
- **公开/局限**：代码和问题记录可访问；图像按原始数据来源取得。本轮读正式摘要/元数据及 VCD 对其协议描述，论文自身方法局限未评估。我们的任务映射：适合 B/C 的机制阳性控制，不能替代新闻主测试。

### V03 CHAIR

- **书目**：Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, Kate Saenko. *Object Hallucination in Image Captioning*. EMNLP 2018，4035–4045。DOI [10.18653/v1/D18-1437](https://aclanthology.org/D18-1437/)，[论文 PDF](https://aclanthology.org/D18-1437.pdf)。
- **任务/指标/结论**：COCO 图像描述；CHAIRs 统计含幻觉对象的描述比例，CHAIRi 统计提及对象中的幻觉比例。论文观察常用描述相似性高不保证对象幻觉少。
- **关联/公开**：经典评估指标，非新闻检测器；COCO 公开；本轮未访问原始 CHAIR 代码仓库，因此不声称检验其可运行性。读摘要及指标定义，方法局限未评估。

### V04 OPERA

- **书目**：Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, Nenghai Yu. *OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation*. CVPR 2024，13418–13427。[正式页](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_OPERA_Alleviating_Hallucination_in_Multi-Modal_Large_Language_Models_via_Over-Trust_CVPR_2024_paper.html)，[代码](https://github.com/shikiw/OPERA)。
- **任务/架构/测量**：MLLM 图像描述和回答；在 beam decoding 中度量生成 token 对 summary tokens 的知识汇聚/过度信任模式，施加 logits 惩罚并回滚重分配；不是空间熵。
- **结论/关联**：作者报告多模型对象幻觉缓解；为 C 提供内部注意力结构已有先例。代码公开，沿用公开幻觉评估；本轮读摘要、方法介绍与仓库，具体数据规模和作者局限未逐项核实。

### V05 AGLA

- **书目**：Wenbin An, Feng Tian, Sicong Leng, Jiahao Nie, Haonan Lin, QianYing Wang, Ping Chen, Xiaoqin Zhang, Shijian Lu. *Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention*. CVPR 2025；[arXiv 2406.12718v3](https://arxiv.org/abs/2406.12718)，[全文](https://arxiv.org/html/2406.12718v3)，[代码](https://github.com/Lackel/AGLA)。
- **任务/数据/方法**：图像问答与描述，POPE/ROPE/MME/COCO-CHAIR；图像-提示匹配定位相关区域，构造突出局部的视图，与原图全局 logits 融合，无需训练目标 VLM。
- **测量与条件**：局部/全局视图辅助幻觉纠偏，非真假声明证据。作者报告所测模型上多类幻觉评估改善。
- **可核验局限**：附录 A1 作者承认受资源约束未测更大模型；仅图文，视频为未来工作。公开代码，使用公共基准；读方法与附录 A1。B 高关联，C 中关联。

### V06 VAR / SVAR

- **书目**：Zhangqi Jiang, Junkai Chen, Beier Zhu, Tingjin Luo, Yankun Shen, Xu Yang. *Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting and Mitigating Object Hallucinations via Attention Lens*. CVPR 2025；[CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_Devils_in_Middle_Layers_of_Large_Vision-Language_Models_Interpreting_Detecting_CVPR_2025_paper.pdf)，[全文](https://arxiv.org/html/2411.16724v3)，[代码](https://github.com/ZhangqiJiang07/middle_layers_indicating_hallucinations)。
- **任务/测量/数据**：COCO 描述的对象 token 幻觉；LLaVA-1.5 7B/13B、MiniGPT-4、Shikra。VAR 是输出位置向图像 token 分配的注意力总量，SVAR 在选定层和头聚合；另有头间信息整合干预。
- **结论/局限**：中间层视觉处理与幻觉有关。附录 A 作者明确：像 Shikra 跨层注意力一直很高时 SVAR 可能失效；层段选择带主观性。
- **关联**：C 必须比较的廉价注意力总量近邻；不能只比文本熵。代码公开，读§3–5与附录 A。

### V07 VADE

- **书目**：Vishnu Prabhakaran, Purav Aggarwal, Vinay Kumar Verma, Gokul Swamy, Anoop Saladi. *VADE: Visual Attention Guided Hallucination Detection and Elimination*. Findings of ACL 2025，14949–14965。DOI [10.18653/v1/2025.findings-acl.773](https://aclanthology.org/2025.findings-acl.773/)，[PDF](https://aclanthology.org/2025.findings-acl.773.pdf)。**作者拼写差异**：Anthology 元数据为 Swamy，PDF 首页为 Swami；保留差异，不静默纠正。
- **方法/任务/数据**：对 attention maps 建模的序列检测器，细粒度幻觉检测；M-HalDetect 与 MSCOCO。作者强调 grounding 对某些 token 是可选的、注意力高维噪声大、历史长度变化。
- **结论/公开/关联**：摘要报告跨四架构的检测和缓解结果；本轮 PDF 搜索未找到 GitHub URL，不据此断言无代码。仅摘要与首页核查，方法局限未评估。C 高关联，说明“读注意力检测幻觉”并非新颖。

### V08 SAE（非常接近 C）

- **书目**：Jiale Song, Jiaxin Luo, Xue-song Tang, Kuangrong Hao, Mingbo Zhao. *Segmentation-Based Attention Entropy: Detecting and Mitigating Object Hallucinations in Large Vision-Language Models*. [arXiv 2603.16558v2](https://arxiv.org/abs/2603.16558)，[全文](https://arxiv.org/html/2603.16558v2)。2026-08-26 作者注记 accepted ACM MM 2026；会议 2026-11 尚未到，当前应写“已接收，预印本全文”，不宣称已正式发表。全文列 DOI 10.1145/3767308.3835345，但本轮未访问 ACM 正式页，不将其升级为独立出版核验。
- **方法/数据**：Mask2Former 语义分割，将视觉注意力按语义类聚合再算归一化熵；可靠性为视觉注意力总量×(1−熵)。COCO 对象 token 检测、CHAIR、POPE、MMHal-Bench，LLaVA 并扩展 Qwen2-VL/InternVL2。
- **结论/条件**：作者报告检测和干预改善。§3.2 明确说明 raw patch entropy 会被对象面积/多个实例混淆，是 C 的直接反例来源。
- **公开性/局限**：所读摘要/全文未定位作者代码链接；不是确认未公开。我们的推论：分割错误会传递到熵，新闻的时间/身份事实不能由语义类别熵单独识别。阅读§3–4，后两项是未验证迁移风险。

### V09 VISAGE（空间熵 + 头间共识）

- **书目**：Vishal Narnaware, Animesh Gupta, Kevin Zhai, Zhenyi Wang, Mubarak Shah. *Seeing to Ground: Visual Attention for Hallucination-Resilient MDLLMs*. [arXiv 2603.25711v1](https://arxiv.org/abs/2603.25711)，[全文](https://arxiv.org/html/2603.25711v1)，2026 预印本，本轮未核实会议发表。
- **任务/测度**：多模态扩散语言模型的并行 masked decoding；空间 cross-attention entropy，加跨头分位数共识，联合文本概率重排承诺 token。测试 MMaDA 等扩散结构、HallusionBench/MMMU/POPE/MME。
- **结论/边界**：作者报告相关基准改善，§6.1 讨论过强惩罚会损害某些任务；Limitations 明确没有建模跨帧时间一致性。定理依赖估计误差有界假设，不是无条件证明注意力熵等于 grounding。
- **公开性/关联**：全文未定位方法 GitHub 链接，代码状态待核验。C 强概念近邻，不能直接当 Qwen 自回归模型的即插即用基线。阅读§4–7及附录 A。

### V10 ACFT / AHAF

- **书目**：Peiyang Xu, Xiaopei Zhu, Jun Zhu, Xiaolin Hu. *Beyond Language Priors: Diagnosing and Fixing Visual-Origin Hallucinations in Multimodal LLM*. [arXiv 2609.00231](https://arxiv.org/abs/2609.00231)，[HTML 全文](https://arxiv.org/html/2609.00231v1)标题末尾写 LLMs；2026-08-31 首发，作者注记 camera-ready / to appear ACM MM 2026；不是截至日已开会的论文。[代码](https://github.com/zxp555/ACFT_MM26)实际可访问（注意摘要有时显示截短 ACFT_MM）。
- **方法/对象/数据**：图文 embedding 对齐、Smooth Grad-CAM entropy 分析视觉源幻觉；最小对抗扰动翻转 hallucination 属性，组成训练对，再对比微调。COCO、POPE、MME及描述级评估，含 LLaVA、MiniGPT-4、Qwen2.5-VL。
- **条件/局限**：作者报告训练后改进；不能当冻结模型免费模块。本轮查摘要和方法/实验节，作者限制节未独立定位。我们的迁移推论：对抗翻转证明模型脆弱性，不能直接证明新闻真值识别。B/C 重要近期近邻。

### V11 EnAR

- **书目**：Yuxuan Liang, Fan Shi, Rui Zhu, Xu Li, Xiaolei Chen, Zhe Liu, Bin Li, Xiangyang Xue. *Envision, Attend, Then Respond: Counterfactual Hallucination Mitigation in Large Vision-Language Models*. CVPR 2026，18261–18272。[正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Envision_Attend_Then_Respond_Counterfactual_Hallucination_Mitigation_in_Large_Vision-Language_CVPR_2026_paper.html)。
- **任务/方法/结论**：图像与世界先验冲突的 counterfactual scenarios；扩散先验生成 prior-consistent counterpart，再定位反事实区域，原图与 padded 输入对比解码。VLMBias 和 POPE，作者报告幻觉减轻。
- **公开/局限/关联**：本轮仅正式摘要；代码、完整模型/数据许可和方法局限未评估。B 强概念关联；它的 counterfactual 含义是图像内容违背世界先验，不应与“旧图配新事件”混用。

### V12 HALP

- **书目**：Sai Akhil Kogilathota, Sripadha Vallabha E G, Luzhe Sun, Jiawei Zhou. *HALP: Detecting Hallucinations in Vision-Language Models without Generating a Single Token*. EACL 2026，6067–6085。DOI [10.18653/v1/2026.eacl-long.287](https://aclanthology.org/2026.eacl-long.287/)，[arXiv 全文](https://arxiv.org/html/2603.05465v1)，[代码](https://github.com/Zesearch/HALP)，[数据](https://huggingface.co/datasets/Zesearch/HALP-Bench)。
- **方法/数据/对象**：冻结 VLM 单次 prefill 的视觉特征、decoder 视觉位置或 query 位置 hidden states 上训练轻量 probe；混合 VQA 基准的 HALP-Bench，含 Qwen2.5-VL。测量将要回答出错/幻觉的风险，不是新闻真伪。
- **结论/局限**：有效层与模态随架构变化。作者 Limitations 承认 VQA 数据偏差、GPT-4 标签偏差、资源约束和更大模型未充分探索；不能承诺迁移到新闻。
- **关联**：C 应加入同等预算 hidden-state probe，避免给简单 embedding 信息换个熵名字。阅读摘要、§6和 Limitations，代码入口可访问，未复现。

### V13 经典解释争论，成对引用

- Sarthak Jain, Byron C. Wallace. *Attention is not Explanation*. NAACL 2019，3543–3556。DOI [10.18653/v1/N19-1357](https://aclanthology.org/N19-1357/)。NLP 模型实验指出注意力与梯度重要度可不一致，替换注意力可能保持输出。经典方法警示；仅摘要核对，不把结论扩写成所有 Transformer 的定理。
- Sarah Wiegreffe, Yuval Pinter. *Attention is not not Explanation*. EMNLP-IJCNLP 2019，11–20。DOI [10.18653/v1/D19-1002](https://aclanthology.org/D19-1002/)。提出 uniform attention、随机种子方差、冻结模型诊断和对抗训练等检查。RNN 任务，不能直接证明现代 VLM 注意力因果有效。摘要与引言已读；ACL 附件可访问，未验证其代码运行性。

### V14 VHD / VHR

- **书目**：Jinghan He, Kuan Zhu, Haiyun Guo, Junfeng Fang, Zhenglin Hua, Yuheng Jia, Ming Tang, Tat-Seng Chua, Jinqiao Wang. *Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence*. ACL 2025，3488–3501。DOI [10.18653/v1/2025.acl-long.175](https://aclanthology.org/2025.acl-long.175/)，[正式 PDF](https://aclanthology.org/2025.acl-long.175.pdf)，[代码](https://github.com/jinghan1he/VHR)。
- **任务/测量/数据**：VHD 测量 attention head 输出对视觉上下文的敏感性，据此增强 vision-aware heads（VHR）以缓解幻觉。仓库提供 COCO/POPE、LLaVA-Bench 以及 LLaVA-1.5/NeXT、InstructBLIP 的脚本入口。
- **结论/边界**：作者摘要报告无训练幻觉改善、额外延迟较低；并非新闻核查。公开代码可访问，本轮未运行。只读摘要、PDF 首页与官方 README，作者方法局限未评估。B/C 都有直接概念关联。

### V15 CausalLens

- **书目**：Junyang Ji, Qifan Liu, Wenming Yang, Zhihai He. *CausalLens: Sensitivity-Guided Multi-Head Causal Intervention for Hallucination Mitigation in Large Vision-Language Models*. CVPR 2026，4199–4209。[正式页](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html)，[PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.pdf)。
- **方法/测量**：将 head 贡献分为视觉、文本、系统提示路径，以敏感性选择可靠视觉 heads，调整中层 hidden states，并在 multi-head 融合后作投影对齐修正；单次前向、无训练。
- **任务/条件/公开性**：论文为幻觉缓解，摘要报告多 VLM、多幻觉基准改善；所读补充材料列 POPE、MME、MMHal、CHAIR、LLaVA-Bench。未核验代码发布。只读摘要、首页和补充协议，不评估方法局限。它进一步削弱“头敏感性/门控干预天然新颖”的论断，不能自动变成新闻分类基线。

## 3. B：反事实视觉扰动的可证伪版本

### 数学定义

固定模型、提示 q、候选标签集合 Y、文本 t，以及相同答案前缀。定义 p0(y)=pθ(y|t,I,q)，pR(y)=pθ(y|t,gR(I),q)，其中 gR 是对区域 R 的已登记干预。使用候选答案的真实条件 log-likelihood 后归一化；若多 token，锁定打分和长度处理。不同自由生成历史下的第 k 个 token 分布不直接比较。

敏感性 D_R=JS(p0,pR)=½ KL(p0||m)+½ KL(pR||m)，m=(p0+pR)/2。它首先是**模型输出对所选干预的敏感性**。若定义人工干预标识 Z∼Bernoulli(½)，Yout|Z=0∼p0、Yout|Z=1∼pR，才有 JS=I(Z;Yout)；这个互信息的对象是干预来源与模型输出，不是 I(新闻真值;视觉证据)。因此可以精确解释 JS 与 MI 的条件等价，但不得简称“新闻信息增益”。

反例：给无关角落加黑块触发模型分布外响应，输出变动很大但没有有效证据损失；真实旧图和对应新事件文本在视觉上完全兼容，遮挡也无法识别年份；删除真实证据会使本来正确的判断变得不确定，D_R 大仍不意味着原样本为假。

### 必要设计

| 干预 | 能支持的有限结论 | 必须匹配的对照 |
|---|---|---|
| 声明相关区域遮挡 | 输出依赖该区域的可见输入，尚非真值依赖 | 等面积、相似位置/形状随机区域；多随机种子；另一独立定位器或人工框 |
| 前景遮挡 | 依赖被删除的前景内容 | 等面积背景块、多个填充策略，控制对象/文字遮挡比例 |
| 背景遮挡 | 依赖背景线索，可能是合法地标也可能来源偏差 | 与前景面积匹配；人工标记地点线索；跨来源测试 |
| 随机遮挡 | 基础视觉破坏敏感性 | 面积和形状分层，不能只做单个随机块 |
| 模糊/均值填充/噪声 | 扰动操作的稳健性 | 固定分辨率与强度；灰色黑色块不单独作为唯一操作 |
| 全图噪声/JPEG/降采样 | 图像质量和分布变化 | 与局部干预的感知质量损失对齐；记录 CLIP 特征变化或 LPIPS 类质量代理 |
| 原图重复/提示改写 | 采样和提示噪声底线 | eval 模式、无 dropout；固定概率前向；多提示模板 |
| 换图/打乱图文 | 模型是否区分宏观图文关系 | 难负样本与来源/对象匹配，不仅明显不同图 |

首选效应量 S=D相关−E[D等面积随机]，同时报告原始 D、面积与强度。统计独立单位是新闻/原始图像或事件；一个图像的多个遮挡不能当作独立样本扩大 n。用事件聚类 bootstrap。遮挡可能改变任务的可回答性，应单独标注“干预后仍可判断/应拒答”，不能沿用原标签机械评估所有干预图。

### 判断 B 是否服务新闻真实性

先检验 D 或 S 是否只识别图像质量、显眼对象和输出长度；再检验其是否在冻结图文 embedding、普通检测概率、预测熵之后仍改善未见事件上的错误风险排序。要有“模型依赖图像且判断错误”“模型几乎不依赖图像却凭文本正确”两个反例组。核查收益另通过检索后错误下降或风险覆盖曲线评价；不能从遮挡导致预测变化推断检索有用。

最小机制实验可用有精确框的对象存在性小集作阳性控制，再在人工区分的可见图文矛盾/视觉不可判定时空错位/一致样本上验证迁移。如果仅阳性控制成立，结论应是视觉依赖探针可工作，但新闻用途未证实。B 的价值可以是揭示模型依赖与真实性脱钩的负结果。

## 4. C：空间注意力熵与文本不确定性的审稿评估

对第 l 层、第 h 个 query head 和固定预测位置 a，取其到图像 token 集合 V 的注意力 A。定义 M=Σi∈V Aai；若 M>ε，定义 ri=Aai/M 和 Hsp=−Σri log ri/log|V|。若 M≈0，记为缺失/无视觉质量，不把归一化后某个低熵数当作可靠 grounding。熵与注意力总量必须同时保存。视觉 patch 数、语义对象数量、空间尺寸影响 Hsp，归一化 log|V| 只消除最大值差异，不建立模型间语义可比性。

两种操作不等价：先跨头平均再算熵，和先算各头熵再平均。两个头分别盯不同对象可使各自低熵但平均高熵。必须预注册聚合规则、层选择与所用 query 位置，禁止在测试集按标签挑头。采用不同分辨率、不同提示和至少两个模型的敏感性检查；数值阈值按模型独立校准。

最强反例：正确判断涉及多个对象/广域火灾，合理关注范围大而高熵；错误判断紧盯一个无关 logo，低熵；图像信息已经传播到文本隐状态，直接图像注意力低但模型仍利用视觉。C 只能从预测增量开始，不能把集中度命名为“真实性置信度”。

最低限度增量比较：同一轻量分类器、训练样本、验证搜索预算和校准方式，依次使用 (1) 图文 embedding+相似度，(2) 加分类熵/MSP，(3) 加 VAR/SVAR，(4) 加 Hsp，(5) 加候选双门控。比较线性融合/浅层 MLP 与双门控，防止收益只来自额外容量。再加入 HALP 式 hidden probe、SAE（若能复现）或语义分段熵，说明超越最接近方法的地方。

伪特征控制：样本间随机置换 Hsp；均匀/随机 attention；保持每图注意力值多重集而置换空间位置（**熵完全不变**，因此若模型只用熵，本来就无法知道关注的是正确区域）；加入亮度、边缘密度、面积、视觉 token 数、输出长度等低成本特征；按来源/事件划分。若打乱 attention 空间位置不改变检测，这不是 bug，而是证明纯熵没有定位正确性信息，必须增加框/对象/声明对齐。

负面结果：仅 IID 提升、未见事件失效；换模型改变符号；加入 embedding 或 VAR 后增益消失；遮挡伪影即可复现同等增益；高置信错误仍低熵。任一出现都限制或推翻“通用可靠视觉接地指标”说法。

## 5. 开源 VLM 可提取性与资源审计

来源：[Qwen2.5-VL 官方 Transformers 文档](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)，[固定 v4.51.3 源码](https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)，[3B 配置](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/config.json)，[7B 配置](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/config.json)，[3B 模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)，[7B 模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)。当前 Qwen2.5-VL GitHub 仓库链接已重定向 Qwen3-VL，不能把新仓库默认分支当作旧版本实现证据。

- 对冻结开放权重模型可请求 forward logits 和 output_hidden_states；有接口不代表已经在本机验证返回形状。生成 scores 可受温度、processor、top-k 等加工，估计基础不确定性优先原始 forward logits；记录到底使用哪种。
- 固定源码的 decoder eager attention 返回权重；SDPA 在 output_attentions=True 时警告并回退手工实现，跨版本行为可能变化。提取采用显式 attn_implementation="eager" 并断言张量非 None；不能假定 FlashAttention 会给完整注意力。
- decoder 文本到视觉位置的 self-attention 与 vision encoder 内部 attention 不同；常规 output_attentions 不能未经核查就说包含 ViT 所有内部图。Qwen 是视觉 token 插入 decoder 的融合，不是每个模型都有独立 cross-attention 层。
- 3B 配置：36 层、16 query heads、2 KV heads；7B：28 层、28 query heads、4 KV heads；均视觉 patch 14 与 spatial_merge 2。GQA 不能把 KV head 数当作空间注意力 query head 数。依 input_ids 中 image_token_id 和 image_grid_thw 映射 merge 后视觉位置；去掉 vision_start/end 与 padding，不硬编码 576 patches。
- prefill 通常含完整输入位置的注意力；cached decode 的 query 可能只有新位置，key 含历史。预测下一个 token 的 logits 对应当前最后位置；把生成 token 的输入后注意力误当生成它之前注意力会偏移一位。给 ground-truth verdict 后提特征会标签泄漏。可在固定、与真值无关的回答前缀末位提特征，或做明确的自生成后验检测。

以下是工程估算而非实际运行证据：以标称参数 P 的 BF16 权重约 2P 字节，3B 约 6GB、7B 约 14GB，仅权重下界；视觉模块、KV cache、图像分辨率、attention/logits、CUDA 工作区另计。RTX4060 的本机显存本轮未核验，不能因型号直接保证 3B BF16 全 attention 可跑。4090 24GB 优先 3B BF16 小分辨率、batch=1 单前向；7B 需测峰值后决定 BF16/4bit，量化作为独立实验因素。

完整 decoder attention 存储约 L·B·H·S²·b；假设 3B 的 36 层、16 query heads、S=2048、BF16、batch1，仅返回注意力约 4.5GiB，float32 则约 9GiB。此为张量量级计算，不含其他显存。只在短序列、少层、少 query 行抽取并立即 CPU 聚合，或定制 hook 避免保存全矩阵。不能因为冻结就认为显存便宜。记录模型 revision、Transformers/PyTorch、量化、图像 token 预算、后端和显存峰值。

## 6. 对抗性结论与后续门槛

**B：建议作为诊断/控制模块，暂不独立押注创新。** 长处是能提出对“模型看没看图”的可操作反例；致命外推是把对输入干预的敏感性当事实正确性。只有在随机与等面积对照后，信号仍对未见事件的核查决策提供稳定增量，才继续升级。

**C：作为低成本探索，暂不作为主创新。** SAE/VISAGE/VAR/VADE/HALP 已覆盖相当大的概念邻域，注意力抽取还增加实现和显存成本。若简单融合已等效、跨事件不稳或与容量/图像复杂度控制后增益消失，应终止“双门控即创新”的叙事。

可预注册的 toy 门槛应由主审计统一指定，不能以本附件制造已通过结果。建议至少要求：(i) 两种填充与多随机控制下方向一致；(ii) 按事件聚类置信区间支持增量、达到事先定义的最小实际效应；(iii) 增量超过相同成本的简单 embedding/熵或 hidden probe；(iv) 无证据样本不被强制当作“冲突”；(v) 所有额外前向与检索成本进入预算。样本很少时只作 go/no-go 机制判断，不把不显著误称等效，也不从 toy 推断最终性能。
