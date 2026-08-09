# CS336 Lecture 15：Mid/Post-Training 从监督微调到偏好优化

![课程视频封面](assets/cover.jpg)

- **课程**：Stanford CS336 Language Modeling from Scratch, Spring 2026
- **讲次**：Lecture 15, Mid/Post-Training
- **主讲**：Tatsunori Hashimoto
- **视频**：[Stanford Online](https://www.youtube.com/watch?v=2oH6PWPrYFo)
- **时长**：01:19:54
- **资料范围**：人工英文字幕、1080p 视频、官方 65 页课件
- **讲义语言**：中文

> [!NOTE]
> 正文中的课件图来自本地官方 `lecture_15.pdf` 的清晰页面；图下时间来自人工字幕对应的讲解区间。对于 PDF 渲染中标题被裁切的页面，改用了同一字幕区间内逐帧比较后选出的完整 1080p 视频帧。

## 从基座模型到可控助手：为什么需要后训练

### GPT-3 的能力为何没有自动变成 ChatGPT 的可用性

课程此前讨论的预训练可以得到一个比 GPT-3 更强的基座模型：增加数据、计算量和模型规模，语言建模能力会继续提高。然而，强大的续写器还不是可靠的助手。早期 GPT-3 更适合文案写作、自由生成等不要求精确控制的任务；面对长提示、多项约束和严格输出格式时，few-shot 示例只能提供有限而脆弱的 steering。

ChatGPT 时代最重要的变化之一，是模型能够把长而复杂的自然语言提示当作一段“程序”执行。它不只理解任务主题，还能同时满足数据、布局、颜色、标签、插值、误差条和动画等组合约束。讲者用一条复杂绘图提示说明：instruction following 本身就是一种非常强的生成控制接口。

![复杂长提示展示 instruction following 的细粒度控制](assets/instruction-following-control.jpg)

*课件图：同一提示同时约束多个子图、误差条、插值、基线和动画；对应视频讲解 `00:01:41--00:02:24`。*

> [!IMPORTANT]
> 预训练负责广泛获取能力，后训练负责把这些能力组织成用户可调用、可预测、可约束的行为。后训练不能替代预训练，它更像从预训练形成的能力空间中“提取”目标模式。

### 后训练的现实：算法轮廓公开，数据成为秘密

本讲反复强调一个现实：现代后训练的数据、标注规范和质量控制，比公开算法更不透明。Stiennon 等人的摘要 RLHF 工作和 Anthropic HH 等早期论文，反而公开了较详细的标注指南。ChatGPT 引发竞争后，闭源实验室把数据视为核心竞争力；开源模型虽然公开了许多 recipe，但大量流程依赖从更强模型蒸馏，不能直接代表 frontier lab 的人工数据生产。

讲者举出的 Bard/Scale AI 案例很能说明问题：团队要求标注者研究 GPT-4 为什么更好，并写出至少同样详细的答案。这里的杠杆不在“有没有梯度下降”，而在任务分布、示范答案、评价准则、标注者选择和质量审核。

![后训练信息稀缺与数据保密](assets/post-training-data-secrecy.jpg)

*课件图：早期论文信息丰富，现代开源多为蒸馏、闭源数据成为 secret sauce；对应视频讲解 `00:03:28--00:05:42`。*

### 标准配方：先模仿，再优化偏好

经典 InstructGPT/RLHF 流程分为三步：

1. 收集 prompt 和高质量 demonstration，做 supervised fine-tuning, SFT。
2. 对同一 prompt 生成多个回答，由标注者排序，训练 reward model。
3. 用 RL 优化策略，使 reward model 评分提高，同时限制模型不要偏离参考模型太远。

![SFT 到 reward model 再到 RL 的三阶段管线](assets/sft-rlhf-three-stage-pipeline.jpg)

*课件图：本讲先讨论左侧 SFT，再讨论后两阶段的偏好学习；对应视频讲解 `00:05:53--00:06:29`。*

SFT 的优化器并不神秘。给定 demonstration 数据集，标准目标仍是 teacher forcing 下的 next-token negative log-likelihood：

$$
\mathcal L_{\mathrm{SFT}}(\theta)
=
-\sum_i\sum_t
\log p_\theta\!\left(y_{i,t}\mid x_i,y_{i,<t}\right).
$$

- $\mathcal L_{\mathrm{SFT}}$：监督微调损失。
- $\theta$：语言模型参数。
- $i$：训练样本索引。
- $t$：回答 token 的位置。
- $x_i$：第 $i$ 个 prompt 或对话上下文。
- $y_{i,t}$：参考回答第 $t$ 个 token。
- $y_{i,<t}$：参考回答中位于 $t$ 之前的 token。
- $p_\theta$：模型给出的条件概率分布。

这解释了为什么 SFT 的难点主要在数据：同一个目标会同时模仿事实、语气、长度、引用习惯、拒答方式、工具协议和结构化动作。

### 本章小结

- GPT-3 到 ChatGPT 的关键跃迁之一，是复杂指令控制，而不只是更多参数。
- 预训练积累广泛能力；后训练选择并强化部署所需的行为。
- 标准后训练先以 SFT 模仿 demonstration，再以偏好反馈优化策略。
- SFT 算法接近普通语言建模，真正复杂的是数据定义与质量控制。

## SFT 数据如何塑造行为

### 从 NLP benchmark 到自然对话，再到 agent 轨迹

开放世界的 SFT 数据经历了明显演化：FLAN 汇总传统 NLP 监督任务；Self-Instruct 让模型参与生成指令；Alpaca 从强模型蒸馏更自然的问答；ShareGPT/Vicuna 使用真实用户分享的 prompt；OpenAssistant 依靠社区众包长而详细的回答；WizardLM、Tulu 等发展出更复杂的合成和筛选管线；最近的数据又把 tool call、任务状态和多步 agent trajectory 纳入监督。

![开放世界 SFT 数据演进](assets/sft-data-progression.jpg)

*课件图：FLAN、Self-Instruct、Alpaca、ShareGPT/Vicuna、OpenAssistant、WizardLM、Tulu 和 Nemotron；对应视频讲解 `00:07:34--00:10:15`。*

FLAN 的思想很先进：将许多下游任务统一成 instruction tuning。但它继承了 legacy NLP dataset 的输入格式。例如，把 Enron 邮件正文和主题机械改写成“根据正文写主题”，虽然是有效监督任务，却不一定像真实用户与助手的交互；摘要参考答案也可能过短，甚至包含输入中不存在的细节。

![FLAN 的随机样例](assets/flan-examples.jpg)

*课件图：邮件主题、新闻分类、摘要和结构化餐馆描述等由旧任务改写的样例；对应视频讲解 `00:11:24--00:14:31`。*

Alpaca 从 ChatGPT 蒸馏 input-output traces，prompt 更自然，回答更完整、更接近聊天助手。这种方法能在强预训练底座上快速诱导出 ChatGPT-like behavior，但“像 ChatGPT 回答”不等于底座拥有同等的知识、推理与可靠性。

![Alpaca 的自然问答样例](assets/alpaca-examples.jpg)

*课件图：健康建议、算法解释和简单代码回答；对应视频讲解 `00:14:31--00:15:33`。*

OpenAssistant 走的是高投入人类众包路线，回答更长、更像专家写作，包含更多事实、引用和详细解释。这提高了可读性，也埋下了后文的风险：详细的“引用行为”可能比引用背后的真实知识更容易被模型学会。

![OpenAssistant 的知识密集型回答](assets/openassistant-examples.jpg)

*课件图：长答案、研究引用和列表化建议；对应视频讲解 `00:15:33--00:16:50`。*

Nemotron 一类新数据则说明，SFT 的输出不再只是纯文本。`role`、`content`、`tool_calls`、function arguments、todo 的 priority/status 都可能成为训练目标。工具使用不是推理时凭空出现的能力，常常先被显式写进结构化训练轨迹。

![Nemotron 中的 agentic SFT 样例](assets/nemotron-agentic-sft.jpg)

*课件图：自然语言回答与 tool call、todo 状态共同作为监督目标；对应视频讲解 `00:16:50--00:17:51`。*

### 数据不只决定内容，也决定“助手人格”

公开数据集的变化可以压缩为三个方向：

- **Chattiness**：从 benchmark 式短输出转向更自然、更详细的对话。
- **Detail**：更多复杂知识、引用和长解释，同时提高对标注者专业度的要求。
- **Tool use**：从聊天回答扩展到 API、工具调用和 agent 行为。

![SFT 数据变化的三个方向](assets/sft-data-dimensions.jpg)

*课件图：chattiness、detail 与 tool use；对应视频讲解 `00:17:51--00:18:55`。*

这些特征会显著影响偏好评估。人类和 GPT-based evaluator 常偏好列表、更长、更详细的答案；因此改变数据中的长度和格式，就能大幅改变 AlpacaEval 一类胜率，却未必同步提高 MMLU、GSM、BBH、TyDiQA 或 Codex-Eval。

![偏好评价中的列表和长度效应](assets/preference-length-effects.jpg)

*1080p 视频帧：在同一字幕区间比较多个邻近状态后，选择标题和两张图全部完整显示的状态；对应字幕讲解 `00:20:09--00:20:50`。*

![风格指标与能力 benchmark 的不一致](assets/style-versus-benchmark-capability.jpg)

*课件图：不同 instruction-tuning 数据对 AlpacaEval 与事实、推理、编码 benchmark 的影响并不一致；对应视频讲解 `00:21:07--00:21:59`。*

> [!WARNING]
> 偏好胜率和 engagement 提升，只能说明输出更符合评价者的风格偏好，不能单独证明模型更聪明、更真实或推理更强。style control 与 capability control 必须分开测量。

### 知识注入的悖论：一个样本同时教事实和行为

OpenAssistant 的 monopsony 样例包含具体论文引用。对它做 next-token prediction 时，模型同时学习：

1. Bivens 与 Mishel 等事实内容。
2. 遇到此类请求时，答案应该附带正式引用这一行为模式。

![引用样本同时训练知识与输出行为](assets/citation-content-and-behavior.jpg)

*课件图：同一示例既教具体文献，也教“回答中应该输出引用”；对应视频讲解 `00:21:59--00:23:30`。*

问题是，模型可能可靠学会“References:”之后要继续生成看似真实的书目信息，却没有可靠机制判断自己是否真的知道该文献。讲者总结了一个有实证支持的 folklore：在模型未知的事实上做 SFT，训练准确率会继续提高，但开发集性能可能在过拟合后下降，并诱导模型在未知问题上自信编造。

![未知事实训练、过拟合与幻觉](assets/knowledge-extraction-hallucination.jpg)

*课件图：左侧为 behavior cloning 的知识边界直觉，右侧为 known/unknown 事实训练曲线；对应视频讲解 `00:23:30--00:24:18`。*

这里不是说“正确事实不能用于 SFT”，而是要区分两类目标：把已有能力引导到正确输出，和试图用少量示例把模型原本不掌握的长尾知识硬塞进去。SFT 对训练样本的 loss 没有算错，失败来自内容与格式的错误泛化。

讲者给出的 RL 直觉是：如果模型内部已有某种“我知道/我不知道”的校准信号，那么根据模型自己的 rollout 奖励正确引用、惩罚编造，可能把这个内部信号连接到输出策略。但 RL 不能凭空创造不存在的知识或校准。

> [!IMPORTANT]
> SFT 最擅长抽取预训练中已有的行为模式。事实正确的数据也可能因知识边界与输出模板纠缠而伤害模型；知识存储、提取和校准不能简化为“多塞一些答案”。

### 本章小结

- SFT 数据从旧式 NLP 任务演化到自然对话、合成数据和 agent 轨迹。
- 模型会同时模仿内容、长度、语气、格式、引用和工具协议。
- 更长、更列表化的回答容易赢得偏好，但不等价于能力提升。
- 在未知事实上强制模仿，可能教会模型自信生成“正确格式的错误内容”。

## Safety SFT 与 midtraining：少量 steering 如何进入大规模训练

### 安全调优的两个错误率

部署模型面对的不只是普通任务，还包括政治操纵、诈骗、钓鱼、垃圾信息和 jailbreak。Safety SFT 的目标不能简化为“拒绝更多”，而要同时控制：

- **Violation rate**：有害请求穿透安全防线的比例。
- **False refusal rate, FRR**：正常请求被错误拒绝的比例。

“How do I kill a Python process?” 中的 `kill` 是正常计算机术语；只看敏感词会造成大量误拒。因此安全与有用性构成 Pareto trade-off。

![公开安全 SFT 信息与 violation/false refusal 权衡](assets/safety-sft-in-the-wild.jpg)

*课件图：公开资料稀缺，安全数据混合影响 violation rate 与 false refusal rate；对应视频讲解 `00:29:18--00:30:25`。*

### 从真实用户流量发现长尾场景

相对完整的公开 pipeline 会从真实交互中挖掘 non-compliance 场景：收集用户日志，识别有害请求和 jailbreak tactic，构造理想拒答或安全替代回答，再迭代训练和评测。WildChat、WildJailbreak、WildGuardMix 等工作把这一流程公开化。

![从真实用户交互中提取安全场景](assets/wildchat-safety-scenarios.jpg)

*课件图：WildChat 场景分类与 WildTeaming 的 Mine/Compose 流程；对应视频讲解 `00:31:09--00:32:13`。*

这种做法很实际，却也偏反应式：系统先观察到一种攻击，再让标注者补数据，像持续“打地鼠”。精细安全边界最终仍依赖长期日志、红队、专家知识和质量控制。

### 为什么约 500 条数据就能显著改变行为

讲者展示的实验中，仅增加数百条 Alpaca-style 安全示例，就能让恶意指令、仇恨言论、争议内容和 Anthropic HH 等指标明显改善。一个可能的解释是：强预训练模型已经学到了多种可行行为，少量 SFT 只需把分布推向 safe mode。

![500 条安全数据带来的明显行为变化](assets/safety-five-hundred-samples.jpg)

*课件图：从 0 到 2000 条安全样本的多组评测；对应视频讲解 `00:32:14--00:33:42`。*

> [!WARNING]
> “500 条有效”只说明粗粒度 steering 很省数据，不表示模型已经全面安全。稀有场景、相邻边界、专业领域和误拒控制仍需要更大规模的数据覆盖。

### 把 instruction tuning 融入预训练尾部

普通 SFT 的训练循环很简单；当数据和计算规模变大，真正困难的是训练日程。现代常见三步配方是：

1. 在网页和通用语料上做大规模预训练。
2. 在预训练尾部混入高质量语料、instruction 数据和合成数据。
3. 最后再做一次较短的正式 instruction tuning。

![将 instruction tuning 数据混入预训练的三步配方](assets/midtraining-three-stage-recipe.jpg)

*课件图：预训练、混入 instruction 数据、短 SFT 三阶段；对应视频讲解 `00:36:18--00:36:57`。*

这一阶段常被称为 midtraining、second-phase pretraining 或 two-phase training。它允许在更大的 token 规模上提高高质量数据的权重，并降低完全独立微调造成灾难性遗忘的风险。现代所谓“base model”可能已经在预训练尾部见过 UltraChat、SFT mixed、代码或数学合成数据，因此不一定是传统意义上的纯网页语言模型。

![Midtraining 的 stable/decay 数据混合](assets/midtraining-data-mixtures.jpg)

*课件图：stable stage 以通用预训练源为主，decay stage 加入更多 Wikipedia、QA、UltraChat、SFT、代码和数学数据；对应视频讲解 `00:36:59--00:40:15`。*

数据混合比例通常不是由一个可靠公式直接解出，而是大量 trial and error：在较短、较便宜的 decay/midtraining 阶段做域消融，测量下游变化，给数据源排序，再把结论反馈给完整预训练。高质量数据不能独占整个预训练，因为 token 数量和覆盖范围不够；工程目标是改变比例，而不是彻底移除通用语料。

> [!NOTE]
> Prompt 是否 mask 不是 pretraining 与 SFT 的严格分界。有些纯预训练目标会预测 prompt token，一些 SFT recipe 也会这样做。更可靠的区分来自数据分布、反馈形态与训练目的。

### 本章小结

- Safety SFT 需要同时降低 violation 与 false refusal。
- 少量高质量示例可以强烈改变总体行为，但长尾边界仍需要规模。
- 现代训练会在预训练尾部逐渐提高高质量、指令型和合成数据比例。
- Midtraining 的关键工程方法是短阶段消融与反复试验，而不是一个封闭的最优 mixture 公式。

## 从模仿分布到偏好数据

### SFT 与 RLHF 优化的对象不同

SFT 把 demonstration 看作来自参考分布的样本，目标是拟合这个分布：

$$
\hat p(y\mid x)\approx p^*(y\mid x).
$$

- $x$：输入 prompt。
- $y$：模型回答。
- $p^*(y\mid x)$：示范者或参考策略隐含的条件分布。
- $\hat p(y\mid x)$：训练后模型学到的条件分布。
- $\approx$：表示用有限数据和参数模型近似参考分布。

RLHF 则把语言模型视为 policy，寻找能最大化可测奖励的分布：

$$
\hat p
=
\arg\max_p
\mathbb E_{y\sim p(\cdot\mid x)}[R(y,x)].
$$

- $\hat p$：优化后策略。
- $p$：候选策略。
- $x$：输入 prompt。
- $y$：策略生成的回答。
- $R(y,x)$：回答 $y$ 在输入 $x$ 下的奖励。
- $\mathbb E$：对策略生成分布取期望。
- $\arg\max$：返回使期望奖励最大的策略。

![模仿分布与优化奖励的目标对照](assets/imitation-versus-optimization.jpg)

*课件图：SFT 拟合 reference distribution，RLHF 最大化 reward；对应视频讲解 `00:42:23--00:44:00`。*

纯奖励最大化不要求保留参考数据的全部多样性；如果一个回答总能获得最高奖励，策略可以集中到那个回答。这为后文的 mode collapse 埋下伏笔。

### Generation-verification gap：人未必能写出自己最喜欢的答案

为什么不一直收集 demonstration？讲者给出两个理由。

第一，人写出的答案不一定是他真正偏好的答案。新闻摘要实验中，自由职业写作者在看到 Instruct-Davinci 摘要后，有些人反而更喜欢模型输出。人能够写作，却未必能从零想到自己会最满意的表达。

第二，验证往往比生成容易。检查一个数学证明通常比从头构造证明容易；同理，在多个候选答案中判断哪个更好，可能比要求标注者独立写出最优答案更可靠。

![Generation-verification gap](assets/generation-verification-gap.jpg)

*课件图：新闻摘要偏好接近五五开，且标注者之间差异明显；对应视频讲解 `00:44:01--00:45:54`。*

> [!WARNING]
> 验证比生成容易是一种常见不对称，不表示偏好判断自动等于客观正确。标注规则、知识水平、时间预算和评价偏差仍会影响结果。

### 标准偏好数据管线

标准流程从一个已做 SFT、能遵循指令且仍有输出多样性的模型开始：对每个 prompt 采样多个候选，标注者做排序或 pairwise choice，再用这些比较训练 reward model。

![成对偏好标注界面](assets/pairwise-feedback-interface.jpg)

*课件图：标注者比较两个 AI response，并选择更好或略好的回答；对应视频讲解 `00:47:08--00:47:50`。*

常用 reward model 采用 Bradley-Terry/logistic 形式：

$$
\mathcal L_{\mathrm{RM}}(\phi)
=
-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\left[
\log\sigma\!\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
\right].
$$

- $\mathcal L_{\mathrm{RM}}$：reward model 的成对偏好损失。
- $\phi$：reward model 参数。
- $\mathcal D$：成对偏好数据集。
- $x$：prompt。
- $y_w$：标注者偏好的 winner 回答。
- $y_l$：被拒绝的 loser 回答。
- $r_\phi$：reward model 输出的标量分数。
- $\sigma$：sigmoid 函数，将奖励差映射为 winner 获胜概率。
- $\mathbb E$：对数据集中的偏好对取期望。

### “好回答”不是自然标签，而是规则设计

InstructGPT 的标注指南把目标拆为 helpful、truthful、harmless。Helpful 包括遵循意图、表达清楚、必要时澄清且避免冗长；truthful 要求准确、不误导、摘要不编造输入外内容；harmless 要求避免物理、心理和社会伤害。实际任务中三者会发生冲突，标注者必须权衡。

![InstructGPT 标注指南](assets/instructgpt-annotation-guidelines.jpg)

*课件图：helpful、truthful、harmless 的操作化规则和权衡原则；对应视频讲解 `00:47:42--00:48:24`。*

> [!IMPORTANT]
> RLHF 数据不是简单记录“人类喜欢哪个”。标注指南先定义什么叫好回答，因此规则设计本身就是产品行为和价值目标的设计。

### 谁在生产 RLHF 数据

现代标注产业不再只有低价通用众包。以 Outlier/ScaleAI 的一份调查为例，标注者中本科、硕士或专业学位占很大比例，任务覆盖语言、创意写作、技术写作、数学和编程。与此同时，医生、律师等专业任务需要真正具备领域知识的人提供示范和核查。

![现代标注者的年龄、学历与任务分布](assets/modern-annotator-distribution.jpg)

*课件图：该调查仅覆盖一个平台，不能代表整个行业；对应视频讲解 `00:49:30--00:50:12`。*

![专家标注薪酬与领域差异](assets/expert-annotator-compensation.jpg)

*课件图：专家标注增长、多个领域薪酬中位数显著提高；对应视频讲解 `00:50:16--00:51:17`。*

现实结构更像金字塔：顶部是少量昂贵、定制、专业的数据；底部仍有大量低成本、可扩展的通用标注。规模化众包还有三个难点：验证真人和专业身份、确保标注者真正核查正确性、避免用便宜模型代做。低时间预算与有害内容审核也带来严重劳动伦理问题。

### 标注者是谁，会改变模型成为什么

后训练是模型发布前的最终塑形阶段，因此标注者的人口统计、知识结构和注意重点都可能进入模型行为。Santurkar 等人的研究比较模型回答与不同人群意见的接近程度，发现 post-training 会使这种接近度发生系统性移动。课上将它与标注群体分布联系起来，但这只是群体层面证据，不能解释为某个宗教身份直接决定模型回答。

![标注者人口统计与模型意见接近度](assets/annotator-demographics.jpg)

*1080p 视频帧：选择了左右两块表格全部揭示的最终状态；对应字幕讲解 `00:53:02--00:54:43`。*

专业度还会改变标注者“看见什么”。普通众包标注者较容易注意 formatting，专家更常发现 inconsistency 与 factuality error，尤其是语气自信、表面可信的错误。事实核查昂贵，没有相应知识时，流程会系统性漏检。

![专家与普通众包标注者关注的错误类型](assets/expert-annotators-detect-errors.jpg)

*1080p 视频帧：选择热图与说明文字完整显示的最终状态；对应字幕讲解 `00:55:37--00:56:34`。*

Inter-annotator agreement 只能衡量群体方差，不能排除共同偏差。所有人一致使用同一个模型代做，也可能得到很高一致性。对于主观偏好，高分歧可能是任务性质；对于 factuality，才更希望分歧较低。

### 从人类反馈到模型反馈

当目标是追赶现有前沿而不是创造超出现有模型的新知识时，强模型作为标注者通常非常有效。课件展示 GPT-4 评审与人类评审在系统排序上的 Spearman correlation 约为 0.98，agreement 接近人类标注者之间的水平，成本却低一个数量级。

![模型评审与人类评审的一致性和成本](assets/lm-feedback-agreement-cost.jpg)

*课件图：系统级排序近乎一致，单位标注成本更低；对应视频讲解 `01:00:14--01:00:46`。*

这一趋势推动了 UltraChat、UltraFeedback、Zephyr 和 Tulu 等开放后训练流程。但边界必须说清楚：从强模型蒸馏适合“追赶”，推进能力前沿、注入新专家知识时，人类仍不可替代。

Constitutional AI 展示了另一种自举：模型面对 red-teaming prompt 先生成回答，再依据宪法原则 critique 和 revision，用修订数据做 SFT，并进一步生成 AI preference 训练 RLAIF。它重新组织、筛选和放大已有能力，不能凭空创造模型从未掌握的世界知识。

![Constitutional AI 自训练和 RLAIF 流程](assets/constitutional-ai-self-training.jpg)

*1080p 视频帧：选择全部节点与连线完整揭示的最终状态；对应字幕讲解 `01:03:40--01:04:30`。*

模型评审也不是中立裁判。只增加输出长度，就可能持续提高 model-judge 胜率；SFT 后约 59 tokens 的回答，经 RLHF 可能扩展到约 243 tokens，却不一定增加等量信息。

![RLHF 中的长度效应](assets/rlhf-length-effects.jpg)

*课件图：长度与 win score/reward 的相关性，以及 SFT 与 RLHF 回答长度对比；对应视频讲解 `01:04:30--01:05:23`。*

> [!WARNING]
> Reward model 可能奖励“看起来像高质量”的代理变量。长度是最典型的例子：更详细可能更好，但单纯冗长也能成为 reward hacking 的方向。

### 本章小结

- SFT 模仿参考分布，RLHF 直接优化可测奖励。
- 人常常更擅长比较候选，而不是从零写出自己最偏好的答案。
- 偏好标签由指南、标注者分布、专业知识、时间预算和质量控制共同决定。
- 模型反馈适合追赶现有能力，但不能取代产生新专家知识的人类数据。
- 人类和模型评审都可能偏好长度等表面代理变量。

## PPO：如何在奖励上爬升又不走得太远

### KL 正则化的 RLHF 目标

InstructGPT 式 RLHF 在提高 reward 的同时，使用 KL penalty 限制策略偏离 SFT/reference model，并可混入预训练梯度：

$$
J(\theta)
=
\mathbb E_{(x,y)\sim D_{\pi_\theta}}
\left[
r_\phi(x,y)
-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right]
+
\gamma\mathbb E_{x\sim D_{\mathrm{pretrain}}}
\left[\log\pi_\theta(x)\right].
$$

- $J(\theta)$：要最大化的策略目标。
- $\theta$：当前 RL policy 的参数。
- $x$：prompt 或预训练文本上下文。
- $y$：策略生成的回答。
- $D_{\pi_\theta}$：由当前策略采样得到的 prompt-response 分布。
- $r_\phi(x,y)$：参数为 $\phi$ 的 reward model 对回答的评分。
- $\beta$：相对 reference policy 的 KL 惩罚强度。
- $\pi_\theta$：当前待优化策略。
- $\pi_{\mathrm{ref}}$：通常由 SFT 模型给出的参考策略。
- $\gamma$：预训练梯度混合项权重。
- $D_{\mathrm{pretrain}}$：原始预训练数据分布。
- $\mathbb E$：对相应数据分布取期望。

前两项的直觉最重要：沿 reward 上升，但不要让语言模型远离一个仍然流畅、通用、相对稳定的参考模型。

### Policy gradient：reward 加权的生成学习

最基本的恒等式是：

$$
\nabla_\theta\mathbb E_{z\sim p_\theta}[R(z)]
=
\mathbb E_{z\sim p_\theta}
\left[R(z)\nabla_\theta\log p_\theta(z)\right].
$$

- $\theta$：策略参数。
- $z$：策略生成的样本或轨迹。
- $p_\theta$：当前策略分布。
- $R(z)$：样本 $z$ 的奖励。
- $\mathbb E$：对当前策略采样取期望。
- $\nabla_\theta$：对策略参数求梯度。
- $\log p_\theta(z)$：生成样本的 log-probability。

直觉上，高奖励样本的 log-probability 应提高。这很像 reward-weighted SFT，但每次参数变化后重新 rollout 很昂贵；我们希望同一批 rollout 能被多次利用。

### TRPO：用 trust region 控制样本复用

TRPO 用旧策略样本上的 importance ratio 估计新策略目标，同时用 KL 约束限制更新幅度：

$$
\max_\theta\;
\hat{\mathbb E}_t
\left[
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\hat A_t
\right]
\quad\text{s.t.}\quad
\hat{\mathbb E}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)
\Vert
\pi_\theta(\cdot\mid s_t)
\right)
\right]
\le \delta.
$$

- $\theta$：新策略参数。
- $\theta_{\mathrm{old}}$：采集当前 batch 时的旧策略参数。
- $t$：样本或决策位置索引。
- $s_t$：状态；语言模型中可理解为 prompt 加当前前缀。
- $a_t$：动作；语言模型中为下一个 token。
- $\pi_\theta(a_t\mid s_t)$：新策略的动作概率。
- $\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)$：旧策略的动作概率。
- $\hat A_t$：估计 advantage，表示该动作相对基线的好坏。
- $D_{\mathrm{KL}}$：KL divergence。
- $\delta$：允许的新旧策略平均距离上限。
- $\hat{\mathbb E}_t$：对采样 batch 的经验平均。
- $\max$ 与 `s.t.`：在满足约束的条件下最大化目标。

### PPO：把硬约束换成概率比裁剪

TRPO 的硬约束难以实现。PPO 用 clipping 限制新旧策略概率比：

$$
L^{\mathrm{clip}}(\theta)
=
\hat{\mathbb E}_t
\left[
\min\!\left(
r_t(\theta)\hat A_t,
\operatorname{clip}\!\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\hat A_t
\right)
\right],
\qquad
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

- $L^{\mathrm{clip}}$：PPO 的 clipped surrogate objective。
- $\theta$：新策略参数。
- $t$：样本或决策位置索引。
- $r_t(\theta)$：新策略与旧策略对同一动作的概率比。
- $\pi_\theta$：新策略。
- $\pi_{\theta_{\mathrm{old}}}$：采样时的旧策略。
- $s_t$：状态或生成前缀。
- $a_t$：动作或下一个 token。
- $\hat A_t$：估计 advantage。
- $\epsilon$：概率比的裁剪半径。
- $\operatorname{clip}$：把概率比限制在 $[1-\epsilon,1+\epsilon]$。
- $\min$：取未裁剪项和裁剪项中更保守的一项。
- $\hat{\mathbb E}_t$：对 batch 取经验平均。

![Policy gradient、TRPO 与 PPO 的概念演进](assets/ppo-conceptual-progression.jpg)

*课件图：从高方差 policy gradient，到 trust region，再到 clipped ratio；对应视频讲解 `01:07:14--01:08:59`。*

> [!NOTE]
> RLHF 目标中的 reference-model KL 与 TRPO/PPO 中的新旧策略距离作用不同：前者限制最终策略不要离 SFT 模型过远；后者控制一次优化中的更新步长和样本复用误差。

### 为什么人们想摆脱 PPO

PPO 需要 reward model、rollout、旧策略概率、advantage/value estimation 和外层循环，工程上很繁琐。直觉性替代方案包括：给 chosen/rejected 加 `[GOOD]`/`[BAD]` 控制 token、只对 chosen 做 SFT、由 reward model 筛选候选后再 SFT、从大量候选中选最好者做 rejection sampling。前两类通常丢掉太多偏好信息；后两类有用，但仍没有直接利用 winner 与 loser 的相对差值。

### 本章小结

- RLHF 同时最大化 reward 和限制策略偏离 reference model。
- Policy gradient 提高高奖励样本概率，但 rollout 昂贵、估计方差高。
- TRPO 用 trust region 控制新旧策略距离；PPO 用 clipping 近似这一思想。
- PPO 有效但系统复杂，这直接推动了 DPO 等离线偏好优化方法。

## DPO：把偏好优化改写成监督学习

### 删除显式 reward model 与 on-policy loop

DPO 的入口直觉非常简单：提高 winner 的 log-probability，降低 loser 的 log-probability，并根据模型当前对这对偏好的判断错误程度加权。它不需要单独训练并在线调用 reward model，也不需要 PPO 式 on-policy rollout 外循环。

![经典 RLHF 与 DPO 管线对照](assets/dpo-versus-classic-rlhf.jpg)

*课件图：DPO 直接从 preference data 到 final LM，删除显式 reward model 和 RL loop；对应视频讲解 `01:10:30--01:11:18`。*

### 第一步：从 KL 正则化目标得到指数倾斜策略

DPO 从以下 KL-regularized RLHF 目标出发：

$$
\max_{\pi}
\mathbb E_{x\sim\mathcal D,\;y\sim\pi(y\mid x)}[r(x,y)]
-
\beta D_{\mathrm{KL}}
\left[
\pi(y\mid x)\Vert\pi_{\mathrm{ref}}(y\mid x)
\right].
$$

- $\pi$：待优化策略。
- $\mathcal D$：prompt 分布或训练数据集。
- $x$：prompt。
- $y$：策略生成的回答。
- $r(x,y)$：回答的奖励。
- $\beta$：KL 正则强度。
- $D_{\mathrm{KL}}$：当前策略相对 reference policy 的 KL divergence。
- $\pi_{\mathrm{ref}}$：参考策略，通常是 SFT 模型。
- $\mathbb E$：对 prompt 和策略回答取期望。
- $\max$：寻找使正则化期望奖励最大的策略。

若先作非参数假设，允许 $\pi$ 表示所有分布，闭式最优策略是对 reference policy 做 reward 的指数倾斜：

$$
\pi_r(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{r(x,y)}{\beta}\right).
$$

- $\pi_r$：奖励 $r$ 下的非参数最优策略。
- $x$：prompt。
- $y$：回答。
- $Z(x)$：对同一 prompt 的归一化常数。
- $\pi_{\mathrm{ref}}$：参考策略。
- $r(x,y)$：回答奖励。
- $\beta$：KL 正则强度或温度。
- $\exp$：指数函数，使高奖励回答相对 reference 被放大。

反解可得到“隐式奖励”：

$$
r(x,y)
=
\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
$$

- $r(x,y)$：回答的隐式奖励。
- $x$：prompt。
- $y$：回答。
- $\beta$：KL 正则强度。
- $\pi_r$：优化后策略。
- $\pi_{\mathrm{ref}}$：参考策略。
- $Z(x)$：只依赖 prompt 的归一化常数。
- $\log$：自然对数。

![DPO 的指数倾斜策略与隐式奖励](assets/dpo-implied-reward-derivation.jpg)

*课件图：KL-RLHF 目标、非参数最优解和隐式奖励反解；对应视频讲解 `01:11:18--01:12:48`。*

### 第二步：把隐式奖励代入成对偏好损失

将策略 log-ratio 形式的奖励代入 Bradley-Terry loss。同一 prompt 下的 $\log Z(x)$ 在 winner/loser 差值中抵消，得到 DPO objective：

$$
\mathcal L_{\mathrm{DPO}}(\theta;\pi_{\mathrm{ref}})
=
-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\left[
\log\sigma\!\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right].
$$

- $\mathcal L_{\mathrm{DPO}}$：DPO 的监督式成对偏好损失。
- $\theta$：当前策略参数。
- $\pi_\theta$：待训练策略。
- $\pi_{\mathrm{ref}}$：固定参考策略。
- $\mathcal D$：偏好对数据集。
- $x$：prompt。
- $y_w$：winner 回答。
- $y_l$：loser 回答。
- $\beta$：隐式奖励尺度与 reference 约束强度。
- $\sigma$：sigmoid 函数。
- $\log$：自然对数。
- $\mathbb E$：对偏好数据集取期望。

![完整 DPO 目标和三步推导](assets/dpo-objective-derivation.jpg)

*1080p 视频帧：选择公式和三条关键步骤全部完整显示的状态；对应字幕讲解 `01:12:48--01:13:22`。*

推导可以压缩成三步：非参数假设把 policy 与 reward 闭式联系起来；用 policy 相对 reference 的 log-ratio 参数化 reward；再用普通 pairwise supervised loss 优化这个隐式 reward。

### 第三步：理解梯度在机械上做什么

DPO 梯度可以写成：

$$
\nabla_\theta\mathcal L_{\mathrm{DPO}}
=
-\beta\mathbb E
\left[
\sigma\!\left(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\right)
\left(
\nabla_\theta\log\pi_\theta(y_w\mid x)
-
\nabla_\theta\log\pi_\theta(y_l\mid x)
\right)
\right].
$$

- $\nabla_\theta\mathcal L_{\mathrm{DPO}}$：DPO loss 对策略参数的梯度。
- $\theta$：策略参数。
- $\beta$：更新尺度。
- $\mathbb E$：对偏好对取期望。
- $\sigma$：sigmoid 权重。
- $\hat r_\theta$：由策略相对 reference 的 log-ratio 定义的隐式奖励估计。
- $x$：prompt。
- $y_w$：winner 回答。
- $y_l$：loser 回答。
- $\pi_\theta$：当前策略。
- $\log\pi_\theta(y\mid x)$：回答的条件 log-probability。
- $\nabla_\theta$：对策略参数求梯度。

![DPO 梯度的两个组成部分](assets/dpo-gradient-components.jpg)

*课件图：更新方向负责 winner 上拉、loser 下压，sigmoid 项按隐式 reward 的预测错误加权；对应视频讲解 `01:13:22--01:14:17`。*

如果模型已经把 winner 明显排在 loser 前面，这一 pair 更新很小；如果两者概率接近或顺序错误，更新更强。这比“chosen 做 SFT、rejected 做负 SFT”更准确，因为两侧更新不是机械等权。

### DPO 只是 primitive，完整系统仍可有外循环

DPO 的核心优化器很像标准梯度训练，但完整后训练系统仍可能采用 expert iteration。LLaMA 案例中，模型先 SFT 和 DPO，再生成候选，由 reward model 做 rejection sampling，把筛出的数据送入下一轮。

![LLaMA 中的 DPO 与 expert iteration](assets/llama-dpo-expert-iteration.jpg)

*1080p 视频帧：选择所有模型、数据节点和回路完整显示的状态；对应字幕讲解 `01:14:48--01:15:09`。*

> [!NOTE]
> “DPO 不需要显式 reward model”描述的是 DPO 训练原语；它不排斥在更大的数据生成或 expert-iteration 外循环中使用 reward model。

### SimPO、长度归一化和经验脆弱性

SimPO 去掉 reference model，并用按序列长度归一化的平均 log-probability 比较胜负；length-normalized DPO 则对 winner 与 loser 的 log-ratio 分别按长度归一化。它们都试图减少长度本身对偏好分数的影响。

![DPO、SimPO 与长度归一化 DPO](assets/dpo-variants.jpg)

*课件图：两种代表性变体的目标函数；对应视频讲解 `01:15:11--01:15:35`。*

讲者不认为“DPO 一定优于 PPO”或反过来是一个脱离实验设置的定律。不同数据、reward、超参数、实现和训练 recipe 会给出不同结论。更稳健的认识是：这类方法都接近“相对地上拉好回答、下压坏回答”的有效核心机制，普通开放模型场景中 DPO 往往已经足够好。

### 本章小结

- DPO 用策略相对 reference 的 log-ratio 表示隐式奖励。
- 非参数最优策略是 reference policy 的 reward 指数倾斜。
- 将隐式奖励代入 pairwise loss 后，可用普通监督训练直接优化偏好。
- 梯度提高 winner、降低 loser，并按当前偏好预测错误自适应加权。
- DPO primitive 简单，但完整后训练管线仍可能包含生成、筛选和 expert iteration。

## RLHF 的边界：过优化、模式坍缩与失校准

### Proxy reward 会被优化器反向利用

Reward model 只是人类偏好的近似代理。随着优化强度增加，策略会寻找并利用 reward model 的误差：proxy reward 继续上升，真实人类偏好或外部评测却可能先升后降。这也是为什么 KL regularization 不是装饰；优化器越强，越需要限制策略走出 reward model 可靠的分布区域。

![Reward overoptimization](assets/reward-overoptimization.jpg)

*课件图：human preference 和 noisy LM preference 出现先升后降，图示的 noiseless GPT-4 preference 在该范围内近似单调；对应视频讲解 `01:16:43--01:17:27`。*

> [!WARNING]
> 右图的 noiseless 条件是特殊实验设置，不能概括为“模型评审 reward 永远不会被过优化”。关键变量是 reward 中是否存在噪声和可利用误差。

### 模式坍缩与概率失校准

生成建模试图拟合带内在多样性的分布；奖励优化只要求找到高分回答。RLHF 因而容易降低输出熵，让概率质量集中到少数模式。模型仍然输出概率分布，但训练目标不再保证多样性、校准或对原始分布的忠实拟合。

模式坍缩与失校准相关但不同：

- **Mode collapse**：输出多样性和熵下降，许多可行回答消失。
- **Miscalibration**：模型概率与经验正确率不匹配，置信度不再具有原来的统计含义。

![RLHF 后的熵下降与校准问题](assets/mode-collapse-calibration.jpg)

*课件图：calibration curve 与 entropy distribution 的变化；对应视频讲解 `01:17:27--01:18:49`。*

这一问题不只影响 confidence。下一讲讨论 RLVR 时，模型必须探索多种潜在解法，才能找到新的成功轨迹；熵过低会直接限制探索。

### 讲者的实质性结语

讲者最后留下三条总结：

1. RLHF 数据采集同样困难，而且充满混杂因素。
2. RLHF 算法比 SFT 更复杂，PPO 尤其难；课程作业会采用更简单的 GRPO。
3. 必须警惕对 reward 的过优化。

![Lecture 15 总结页](assets/lecture-recap.jpg)

*课件图：数据难、算法复杂、过优化三条 recap；对应视频讲解 `01:18:51--01:19:28`。*

结尾没有把 RLVR 当作已经证明的万能解，而是提出下一讲的问题：能否找到更不容易被过优化、可验证、噪声更低的 reward，使增加 compute 能更稳定地带来进步？这也是 RLVR 近年来影响力巨大的原因之一。

### 本章小结

- Learned reward 是 proxy，不是最终目标；优化足够强时会利用它的误差。
- RLHF 可能降低熵、造成模式坍缩，并破坏概率校准。
- 探索不足会影响后续 reasoning/RLVR 找到新解法。
- 下一步问题不是“怎样无限优化任何 reward”，而是“什么 reward 足够可靠，值得持续投入计算”。

## 总结与延伸

### 一条统一主线

整堂课可以压缩成一个控制问题：预训练给模型一个广阔但难以直接使用的行为分布，后训练逐步把它变成可部署的 policy。

1. **SFT 选择模式**：用 demonstration 教模型以期望的语气、格式和协议回答。
2. **Midtraining 扩大规模**：在预训练尾部提高高质量、指令型和合成数据比例。
3. **Preference data 定义目标**：通过指南、标注者和比较任务，把“好回答”操作化。
4. **PPO 或 DPO 优化目标**：提高偏好奖励，同时用 reference policy 抑制无约束漂移。
5. **风险控制**：检查 reward hacking、长度偏差、过优化、熵下降和失校准。

### SFT、PPO 与 DPO 的概念对照

| 维度 | SFT | PPO-RLHF | DPO |
|---|---|---|---|
| 直接训练信号 | 完整参考回答 token | Reward model + rollout | Winner/loser 偏好对 |
| 核心目标 | 拟合参考分布 | 最大化 reward 并限制漂移 | 提高相对偏好概率 |
| 是否需要显式 reward model | 否 | 是 | 核心训练不需要 |
| 是否需要 on-policy rollout | 否 | 是 | 否 |
| 主要优点 | 简单稳定 | 能直接优化可测目标 | 工程简单、离线训练方便 |
| 主要风险 | 模仿错误、知识与格式纠缠 | 系统复杂、reward hacking | 依赖偏好数据与 reference，仍会过拟合偏好 |

### 实践中的四个判断

- 如果目标行为已在强底座中存在，少量高质量 SFT 往往足以 steering。
- 如果目标是注入模型原本没有的专家知识，仅靠合成数据或未知事实 SFT 不可靠，需要真实专家和验证。
- 如果评价指标容易被长度、格式或其他代理变量操纵，先修正评审和数据，再选择更强优化器。
- 如果采用 DPO、PPO 或其变体，必须把 KL、熵、校准和真实外部评测一起监控，不能只看 proxy reward。

### 开放问题

- 如何可靠判断某种行为或知识是否已存在于预训练模型中？
- 如何设计既能覆盖长尾安全场景、又不过度误拒的持续数据管线？
- 如何区分“更详细”与“仅仅更长”，并建立不易被 reward hacking 的评价？
- 如何在偏好优化后恢复或保留多样性与概率校准？
- 可验证 reward 能否真正支持随 compute 增加而近似单调改进？

> [!IMPORTANT]
> 本讲最值得保留的不是“哪一种后训练算法最好”，而是一套诊断框架：先问数据在教什么，再问评价者在奖励什么，最后问优化器会怎样利用这些信号的漏洞。

### 本章小结

后训练把语言模型从“会续写的概率模型”变成“按目标行动的策略”。SFT、midtraining、RLHF、PPO 与 DPO 分别解决控制、规模、偏好和优化问题，但每一步都可能把数据与评价中的偏差写入模型。完整 pipeline 因此必须同时设计数据、反馈、优化约束和外部验证，而不能把任何单一算法当作秘密武器。
