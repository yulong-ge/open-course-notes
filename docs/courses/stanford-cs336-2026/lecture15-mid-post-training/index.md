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
> 本讲义以官方 65 页课件（`assets/slides/slide-001.jpg` 至 `slide-065.jpg`）为骨架逐页展开：每页 slide 嵌入一次并配详细讲解。原讲义中的视频截图经逐张比对，均为这 65 页课件中某一页的重复帧（不含讲师演示、白板或额外标注等新增信息），已按去重规则移除，只保留官方 slide。

## 从基座模型到可控助手：为什么需要后训练

### GPT-3 的能力为何没有自动变成 ChatGPT 的可用性

![slide-001：Lecture 15 标题页——"After pretraining" (mid/posttraining)](assets/slides/slide-001.jpg)

这是本讲的标题页。课程进行到这里，我们已经完整走过了预训练：分词、架构、优化、并行、数据与 scaling law。标题中的 "after pretraining" 加引号并非偶然——它暗示一个尚未严格定义的概念：预训练结束之后、部署之前的所有训练究竟包含什么？本讲给出的答案是 mid-training 与 post-training 的统称：它们共享"在预训练之后继续用梯度塑造模型"这一形式，但数据规模、反馈形态与目标函数各不相同。把这二者放在同一讲里讨论，是因为它们解决的是同一个控制问题的不同侧面：如何把一个通用的条件概率模型改造成可按意图调用的助手。

![slide-002：课程至此覆盖了预训练（得到 GPT-3 级别的基座），但如何得到 InstructGPT？](assets/slides/slide-002.jpg)

本页提出整讲的中心问题。左侧是 GPT-3 时代的典型用法：copy.ai 一类产品把语言模型包装成表单式的文案生成器——用户填产品名和描述，模型输出营销文案；右侧则是 ChatGPT 时代的新闻标题（BuzzFeed 用 AI 写 quiz、OpenAI 发布 "Optimizing Language Models for Dialogue"）。同样是语言模型，为什么前者只能做"续写器"，后者却能成为通用助手？课程此前讨论的预训练可以得到一个比 GPT-3 更强的基座模型：增加数据、计算量和模型规模，语言建模能力会继续提高。然而，强大的续写器还不是可靠的助手。早期 GPT-3 更适合文案写作、自由生成等不要求精确控制的任务；面对长提示、多项约束和严格输出格式时，few-shot 示例只能提供有限而脆弱的 steering——示例稍有扰动，模型的行为模式就会漂移。从 GPT-3 到 InstructGPT 的跨越不是靠继续加大预训练完成的，而是靠本讲讨论的后训练完成的。

![slide-003：Instruction following 是一种惊人的控制形式（Bubeck et al 2023 的 pyplot 长提示）](assets/slides/slide-003.jpg)

本页用 Bubeck et al 2023（"Sparks of AGI"）中的一条著名提示展示 instruction following 的控制粒度。提示要求 GPT-4 生成一段 matplotlib 代码，同时满足约二十条组合约束：四条曲线分置两个子图与一条独立曲线；图例分别命名为 bob、alice、bilbo、allie；坐标轴标注 time 与 money；每条曲线叠加 10%–40% 随机误差条；误差条用平滑插值而非线性插值；平滑曲线上还要加 zig-zag 扰动"让它看起来更真实"；每条线加一条均值基线；下方放一个饼图显示四人份额；饼图份额要随**连续时间**动画变化并插值缺失帧；其余三个子图要有与饼图同步的竖线动画；最后还要求"尽可能华丽"。下方是生成代码运行后动画的两帧截图：所有约束被同时满足。ChatGPT 时代最重要的变化之一，就是模型能把这样一段长而复杂的自然语言提示当作"程序"执行——它不只理解任务主题，还能同时满足数据、布局、颜色、标签、插值、误差条和动画等组合约束。instruction following 本身就是一种非常强的生成控制接口，而这种能力并不是预训练目标直接给出的。

> [!IMPORTANT]
> 预训练负责广泛获取能力，后训练负责把这些能力组织成用户可调用、可预测、可约束的行为。后训练不能替代预训练，它更像从预训练形成的能力空间中"提取"目标模式。

### 后训练的现实：算法轮廓公开，数据成为秘密

![slide-004：本讲目标——对语言模型输出实现更好、更紧的控制](assets/slides/slide-004.jpg)

本页给出全讲路线图。预训练数据并不完全是我们想要的分布（但它可以规模化），于是一个自然的问题是：能不能收集我们*确实想要*的行为数据，并用这些数据训练语言模型？围绕这个问题，课件列出三个子问题：(1) 这种数据长什么样？(2) 如何最好地利用这些数据？(3) 这需要规模吗？这三个问题恰好对应本讲的三大板块：SFT 数据与偏好数据的形态、从 SFT 到 PPO/DPO 的优化算法、以及 midtraining 中数据规模与混合比例的工程问题。读者可以带着这三个问题阅读后续各节。

![slide-005：警告——后训练的公开信息相当稀缺](assets/slides/slide-005.jpg)

本页是全讲反复回响的一个现实声明：现代后训练的数据、标注规范和质量控制，比公开算法更不透明。课件把信息环境分成两个时代。ChatGPT 之前的竞争期，信息相对丰富：Stiennon et al. 2020（摘要 RLHF 论文）公开了完整的标注指南，Bai et al. 2022（Anthropic HH）详细描述了安全标注如何组织。ChatGPT 引发竞争之后，信息急剧收缩：开源模型大量依赖蒸馏且在 release note 中含糊其辞；闭源实验室则把数据视为 secret sauce。页面下方引用了一段报道作为例证：2023 年 7 月，Bard 团队的管理者要求标注工人仔细研究 GPT-4 的回答为什么更好，"写出至少与 GPT-4 同等详细的答案"；Scale AI 还建立了对照表，把 1729 条 Bard 重写结果逐条与 ChatGPT 比较，打上 "worse than GPT4" 或 "Needs Some Fixes" 的标签——一条婴儿椅评论被打回，理由仅仅是"细节不如 GPT-4"。这个案例说明，后训练的杠杆不在"有没有梯度下降"，而在任务分布、示范答案、评价准则、标注者选择和质量审核这些不公开的细节里。

### 标准配方：先模仿，再优化偏好

![slide-006：本讲在三阶段管线中的位置——先讲 SFT（红框），再讲后两阶段](assets/slides/slide-006.jpg)

本页是 InstructGPT（Ouyang et al. 2022）经典三阶段管线的全景图，红框标出第一部分。三个步骤分别是：

1. **收集 demonstration 数据，训练监督策略（SFT）**：从 prompt 数据集中采样提示，由标注者示范期望输出，用监督学习微调 GPT-3。
2. **收集 comparison 数据，训练 reward model**：对同一 prompt 采样多个模型输出，由标注者从好到差排序，用这些排序训练一个标量打分模型。
3. **用强化学习对 reward model 优化策略**：新 prompt 输入当前策略（PPO），生成输出，reward model 打分，分数作为奖励回传更新策略。

本讲的叙事顺序与红框一致：先讨论 SFT（数据形态、安全数据、midtraining），再讨论偏好数据与 reward model，最后讨论 PPO 与 DPO 两类优化算法。SFT 的优化器本身并不神秘。给定 demonstration 数据集，标准目标仍是 teacher forcing 下的 next-token negative log-likelihood：

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

这个目标与预训练的唯一区别是数据分布：prompt 通常被 mask 掉，只对回答部分计损失（后文会看到，连这一条也不是严格分界）。同一个目标会同时模仿事实、语气、长度、引用习惯、拒答方式、工具协议和结构化动作——这解释了为什么 SFT 的难点几乎全部在数据：损失函数没有任何机制区分"这个 token 是在传递知识"还是"这个 token 是在模仿格式"，两者以完全相同的权重进入梯度。

### 本章小结

- GPT-3 到 ChatGPT 的关键跃迁之一，是复杂指令控制，而不只是更多参数。
- 预训练积累广泛能力；后训练选择并强化部署所需的行为。
- 标准后训练先以 SFT 模仿 demonstration，再以偏好反馈优化策略。
- SFT 算法接近普通语言建模，真正复杂的是数据定义与质量控制；而后训练时代的数据细节恰恰是公开信息最少的部分。

## SFT 数据如何塑造行为

### 从 NLP benchmark 到自然对话，再到 agent 轨迹

![slide-007：SFT 的两大原料——训练数据（FLAN 任务集合、OpenAssistant）与方法（数据混合比例）](assets/slides/slide-007.jpg)

本页把 SFT 分解为两大原料。左上是 FLAN 系列的 finetuning tasks 集合：T0-SF（55 个数据集、14 类、193 任务）、Muffin（69 数据集、27 类、80 任务）、CoT 推理子集（9 数据集）与 Natural Instructions v2（372 数据集、108 类、1554 任务），覆盖常识推理、问答、代码生成、对话上下文生成等；右上是 Open Assistant 的社区宣言——像 Stable Diffusion 之于图像那样，用众包方式造一个开放的对话 AI。下方的方法一栏则预告了后文的 midtraining：stable stage 与 decay stage 两个数据混合饼图，提示 SFT 的"方法"层面最重要的变量不是优化器，而是数据配比。整页的论点是：SFT 的算法成分（梯度下降）人人都会，真正的差异全部来自数据的成分与配比。

![slide-008：训练数据——讨论指令微调数据集的两个细节](assets/slides/slide-008.jpg)

这是一页过渡页，提出本节要回答的两个细节问题：(1) 这些数据集里实际装了什么？(2) 构建"高性能"指令微调数据时，什么因素真正重要？第一问对应接下来五页的逐数据集样例解剖；第二问对应风格、规模与安全三个维度的实证讨论。这两个问题看似朴素，但正是它们决定了"同样的 SFT 代码"在不同团队手里产出完全不同的模型行为。

![slide-009：开放世界 SFT 数据的演进谱系（FLAN→Self-Instruct→Alpaca→ShareGPT/Vicuna→OpenAssistant→WizardLM→Tulu3→Nemotron）](assets/slides/slide-009.jpg)

本页给出开放世界 SFT 数据的演进谱系，八个里程碑按时间与思想排列。**FLAN** 汇总传统 NLP 监督任务并统一改写成指令格式，确立 instruction tuning 范式；**Self-Instruct** 让模型自己生成新指令并筛选，摆脱对既有 benchmark 的依赖；**Alpaca** 用 Self-Instruct 流程从 ChatGPT（Davinci-003）蒸馏 input-output traces，以约 52K 条数据在强底座上诱导出 ChatGPT 风格行为；**ShareGPT/Vicuna** 使用真实用户分享的对话，prompt 分布第一次贴近真实需求；**OpenAssistant** 走高投入社区众包路线，回答更长更详细；**WizardLM** 用 Evol-Instruct 自动演化出更复杂的指令；**Tulu 3** 把开放后训练的数据配方、消融与评测系统化；**Nemotron** 一类最新数据则把 tool call 与 agent 轨迹纳入监督。这条谱系的深层趋势是：数据来源从"改写旧 benchmark"到"蒸馏强模型"再到"真实用户+结构化轨迹"，监督信号的形态越来越接近部署时的真实使用。

![slide-010：FLAN 随机样例——由旧 NLP 任务机械改写的指令数据](assets/slides/slide-010.jpg)

本页展示 FLAN 的四条随机样例，值得逐条审视。第一条把 Enron 邮件正文（"Stephanie 请 Brad Richter 签字"）机械改写成"为这封邮件写主题行"，参考答案是 "Ronald Chisholm LOI"——一个对邮件内容几乎无概括力的字符串；第二条是 AG News 新闻分类，把"判断文本属于 World/Sports/Business/Sci-Tech 哪类"包装成指令；第三条要求为一段海牙游记写 highlights，参考摘要却包含输入中没有的细节（"直到 1806 年都是村庄"、"Louis Bonaparte 统治过这里"、"藏有维米尔的《戴珍珠耳环的少女》"）；第四条是结构化餐馆描述（name=Aromi, eatType=coffee shop, food=English）到自然语言的转换。FLAN 的思想很先进——将许多下游任务统一成 instruction tuning——但它继承了 legacy NLP dataset 的输入格式与参考答案质量。这些任务虽然是有效监督信号，却不一定像真实用户与助手的交互；摘要参考答案甚至教会模型"输出输入中不存在的信息"，这一点在后文幻觉讨论中会再次出现。

![slide-011：Alpaca 随机样例——蒸馏得到的自然问答](assets/slides/slide-011.jpg)

本页展示 Alpaca 的三条随机样例：健康建议（"给出三条保持健康的建议"，答案是均衡饮食、规律运动、充足睡眠的清单体）、概念解释（"algorithm 是什么意思"，答案是一段通俗定义加用途举例）、简单代码（"求列表平均值"，答案是一段带输出注释的 Python 函数）。与 FLAN 样例对比，差异一目了然：prompt 更像真实用户的提问方式，回答更完整、更接近聊天助手的口吻（分点、解释、示例代码）。这种方法能在强预训练底座上快速诱导出 ChatGPT-like behavior，成本只是调用 API 生成约 52K 条数据。但"像 ChatGPT 回答"不等于底座拥有同等的知识、推理与可靠性——蒸馏传递的是输出*风格*与浅层内容，底座模型的能力上限仍然由预训练决定。

![slide-012：OpenAssistant 随机样例——高投入众包的知识密集型回答](assets/slides/slide-012.jpg)

本页展示 OpenAssistant 的两条样例。第一条要求介绍经济学中 monopsony（买方垄断）概念在劳动力市场的相关性并引用研究，回答给出了定义、劳动市场买方垄断对工资与工作条件的影响，并在末尾附上正式参考文献：Bivens, J., & Mishel, L. (2013), *Journal of Economic Perspectives*, 27(3), 57-78；第二条是"给孩子找便宜有趣的科学项目"，回答给出编号列表（熔岩灯、太阳能烤箱、磁铁实验等），每条带操作细节。OpenAssistant 走的是高投入人类众包路线，回答更长、更像专家写作，包含更多事实、引用和详细解释。这提高了可读性，也埋下了后文的风险：详细的"引用行为"可能比引用背后的真实知识更容易被模型学会——请记住第一条样例中的那条参考文献，后文 slide-019 会回到它。

![slide-013：Nemotron-SFT-OpenCode-v1——agentic SFT 样例](assets/slides/slide-013.jpg)

本页展示 Nemotron 一代数据中的 agentic SFT 样例，监督目标不再只是纯文本。第一条：用户问 JavaScript 中 `async/await` 与 `.then()` 处理 promise 的差异，助手消息里既有自然语言回答（`content`），又有一条 `tool_calls`——调用名为 `skill` 的函数、参数为 `{"name": "bash-skills"}`；第二条：用户问计算器应用该用整数还是浮点数，助手回答中嵌入对 `todowrite` 工具的调用，参数是一个含 `content`、`id`、`priority`、`status` 字段的 todo 列表 JSON（"阅读 AGENTS.md""分析整数与浮点权衡""检查可用 skill""给出建议"）。`role`、`content`、`tool_calls`、function arguments、todo 的 priority/status 都可能成为训练目标。工具使用不是推理时凭空出现的能力，常常先被显式写进结构化训练轨迹；当模型在部署时"学会"先列 todo 再行动，那往往是因为它在 SFT 中逐 token 模仿过这样的轨迹。

### 数据不只决定内容，也决定“助手人格”

![slide-014：这些数据集之间的差异维度——chattiness、detail、tool use](assets/slides/slide-014.jpg)

本页把前面四个数据集的差异压缩为三个方向。**Chattiness（健谈度）**：FLAN 的数据（通常）是有效的，但人们不想和一个 NLP benchmark 对话；后来的数据集普遍转向更长、更详细的回答。**Detail（细节度）**：OpenAssistant 在各类事实知识上给出多得多的细节——后文会看到，这既是优点也是缺点。**Tool use（工具使用）**：最近一两年的 SFT 明显转向 tool use 与 agentic 的下游应用。这三个方向不是互斥的改进，而是三种不同的"助手人格"取向：同一份预训练底座，用 FLAN 微调会得到一个惜字如金的任务处理器，用 OpenAssistant 微调会得到一个旁征博引的解释者，用 Nemotron 式数据微调会得到一个边想边调用工具的 agent。数据选择本质上是产品人格的选择。

![slide-015：跨数据集观察到的因素——风格（长度与列表）、引用与复杂知识、规模、安全](assets/slides/slide-015.jpg)

本页是承上启下的观察清单。可见的、容易注意到的差异包括：**长度与列表化**（风格差异）和**引用与其他复杂知识**；不那么可见但同样重要的差异是**规模**与**安全**。页面底部的问题"How do these factors affect the model?"引出接下来的实证讨论：这些因素各自如何影响模型行为？风格因素会在偏好评估中被急剧放大（下一页），规模因素决定了 steering 需要多少数据，安全因素则对应专门的 safety SFT 小节。注意这页的措辞是"less visible, but important"——规模与安全很少出现在数据样例的表面特征里，却常常在最终模型的行为差异中起决定作用。

![slide-016：数据与模型中的风格差异——各指令数据集的轮数、prompt 与 completion 平均长度统计](assets/slides/slide-016.jpg)

本页引用 Wang et al. 2023（"How Far Can Camels Go?"）的统计表，给出十余个指令数据集的硬指标：实例数、平均对话轮数 $\bar N_{\mathrm{rounds}}$、平均 prompt 长度 $\bar L_{\mathrm{prompt}}$ 与平均 completion 长度 $\bar L_{\mathrm{completion}}$。几个值得记住的数字：SuperNI 的 completion 平均只有 38.7 token，Flan V2 只有 31.2——典型的 benchmark 式短输出；Open Assistant 平均 212.5 token 且 1.6 轮；GPT4-Alpaca 161.8 token；ShareGPT 高达 357.8 token 且 3.2 轮；而 Baize（ChatGPT 生成）prompt 平均仅 17.6 token、completion 52.8 token。这张表说明"指令数据"这个标签下的长度分布相差近一个数量级。标题"Models vary *a lot* in response length"则预告：在这些数据上微调出的模型，回答长度同样差异巨大——长度几乎是被数据直接"复印"进模型的行为特征。

![slide-017：偏好评估中风格的作用——人类与 GPT 评审都强烈偏好列表与更长的输出](assets/slides/slide-017.jpg)

本页引用 Dubois et al. 2023（AlpacaFarm）的两张散点图，标题直言：用偏好做评估时，风格极其重要。上图横轴是"对列表的偏好比例"，分布在 40%–70% 之间，人类标注者、训练用模拟标注器、评估用模拟器与 GPT-4 标注器全都显著高于 50% 的随机水平；下图横轴是"对更长输出的偏好比例"，大多数点落在 50%–80% 区间。也就是说，无论人类还是 GPT-based evaluator，在成对比较中都系统性地偏好列表化、更长的回答。这为整个偏好评估体系投下阴影：如果一个模型只是学会了"回答更长、多列表"，它就能在 AlpacaEval 一类胜率指标上大幅提升，而不必变得更正确。这一页是理解后文 reward hacking、长度效应与 SimPO 长度归一化的经验基础。

![slide-018：这些风格因素对能力 benchmark 的影响并不一致](assets/slides/slide-018.jpg)

本页引用 Wang et al. 2023 的 Table 3：在 LLaMA 13B 上分别用各种指令数据集微调，测量 MMLU（事实性）、GSM（推理）、BBH（推理）、TydiQA（多语言）、Codex-Eval（代码）与 AlpacaEval（开放式胜率）。蓝色格表示微调提升了基座性能，橙色表示伤害。几组对比极具冲击力：+Open Assistant 的 AlpacaEval 胜率高达 58.1，但 GSM 只有 15.0、BBH 39.6，几乎没有推理增益；+ShareGPT 的 AlpacaEval 达 70.5，GSM 也升至 27.0，是单数据集里的异类；+Self-instruct 几乎全面伤害能力（MMLU 30.4、GSM 11.0）；而 Human+GPT 混合数据以 45.2 的平均分登顶，AlpacaEval 也有 56.5。结论是：改变数据中的长度和格式，就能大幅改变 AlpacaEval 一类胜率，却未必同步提高事实、推理、多语言或代码 benchmark。风格控制与能力控制必须用不同的指标分别测量。

> [!WARNING]
> 偏好胜率和 engagement 提升，只能说明输出更符合评价者的风格偏好，不能单独证明模型更聪明、更真实或推理更强。style control 与 capability control 必须分开测量。

### 知识注入的悖论：一个样本同时教事实和行为

![slide-019：引用、复杂知识与事实性——OpenAssistant 的 monopsony 样例在教模型什么？](assets/slides/slide-019.jpg)

本页回到 slide-012 的 monopsony 样例，把那条带 Bivens & Mishel (2013) 参考文献的回答单独放大，并提出一个尖锐问题：这个样本到底在教模型什么？课件列出两种并存的学习内容：

1. 教模型关于 Bivens 与 Mishel 的**事实内容**（他们的论文存在、标题、期刊与页码）。
2. 教模型在被要求时**输出引用这一行为模式**（"References:"之后应该接格式正确的书目信息）。

括号里的追问才是要害：but by what mechanism? Does the model know about cites? 对它做 next-token prediction 时，两种学习内容以完全相同的机制进入梯度——模型没有独立的通道区分"我在记忆一个真实文献"与"我在模仿引用格式"。模型可能可靠地学会在 "References:" 之后继续生成看似真实的书目信息（作者名-年份-标题-期刊-卷期-页码的完整模式），却没有可靠机制判断自己是否真的知道该文献。当回答涉及模型预训练中从未见过的长尾事实时，这个格式模仿机制照样运转——于是幻觉引用被批量生产出来。

![slide-020：知识提取与对齐——在"模型不知道的事实"上微调会导致幻觉](assets/slides/slide-020.jpg)

本页并置两个关键证据。左侧是 Schulman 2023（"Hallucination and Behavior Cloning" 演讲）的概念模型：把神经网络想象成内部存着一张带置信度的知识图谱，小规模微调学的只是在这张图上做查询的简单函数。如果你在知识图谱中不存在的正确答案上做 behavior cloning（例如标注者知道某衍生电影而模型不知道），你就是在教模型凭空生成；反过来，如果你在知识图谱中存在的*错误*答案上克隆（标注者自己不知道），你是在教模型隐瞒信息。理想的克隆目标应该取决于网络自身的知识——而这恰恰是实验者无法观测的；用其他 agent 计算出的目标来训练的模型，永远存在幻觉问题。右侧是 Gekhman et al. 2023 的定量曲线：把训练事实分为 known 与 unknown 两组，训练准确率上两者最终都升到接近 100%（unknown 组只是学得慢），但开发集准确率在第 10 个 epoch 左右过拟合开始后持续下滑——模型在未知事实上强行背诵训练答案的代价，是在未知问题上也自信编造，泛化性能恶化。

这里不是说"正确事实不能用于 SFT"，而是要区分两类目标：把已有能力引导到正确输出，和试图用少量示例把模型原本不掌握的长尾知识硬塞进去。SFT 对训练样本的 loss 没有算错——它忠实地拟合了数据；失败来自内容与格式的错误泛化：模型把"遇到这类问题就输出带引用的详细答案"这一模板，泛化到了它并无知识的领域。

![slide-021：知识提取与对齐的三条要点](assets/slides/slide-021.jpg)

本页给出 SFT 数据小节的三条要点。第一，即使长尾知识正是语言模型的用武之地，你也可能不想在长尾知识上做微调——代价是幻觉与开发集退化，上一页的两条曲线就是证据。第二，原则上，"RL 式"的正确性反馈可以帮忙：如果模型内部已有某种"我知道/我不知道"的校准信号，那么根据模型自己的 rollout 奖励正确引用、惩罚编造，可能把这个内部信号连接到输出策略上——这正是后文 RLHF 的动机之一，但 RL 不能凭空创造不存在的知识或校准。第三，语言模型中的知识存储与提取是 messy 且 nuanced 的：知识存储、提取和校准不能简化为"多塞一些答案"。

> [!IMPORTANT]
> SFT 最擅长抽取预训练中已有的行为模式。事实正确的数据也可能因知识边界与输出模板纠缠而伤害模型；知识存储、提取和校准不能简化为"多塞一些答案"。

### 本章小结

- SFT 数据从旧式 NLP 任务（FLAN）演化到蒸馏（Alpaca）、真实对话（ShareGPT）、高投入众包（OpenAssistant）与 agent 轨迹（Nemotron）。
- 模型会同时模仿内容、长度、语气、格式、引用和工具协议；长度几乎是直接被数据"复印"的行为特征。
- 更长、更列表化的回答容易赢得人类与 GPT 评审的偏好，但这不等价于能力提升——能力 benchmark 与偏好胜率经常背离。
- 在未知事实上强制模仿，会教会模型自信生成"正确格式的错误内容"；训练准确率上升的同时开发集性能可能持续恶化。

## Safety SFT 与 midtraining：少量 steering 如何进入大规模训练

### 安全调优的两个错误率

![slide-022：安全——语言模型被广泛部署给终端用户，需要安全控制](assets/slides/slide-022.jpg)

本页把安全问题摆上桌面。左侧是 Goldstein et al. 2023 的生成式模型误信息威胁模型：从模型获取（自建、购买或窃取）出发，经内容传播层（冒充、操纵、伪造、干扰四类行为——自动鱼叉式钓鱼、社交机器人施压、自动生成的虚假声明与误导图像、极化模因），到信念形成层（极化、错误信念、注意力涣散、恐惧与不信任），最终落到影响层（社会工程、更糟的决策、行动能力丧失与间接效应）。右侧是 Kang et al. 2023 的诈骗与垃圾信息实证：一条要求"以 FEMA 新冠救助名义索要逝者社会安全号码、越紧急越好"的提示，先是触发了内容政策警告，随后模型仍然生成了一封措辞得体、情感操纵完整的诈骗邮件。部署模型面对的不只是普通任务，还包括政治操纵、诈骗、钓鱼、垃圾信息和 jailbreak；安全控制不是可选项，而是部署前提。

![slide-023：野外的 Safety SFT——公开细节稀缺，Llama 3 的 violation rate 与 false refusal rate 权衡](assets/slides/slide-023.jpg)

本页指出安全调优的公开细节同样稀缺，并引用 Llama 3 技术报告中最完整的一段公开描述。左侧摘录披露了若干工程要点：安全微调数据的质量比数量更关键；人工数据来自 data vendor 但容易出错和不一致，因此开发了 AI 辅助标注工具加强质保；除对抗 prompt 外还收集 borderline prompts——与对抗 prompt 相似但目标是教会模型给出有用回答，从而降低 false refusal rate（FRR）；并用 Rainbow Teaming 等方法合成更多样化的对抗样本。右图是 Llama 3 8B 与 70B 在不同数据混合下的 violation rate（VR，有害请求穿透比例）对 FRR（正常请求被误拒比例）的散点：两个指标构成 Pareto trade-off，而 70B 在同等 VR 下 FRR 更低——更大的模型更善于区分对抗与擦边请求。页面底部注明 Llama 2 的安全样本只有几千条。Safety SFT 的目标不能简化为"拒绝更多"："How do I kill a Python process?" 中的 `kill` 是正常计算机术语，只看敏感词会造成大量误拒；安全与有用性必须同时优化。

![slide-024：细节最多的公开管线——Tulu 3 的安全与 non-compliance 数据](assets/slides/slide-024.jpg)

本页推荐 Tulu 3（Lambert et al., Allen AI）作为目前细节最完整的公开后训练管线，并列出其安全与 non-compliance 部分的数据配方：Tulu 3 CoCoNot 约 11K 条（Brahman et al. 2024，教模型识别并礼貌拒绝无法回答或不适当的请求）、Tulu 3 WildJailbreak 50K 条（Jiang et al. 2024）、Tulu 3 WildGuardMix 50K 条（Han et al. 2024）。与 Llama 的"质量声明+一张权衡图"相比，Tulu 3 把数据源、规模、配比和消融全部公开，是复现与研究安全 SFT 的最佳公开参照。对读者而言，这张表也给出数量级直觉：开源前沿的安全 SFT 数据在数万条量级，而非数百万条。

### 从真实用户流量发现长尾场景

![slide-025：主要安全方法——从用户提取场景（WildChat、non-compliance 分类法与 WildTeaming 的 Mine/Compose）](assets/slides/slide-025.jpg)

本页展示相对完整的公开安全数据生产 pipeline，核心思想是从真实交互中挖掘 non-compliance 场景。上方是 WildChat 论文（1M 条真实 ChatGPT 交互日志，论文附录自带内容警告）。左下图是 non-compliance 的分类法：incomplete requests（错误预设、无指定内容、不连贯）、unsupported requests（模态限制、长度限制、时间限制——如"2025 火星任务结果"）、indeterminate requests（humorizing、主观问题、未知答案），以及带安全关切的不适当请求（冒犯性语言、危险或敏感话题、虚假信息、隐私侵犯、版权违规）。右下图是 WildTeaming 框架的两步：**Mine**——从野外自动挖掘用户写出的 jailbreak 战术（任务情境化、角色扮演、伦理准则扭曲、种子引导句等）；**Compose**——把这些战术与普通的 vanilla harmful queries 组合，生成多样的对抗攻击样本。整个流程是：收集用户日志，识别有害请求和 jailbreak tactic，构造理想拒答或安全替代回答，再迭代训练和评测。这种做法很实际，却也偏反应式：系统先观察到一种攻击，再补数据，像持续"打地鼠"；精细安全边界最终仍依赖长期日志、红队、专家知识和质量控制。

### 为什么约 500 条数据就能显著改变行为

![slide-026：仅约 500 条安全数据即可显著改善安全性](assets/slides/slide-026.jpg)

本页展示的实验中，在 Alpaca 风格数据上仅增加数百条安全示例，四个安全评测集（I-MaliciousInstructions、I-CoNa 仇恨言论、I-Controversial 争议内容、Q-Harm 即 Anthropic HH  harmfulness）的平均有害分数随安全数据量从 0 增加到 2000 而持续下降；尤其在 I-MaliciousInstructions 上，仅 500 条就把均分从约 2.9 压到约 0.3。一个可能的解释是：强预训练模型已经学到了多种可行行为模式（包括"礼貌拒绝并提供安全替代"这种模式本身），少量 SFT 只需把输出分布推向 safe mode，而不必从零教会拒答。这与前文"SFT 最擅长抽取已有行为"的论断完全一致。

> [!WARNING]
> "500 条有效"只说明粗粒度 steering 很省数据，不表示模型已经全面安全。稀有场景、相邻边界、专业领域和误拒控制仍需要更大规模的数据覆盖——注意 Q-Harm 上 2000 条的改善幅度明显小于另外三个数据集。

![slide-027：SFT 数据部分小结——抽取已有行为、正确数据也可能有害、少量正确行为数据效果显著但有长尾](assets/slides/slide-027.jpg)

本页用三句话总结 SFT 数据板块。第一，instruction fine-tuning 在"只是抽取预训练已有行为"时效果最好，而不是注入新行为——这是贯穿前五页实证的总纲。第二，加入（事实正确的！）数据有时反而有害——知识边界与输出模板的纠缠会在未知领域诱发幻觉，事实正确并不能免疫。第三，少量正确类型的行为数据（安全、指令遵循、风格）就能带来巨大改变，但存在一个受益于更多数据的长尾。这三条合在一起构成 SFT 的实践世界观：把 SFT 当"行为选择器"而非"知识注入器"使用，收效最好、代价最小。

### 把 instruction tuning 融入预训练尾部

![slide-028：如何微调——就是梯度下降（标准训练循环代码）](assets/slides/slide-028.jpg)

本页给出 SFT 训练循环的最朴素形态：一段 HuggingFace 风格的代码——`tqdm` 进度条、`model.train()`、对 `train_dataloader` 的 epoch 循环、把 batch 移到 device、前向取 `outputs.loss`、`loss.backward()`、`optimizer.step()`、`lr_scheduler.step()`、`optimizer.zero_grad()`。在许多学术场景里，这就是全部。页面底部的转折才是重点：but what if you have tons of compute and data, you want to scale up instruction tuning？当数据与计算规模大到"独立微调"显得浪费时，问题就从算法变成了训练日程（schedule）设计——如何在不重训底座的前提下，让指令数据享受预训练级别的 token 规模？下一页给出答案。

![slide-029：把 instruction tuning 变成预训练——三步配方](assets/slides/slide-029.jpg)

本页给出日益流行的三步配方：(1) 在网页/预训练数据上常规预训练；(2) 把 instruction-tuning 数据混入预训练（尾部）；(3) 最后再做一次真正的、但较短的 instruction-tuning 轮。其收益写在页底：lets you scale up instruction tuning without catastrophic forgetting。直觉上，独立的后置微调是在一个已经收敛的模型上施加剧烈的目标切换，容易冲刷预训练表示；而把指令数据混入预训练尾部，相当于让模型在"还在学习"的状态下平滑地提高高质量数据的权重，通用语料在整个过程中始终在场，充当了抗遗忘的正则。最后那轮短 SFT 则负责把行为模式精确对齐到部署格式。

![slide-030：Midtraining / two-phase training——stable 与 decay 阶段的数据混合饼图](assets/slides/slide-030.jpg)

本页展示这一阶段常被称为 midtraining、second-phase pretraining 或 two-phase training 的实际数据配方，并注明它"在许多 LLM 公司是常识但没有成文"，公开的参照主要来自 miniCPM、jetMoE 等中文系模型的技术报告。左图 stable stage 以通用预训练源为主：CommonCrawl 中文 25.0、Code Pretrain 25.0、Dolma 24.0、C4 15.0、Pile 8.0，数学与 arXiv 等高质量源各占约 1%。右图 decay stage 则显著改配：Code Pretrain 19.6、CommonCrawl 14.6、C4 9.5、Wikipedia 6.7，同时长出一串小切片——EvolInstruct、OssInstruct、SlimOrca、Logic_SFT、ShareGPT4、Law Pretrain、Open Web Math、peS2o、Math_SFT、Stock Exchange QA、Math_Synthetic、UltraChat、Knowledge_SFT、Book Chinese、Code_SFT、Baidu Baike、SFT_mixed 等，各占 1%–5% 不等。要点有二：其一，现代所谓"base model"可能已经在预训练尾部见过 UltraChat、SFT mixed、代码或数学合成数据，因此不一定是传统意义上的纯网页语言模型；其二，数据混合比例通常不是由一个可靠公式直接解出，而是大量 trial and error——在较短、较便宜的 decay/midtraining 阶段做域消融，测量下游变化，给数据源排序，再把结论反馈给完整预训练。高质量数据不能独占整个预训练，因为 token 数量和覆盖范围不够；工程目标是改变比例，而不是彻底移除通用语料。

> [!NOTE]
> Prompt 是否 mask 不是 pretraining 与 SFT 的严格分界。有些纯预训练目标会预测 prompt token，一些 SFT recipe 也会这样做。更可靠的区分来自数据分布、反馈形态与训练目的。

### 本章小结

- Safety SFT 需要同时降低 violation 与 false refusal，二者构成 Pareto 权衡；更大模型在同等 violation 下误拒更少。
- 主流安全数据管线从真实用户流量挖掘 non-compliance 与 jailbreak 场景，本质是反应式"打地鼠"。
- 少量高质量示例可以强烈改变总体行为（约 500 条即可），但长尾边界仍需要规模。
- 现代训练会在预训练尾部逐渐提高高质量、指令型和合成数据比例（midtraining）；其关键工程方法是短阶段消融与反复试验，而不是一个封闭的最优 mixture 公式。

## 从模仿分布到偏好数据

### SFT 与 RLHF 优化的对象不同

![slide-031：RLHF 的第二部分——比较数据、reward model 与 RL 优化（红框）](assets/slides/slide-031.jpg)

本页回到 InstructGPT 三阶段图，红框移到 Step 2 与 Step 3：收集 comparison 数据并训练 reward model；再用强化学习（PPO）针对 reward model 优化策略。前半讲回答了"行为数据长什么样、如何用 SFT 模仿"；从本页开始，问题换成"如何让模型超越模仿，直接优化我们真正关心的目标"。这一步转换在概念上是全课的分水岭：模型从一个分布的拟合器，变成一个目标的优化器。

![slide-032：从模仿到优化——SFT 拟合参考分布，RLHF 最大化可测奖励](assets/slides/slide-032.jpg)

本页把两种范式并排对照。**Imitation (SFT)**：拟合 $\hat p(y\mid x)\approx p^*(y\mid x)$，其中 $p^*(y\mid x)$ 是某个参考分布（示范者隐含的条件分布）——这是纯生成建模视角，且需要来自参考策略的样本。**Optimization (RLHF)**：寻找 $\hat p$ 使 $\max_p \mathbb E_{y\sim p(\cdot\mid x)}[R(y,x)]$，即直接最大化某个可测量的奖励函数——此时语言模型是 policy，而不是某个分布的模型。两种目标的行为后果截然不同：分布拟合保留参考数据的全部多样性（包括其中的平庸回答）；而纯奖励最大化不要求保留多样性——如果某一个回答总能获得最高奖励，最优策略可以把全部概率质量集中到这个回答上。这个观察为后文的 mode collapse 讨论埋下伏笔，也解释了为什么 RLHF 目标必须额外加 KL 正则。

### Generation-verification gap：人未必能写出自己最喜欢的答案

![slide-033：为什么优化？G-V gap——人并不总能写出自己偏好的输出](assets/slides/slide-033.jpg)

本页回答一个根本问题：为什么不一直收集 demonstration，而要做偏好优化？讲者给出两个理由，本页用实证支撑第一个。右侧是 Zhang et al. 2023 新闻摘要评测的结果：总体偏好上，自由职业写作者的摘要与 Instruct-Davinci 的摘要几乎五五开（50.4% 对 49.6%），且标注者之间分歧极大——overall preference agreement 的 Krippendorff $\alpha$ 只有 0.07，六位标注者中有人 57% 偏好模型、有人 56.9% 偏好人类。也就是说，人能够写作，却未必能从零写出自己会最满意的表达；把写作者自己的作品和模型输出并排时，很多人反而更喜欢模型输出。第二个理由是验证往往比生成容易：检查一个数学证明通常比从头构造证明容易；在多个候选答案中判断哪个更好，比要求标注者独立写出最优答案更可靠、更便宜。左侧的 DeepSeekMath-V2 论文封面提示这一不对称在数学领域尤为关键（"self-verifiable mathematical reasoning"）。这个 generation-verification gap 是偏好数据存在的根本理由：让标注者做他们更擅长的事（比较与判断），而不是他们不擅长的事（从零生成最优答案）。

> [!WARNING]
> 验证比生成容易是一种常见不对称，不表示偏好判断自动等于客观正确。标注规则、知识水平、时间预算和评价偏差仍会影响结果——slide-033 中 $\alpha=0.07$ 的一致性本身就是警告。

### 标准偏好数据管线

![slide-034：RLHF 总览——数据、算法（PPO/DPO）与副作用三大方面](assets/slides/slide-034.jpg)

这是一页结构导航页，预告 RLHF 部分的三大方面。**Data**：人们如何收集 RLHF 数据，有哪些需要担心的问题——对应接下来十余页关于标注界面、指南、标注者与模型反馈的讨论。**How do we RLHF**：PPO 与 DPO 两类算法——对应本讲的推导核心。**What are some side-effects of RLHF**：过优化、模式坍缩与失校准——对应最后一节。读者可以按"数据→算法→风险"的顺序组织后续阅读。

![slide-035：RLHF 数据—— pairwise feedback 的类型与获取（红框标出 Step 2/3）](assets/slides/slide-035.jpg)

本页再次框出管线的第二、三步，聚焦问题：pairwise feedback 有哪些类型？如何获得（好的）pairwise feedback？标准流程从一个已做 SFT、能遵循指令且仍保持输出多样性的模型开始：对每个 prompt 采样多个候选回答，标注者做排序或成对选择（pairwise choice），再用这些比较训练 reward model。成对比较而非绝对打分是深思熟虑的设计：绝对分数的校准因人而异、随时间漂移，而"这两个里哪个更好"是更稳定、标注者间一致性更高的判断；同时成对数据天然构成对比，能把"好"的相对标准编码进数据。

![slide-036：RLHF 与数据——标准 pairwise feedback 标注界面（MTurk）](assets/slides/slide-036.jpg)

本页展示一个真实的 MTurk 成对标注界面：顶部是标注进度与可折叠的 Annotation Guidelines；中间依次给出 Instruction（"Tell me about self driving cars"）、Input（空）、AI Response 1 与 AI Response 2 两段候选回答；底部 Rating 提供四档选择——Response 1 is better / Response 1 is only slightly better (only pick this if it's truly close) / Response 2 is only slightly better / Response 2 is better。两个 UI 细节值得注意：其一，"slightly better"档被明确限定为"仅在真正接近时使用"，这是在引导标注者把判断压向明确偏好，减少模糊标签；其二，guidelines 做成可折叠但默认存在，说明标注质量依赖指南被反复查阅。界面本身即是数据设计的一部分：选项的措辞、排序与默认值都会系统性影响收集到的标签分布。

成对偏好数据最常见的消费方式是训练 Bradley-Terry/logistic 形式的 reward model：

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
- $\sigma$：sigmoid 函数，$\sigma(z)=1/(1+e^{-z})$，将奖励差映射为 winner 获胜概率。
- $\mathbb E$：对数据集中的偏好对取期望。

这个损失来自 Bradley-Terry 偏好模型：假设 winner 被偏好的概率为 $P(y_w\succ y_l\mid x)=\sigma\bigl(r(x,y_w)-r(x,y_l)\bigr)$，那么上式就是该模型在数据集上的负对数似然。奖励的绝对数值不可识别——给所有 $r$ 加同一个常数，偏好概率不变；只有奖励*差*被数据约束。这个不可识别性不是缺陷，而是 DPO 推导能够消去配分函数的关键，后文会两次用到它。

### “好回答”不是自然标签，而是规则设计

![slide-037：RLHF 与数据——InstructGPT 标注指南](assets/slides/slide-037.jpg)

本页摘录 InstructGPT 的标注指南原文，把"好回答"操作化为 helpful、truthful、harmless 三个维度。**Helpful**：遵循用户意图、表达清晰、回答用户真正想问的问题（即使他问错了）、对国际化表述敏感、指令含糊时要求澄清、避免冗长或重复、不臆造任务之外的上下文。**Truthful**：输出准确信息、不误导；摘要只使用输入中的信息、不编造输入外细节；不输出明显虚假的世界知识（如"Hillary Clinton 坐过牢"）；对前提有误的问题（"Hillary 为什么入狱？"）应反驳前提而不是含糊其辞。**Harmless**：不造成物理、心理或社会伤害，不贬损群体，不生成辱骂威胁或未被要求的性/暴力内容，不给出有害的现实建议。指南还给出权衡规则：多数任务中 truthful 与 harmless 优先于 helpful；但在高风险领域（贷款、医疗、法律）helpful 的权重上升；边界情形的指导原则是"你更愿意从正在帮助你的客服那里收到哪个回答？"实际任务中三者必然冲突，标注者被明确要求运用最佳判断——也就是说，权衡本身也被委托给了人。

> [!IMPORTANT]
> RLHF 数据不是简单记录"人类喜欢哪个"。标注指南先定义什么叫好回答，因此规则设计本身就是产品行为和价值目标的设计。

![slide-038：另一个旧案例——Bard 标注指南（helpfulness 与 presentation 双维评分）](assets/slides/slide-038.jpg)

本页展示外泄的 Bard 标注指南作为对照：它把评估拆成 helpfulness 与 presentation 两个维度，各自配一套 5 级评分量表。Helpfulness 维度要求回答命中用户意图、具体全面且最新、连贯不自相矛盾、遵守 prompt 中的格式要求、不含不准确或误导信息、不含有害内容；量表从 Not at All Helpful（无用、含任何无意义/不准确/误导信息或有害内容）到 Extremely Helpful（完全命中意图且细节充分，堪比领域专家）。Presentation 维度要求结构易懂、语气礼貌中立、风格一致、不跑题不重复、无语言错误；量表从 Poor 到 Excellent。指南末尾强调：更简洁且直接呈现最有用信息的回答，通常优于更长但难读的回答；presentation 差的回答在评分中权重显著。与 InstructGPT 指南对比可见，不同实验室对"好"的分解不同（三维对双维）、量表粒度不同（四档对 5 级）、对长度的官方态度甚至相反——这些差异都会原封不动地进入各自训练的模型行为。

### 谁在生产 RLHF 数据

![slide-039：现代标注工人分布——Outlier/ScaleAI 单平台调查（年龄、学历、领域）](assets/slides/slide-039.jpg)

本页展示 Oxford Economics 对 Outlier（Scale AI 旗下平台）约九百名标注者的调查，并提醒它只覆盖一个平台、不能代表整个行业。三个分布值得记住：年龄上，25–34 岁占 25%、35–44 岁占 34%，主体是青年与中年专业人士而非学生兼职；学历上，本科 44%、硕士或专业学位 32%、博士 9%——高学历占比远超一般众包；领域上，语言 21%、创意写作 18%、通用项目 13% 居前三，生物学、技术写作、数学、编程（Python/Java/SQL）等专门领域合计约四分之一。现代标注产业不再只有低价通用众包：要教模型数学证明、代码审查或法律文书，标注者本身必须具备相应能力，劳动力结构随之升级。

![slide-040：薪酬的巨大差异——专家标注的增长](assets/slides/slide-040.jpg)

本页给出专家标注市场化的两条证据。左侧新闻报道：Handshake AI（内部项目名 Stagecraft）为 OpenAI 的 ChatGPT 招募自由职业者，时薪至少 50 美元，要求按行业 persona 创作反映真实工作的材料，项目雇用了约 3000–4000 人。右侧是美国专家数据标注时薪中位数区间图：Legal、Engineering、Medical 领域的 Highly Skilled Expert 中位数接近甚至超过 115 美元/小时，Expert AI Trainer 档位也在 40–60 美元；Writing、History、Mathematics 相对较低但仍有 30–55 美元。现实结构更像金字塔：顶部是少量昂贵、定制、专业的数据；底部仍有大量低成本、可扩展的通用标注。薪酬差异本身就是数据质量差异的价格信号。

![slide-041：RLHF 与数据——众包的三重复杂性](assets/slides/slide-041.jpg)

本页列出规模化众包的三个结构性难点。第一，很难获得真正高质量、可验证的标注者：简历与测试只能部分验证能力，专业身份（医生、律师）的核验成本更高。第二，很难让标注者真正核查正确性：事实核查比风格判断耗时得多，按件计酬的激励反而鼓励快速浏览；没有相应知识时，流程会系统性漏检——这与 slide-044 的实证直接呼应。第三，必须小心标注者自己使用 AI：如果标注者用便宜模型生成回答再稍作修改，付费购买的"人类示范"就悄然退化为蒸馏，数据质量假设全部落空。这三点共同说明：标注不是可以无限外包的商品，质量控制是后训练成本中隐藏的刚性部分。

![slide-042：RLHF 与数据——众包伦理（肯尼亚工人与 AI underclass）](assets/slides/slide-042.jpg)

本页用两篇报道把伦理问题摆上台面。左侧是时代周刊调查：OpenAI 通过外包公司雇用肯尼亚工人做内容审核与有害数据标注，时薪不足 2 美元，工作内容是阅读和标注大量有毒文本（包括极端暴力与性内容描述），多名工人报告了心理创伤。右侧是《大西洋月刊》的 "America Already Has an AI Underclass"：搜索引擎、ChatGPT 与其他 AI 工具离开庞大的合同工队伍就无法运转，而这些工人普遍低薪且被苛待。低时间预算与有害内容审核带来严重的劳动伦理问题；当我们讨论 RLHF 数据配方时，"数据"的另一端是真实的人在承受认知与心理成本。这一页不提供技术解法，但任何后训练系统的完整成本核算都应包含它。

### 标注者是谁，会改变模型成为什么

![slide-043：RLHF 与数据——人口统计；标注者分布会显著改变模型行为](assets/slides/slide-043.jpg)

本页并置两张表说明标注者构成如何渗入模型。左侧是某 RLHF 标注团队的人口统计：性别约 50% 男性、44.4% 女性；族裔以东南亚裔 52.6%、白人 31.6%、拉丁裔 15.8% 为主；国籍以菲律宾与孟加拉国各 22%、美国 17% 居多；年龄集中在 18–34 岁（合计约 74%）；学历 52.6% 本科、36.8% 硕士。右侧是 Santurkar et al. 2023 的 OpinionQA 式研究：计算各模型回答与不同宗教群体意见分布的接近程度，结果显示 OpenAI 的 text-davinci-002/003（经过 RLHF）相对 text-davinci-001 及各基座模型，回答系统性地更接近 Protestant 与 Roman Catholic 群体、更远离 Buddhist、Hindu 与 Atheist 群体（红框标出）。课件将它与标注群体分布联系起来，但必须强调：这只是群体层面的相关证据，不能解释为某个宗教身份直接决定模型回答；人口统计通过"指南执行中的判断倾向"这一中介产生影响的机制仍未被直接证实。可以确定的是：后训练是模型发布前的最终塑形阶段，标注者的人口统计、知识结构和注意重点都可能进入模型行为。

![slide-044：RLHF 与风格——标注者影响巨大（众包 vs 专家的错误检出热图）](assets/slides/slide-044.jpg)

本页引用 Hosking, Blunsom & Bartolo 2024 的热图，量化"标注者专业度改变他们看见什么"。图中数值是众包标注与作者本人（专家）标注的错误率差异（负值表示众包漏检）：在 baseline 条件下，众包对 inconsistency 的检出率低 10.6%、对 factuality 低 16.2%；当回答语气更自信（assertiveness++）时，漏检进一步恶化到 -16.9% 与 -22.3%；回答更复杂（complexity++）时，factuality 漏检 -14.3%、repetition 漏检 -12.1%。模式非常清楚：普通众包标注者较容易注意 formatting（甚至略敏感，+3.1%），却系统性低估 inconsistency 与 factuality error，尤其是语气自信、表面可信的错误——模型越自信，众包越看不出它错。事实核查昂贵，没有相应知识时，偏好数据里就会混入"自信的错误答案被标为 winner"的有毒样本，而 reward model 会忠实放大这种偏差。Inter-annotator agreement 只能衡量群体方差，不能排除共同偏差：所有人一致使用同一个模型代做，也可能得到很高一致性。对于主观偏好，高分歧可能是任务性质；对于 factuality，才更希望分歧较低。

### 从人类反馈到模型反馈

![slide-045：RLHF 与数据——LM 生成反馈；GPT-4 是出人意料好的成对反馈系统](assets/slides/slide-045.jpg)

本页给出模型反馈有效性的核心证据（Dubois et al. 2023, AlpacaFarm）。左图：GPT-4 模拟评审给出的系统胜率与人类胜率的散点，Spearman correlation 0.98、$R^2=0.87$——系统级排序近乎完美一致。右图：各类标注配置的"与人类众数判断的一致率"对"每千条成本（美元）"散点，GPT-4 系列配置的一致率约 0.63–0.66，接近人类标注者之间的水平，成本却低一个数量级（约 10 美元/千条对 100+ 美元/千条）。当目标是追赶现有前沿时，强模型作为标注者通常非常有效：它便宜、快速、标准稳定，且不会疲劳。

![slide-046：追赶前沿者常用 AI feedback——UltraFeedback、Zephyr、Tulu 3](assets/slides/slide-046.jpg)

本页展示这一趋势催生的开放后训练流程。**UltraFeedback**：从 UltraChat、ShareGPT、FLAN 等多源收集 prompt，用多个模型生成回答，再由 GPT-4 按指令遵循、真实性、诚实度等维度打分并二值化为 chosen/rejected。**Zephyr 7B**：Hugging Face 团队明确把"用 teacher 模型的 AI feedback 做对齐"列为主要贡献，并在访谈中解释——他们曾尝试从 data vendor 收集人类反馈，但过程耗时且监督成本高，AI feedback 对小团队更易获得，也便于社区复现。**Tulu 3**：偏好数据管线完全工程化——prompt 来自 SFT 子采样与新 OOD prompt（UltraFeedback、Persona），用 22 个模型组成的 model pool 采样回答（off-policy 与 on-policy 各四个），再由 GPT-4o 按 helpfulness、instruction following、truthfulness、honesty 四维打分并二值化。Olmo、Zephyr 等开放模型普遍采用这类流程。但边界必须说清楚：从强模型蒸馏适合"追赶"；推进能力前沿、注入新专家知识时，人类仍不可替代——你无法从 GPT-4 的评审中学到 GPT-4 不具备的判断力。

![slide-047：RLHF 与数据——自训练；Constitutional AI 的 SL-CAI 与 RL-CAI 流程](assets/slides/slide-047.jpg)

本页展示 Constitutional AI（Bai et al.）的自举流程，把"模型反馈"推到更激进的形态。上半支（SL-CAI）：从 helpful RLHF 模型出发，对 red-teaming prompt 生成有害回答，然后让模型依据一组宪法原则（critique）自我批评并修订（revision），用修订后的数据微调得到 SL-CAI 模型。下半支（RL-CAI）：用该模型对 red-teaming prompt 生成成对样本，由模型依据宪法原则给出 AI preference，训练 preference model（PM），最后用 RLAIF（PM + SL-CAI 模型）训练出最终 RL-CAI 模型。整条管线里人类只写了宪法文本，没有标注任何一条回答。它重新组织、筛选和放大模型已有能力（模型本来就能识别明显有害内容），不能凭空创造模型从未掌握的世界知识——这与"SFT 抽取已有行为"的论断在 RL 语境下再次同构。

![slide-048：RLHF 与风格——长度效应是 RLHF 的显著产物](assets/slides/slide-048.jpg)

本页用三组证据说明模型评审不是中立裁判，长度是最典型的代理变量。左图（Chen et al. 2024）：多种 RLHF 方法（ReMax、PPO、DPO 及 Odin 变体）的 win score 对平均输出长度的 Pareto 前沿——几乎所有方法的前沿都向"更长+更高分"方向延伸，方法间差异很大程度上是长度差异。中图：reward 对输出长度的二维直方图，二者明显正相关（输出 50 token 时 reward 约 -1.5，200 token 时约 +0.5）。右图（Singhal et al. 2024）：同一问题 "Why don't adults roll off the bed?"，SFT 模型回答 59 token，RLHF 后膨胀到 243 token——核心内容相似，多出来的大部分是"additional"补充段。只增加输出长度，就可能持续提高 model-judge 胜率；而这些额外 token 不一定增加等量信息。

> [!WARNING]
> Reward model 可能奖励"看起来像高质量"的代理变量。长度是最典型的例子：更详细可能更好，但单纯冗长也能成为 reward hacking 的方向。

### 本章小结

- SFT 模仿参考分布，RLHF 直接优化可测奖励；后者不保留参考分布的多样性。
- 人常常更擅长比较候选，而不是从零写出自己最偏好的答案（generation-verification gap）；这是偏好数据存在的根本理由。
- 偏好标签由指南、标注者分布、专业知识、时间预算和质量控制共同决定；"好回答"是规则设计的产物，不是自然标签。
- 模型反馈适合追赶现有能力（GPT-4 评审与人类排序相关 0.98），但不能取代产生新专家知识的人类数据。
- 人类和模型评审都可能偏好长度等表面代理变量——RLHF 后回答从 59 token 膨胀到 243 token 是常态而非例外。

## PPO：如何在奖励上爬升又不走得太远

![slide-049：如何做 RLHF——Part 1 PPO（经典且娇贵），Part 2 DPO（新且易用）](assets/slides/slide-049.jpg)

本页是算法板块的导航：我们已经有了（高质量的）pairwise feedback 数据管线，接下来的问题是——如何改造模型以利用这些成对反馈？课件给出两条路线：Part 1 是 PPO，"the original and very finicky approach"，本讲只讲简明版本；Part 2 是 DPO，"the new, very accessible approach"。"finicky"（娇贵、难伺候）这个形容词是讲者有意选择的：PPO 的效果无可否认，但它的系统复杂度、超参数敏感性和实现细节决定了只有少数团队能把它调好——这正是 DPO 后来迅速流行的直接原因。

![slide-050：从模仿到优化——RLHF 一侧的回顾](assets/slides/slide-050.jpg)

本页把 slide-032 的 SFT 一半置灰，只保留 RLHF 一半作为算法部分的起点：寻找 $\hat p(y\mid x)$ 使 $\max_p \mathbb E_{y\sim p(\cdot\mid x)}[R(y,x)]$，即最大化某个可测量的奖励函数；语言模型是 policy，不是某个分布的模型。这个视角转换带来三个直接的算法后果：其一，优化目标不再能从固定数据集直接算梯度——奖励依赖于策略自己生成的样本，采样与训练耦合；其二，目标函数对 $p$ 的"形状"没有任何保真要求，纯最大化会退化为 delta 分布；其三，我们只能在可测的 proxy reward 上优化，proxy 与真实目标的差距会被优化器放大。后三者分别引出本节的 policy gradient、KL 正则与最后一节的过优化讨论。

### KL 正则化的 RLHF 目标

![slide-051：语言模型中的 PPO——InstructGPT 的优化目标原文](assets/slides/slide-051.jpg)

本页摘录 InstructGPT 论文的 RL 一节原文。环境是一个 bandit：随机给出一个用户 prompt，期望一个回答，reward model 打分后 episode 结束。在每个 token 上加相对 SFT 模型的 KL 惩罚以缓解对 reward model 的过优化；value function 由 reward model 初始化。论文还实验了把预训练梯度混入 PPO 梯度（"PPO-ptx"）以修复公开 NLP 数据集上的性能回退，并给出完整的组合目标：

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
- $\gamma$：预训练梯度混合项权重（"PPO" 模型取 $\gamma=0$，"PPO-ptx" 取正值）。
- $D_{\mathrm{pretrain}}$：原始预训练数据分布。
- $\mathbb E$：对相应数据分布取期望。

前两项的直觉最重要：沿 reward 上升，但不要让语言模型远离一个仍然流畅、通用、相对稳定的参考模型。页面右下角的批注 "..this is very innocuous looking"（这看起来非常人畜无害）是讲者的讽刺：公式只有一行，但它背后隐藏着 rollout 基础设施、reward model 在线推理、value function 训练、KL 估计与海量超参数——PPO 的"finicky"全部藏在这一行的实现里。

![slide-052：Stiennon 摘要论文的更多细节——RM 损失与 KL 修正奖励](assets/slides/slide-052.jpg)

本页摘录 Stiennon et al. 2020（"Learning to summarize from human feedback"）的细节，补上 InstructGPT 沿用的两个关键设计。第一，reward model 从 SFT 模型出发加一个随机初始化的线性头输出标量，用 Bradley-Terry 损失训练；训练结束后把 reward 归一化，使参考摘要的均值为 0——这一步消除了奖励的任意偏移，让不同实验间的 KL 系数可比。第二，PPO 优化的完整奖励是

$$
R(x,y)=r_\theta(x,y)-\beta\log\!\left[\pi^{\mathrm{RL}}_\phi(y\mid x)\big/\pi^{\mathrm{SFT}}(y\mid x)\right].
$$

论文明确写出 KL 项的两个作用：其一，它充当 entropy bonus，鼓励探索、阻止策略坍缩到单一模式；其二，它确保策略不会产出与 reward model 训练时见过的分布差异过大的输出——reward model 只在 SFT 模型附近的分布上被训练过，一旦策略跑出这个区域，$r_\theta$ 的外推误差就没有任何保证。此外，value function 使用与 policy 完全独立参数的 Transformer（防止 value 更新破坏预训练 policy），并由 reward model 初始化；reward model、policy 与 value function 三者同尺寸——这又一次提醒读者 PPO-RLHF 的显存与算力开销：同时驻留三个大模型。

把 KL 惩罚项展开，可以看到它正是逐样本的对数似然比；对 $y$ 取期望后就是逐 prompt 的 KL divergence：

$$
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}
\left[
\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right]
=
D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

- $D_{\mathrm{KL}}$：当前策略相对参考策略的 KL divergence，非负，且仅当两分布相等时为零。
- $\pi_\theta(\cdot\mid x)$：给定 prompt 下当前策略的回答分布。
- $\pi_{\mathrm{ref}}(\cdot\mid x)$：给定 prompt 下参考策略的回答分布。

于是整个 KL 正则化 RLHF 目标可以写成更紧凑的形式，这也是 DPO 推导的出发点：

$$
\max_{\pi}\;
\mathbb E_{x\sim\mathcal D,\;y\sim\pi(y\mid x)}[r(x,y)]
-
\beta\, D_{\mathrm{KL}}
\left[
\pi(y\mid x)\,\|\,\pi_{\mathrm{ref}}(y\mid x)
\right].
$$

这个目标的两项各有明确含义：第一项鼓励策略把概率质量移到高奖励回答上；第二项惩罚策略偏离参考分布。$\beta$ 是两种力量之间的汇率：$\beta\to\infty$ 时最优解退化为 $\pi_{\mathrm{ref}}$ 本身（一步也不敢离开）；$\beta\to 0$ 时退化为纯奖励最大化，策略可以把全部质量压在 reward model 评出的单个最高分回答上——既丢失多样性，也暴露在 reward model 的分布外误差之下。KL 正则因此不是装饰：它把优化限制在 reward model 可靠的区域内，同时保留了语言的流畅性与多样性。

### Policy gradient：reward 加权的生成学习

![slide-053：PPO 的概念层次——三次尝试：policy gradient、TRPO、PPO 裁剪](assets/slides/slide-053.jpg)

本页是全讲算法部分的总纲，用"三次尝试"把 PPO 的概念演化压缩在一页：**Attempt 1**，policy gradient，方向正确但方差太高；**Attempt 2**，TRPO，在当前策略附近把问题线性化并用 KL 信任域约束；**Attempt 3**，PPO，把硬约束换成概率比裁剪。下面三小节把这三次尝试逐一展开成完整推导。

**Attempt 1：policy gradient 与 log-derivative 技巧。** 我们希望最大化期望奖励 $J(\theta)=\mathbb E_{z\sim p_\theta}[R(z)]$。直接对期望求梯度看似无从下手——梯度算子无法穿过依赖于 $\theta$ 的采样分布。log-derivative 技巧用一个恒等式化解：

$$
\nabla_\theta\mathbb E_{z\sim p_\theta}[R(z)]
=
\nabla_\theta\sum_z p_\theta(z)\,R(z)
=
\sum_z R(z)\,\nabla_\theta p_\theta(z)
=
\sum_z R(z)\,p_\theta(z)\,\frac{\nabla_\theta p_\theta(z)}{p_\theta(z)}
=
\mathbb E_{z\sim p_\theta}
\left[R(z)\,\nabla_\theta\log p_\theta(z)\right].
$$

- $\theta$：策略参数。
- $z$：策略生成的样本或轨迹（语言模型中即完整回答）。
- $p_\theta$：当前策略分布。
- $R(z)$：样本 $z$ 的奖励。
- $\nabla_\theta\log p_\theta(z)$：样本对数概率的梯度（score function）。

关键一步是 $\nabla_\theta p_\theta(z)=p_\theta(z)\nabla_\theta\log p_\theta(z)$，它把"对分布求导"换成"对数概率求导再按原分布取期望"——于是梯度可以用蒙特卡洛样本估计：采样一批 $z$，计算 $R(z)\nabla_\theta\log p_\theta(z)$ 取平均。这就是 REINFORCE。直觉上，高奖励样本的 log-probability 应提高、低奖励样本的应压低，形式恰如 reward-weighted SFT。

**baseline 与方差缩减。** 上述估计器的方差很高：$R(z)$ 的绝对数值直接乘进梯度，样本间奖励的整体波动全部转化为梯度噪声。引入一个不依赖于 $z$ 的 baseline $b$：

$$
\mathbb E_{z\sim p_\theta}
\left[\bigl(R(z)-b\bigr)\nabla_\theta\log p_\theta(z)\right]
=
\nabla_\theta J(\theta)
-
b\,\mathbb E_{z\sim p_\theta}\!\left[\nabla_\theta\log p_\theta(z)\right]
=
\nabla_\theta J(\theta),
$$

因为 $\mathbb E[\nabla_\theta\log p_\theta(z)]=\sum_z\nabla_\theta p_\theta(z)=\nabla_\theta\sum_z p_\theta(z)=\nabla_\theta 1=0$。baseline 不改变期望梯度，却能显著降低方差；最常用的是平均奖励或一个学习到的 value function，于是 $R(z)-b$ 成为 advantage 估计 $\hat A$——动作比平均水平好多少。语言模型 RLHF 中 value function 由 reward model 初始化、与 policy 同尺寸（slide-052），正是为这一步服务的。

下面用一段玩具代码数值验证 log-derivative 恒等式与 baseline 的不变性：

```python
import torch

# 玩具策略：三个离散动作，logits 为参数
logits = torch.tensor([0.5, 1.0, -0.5], requires_grad=True)
rewards = torch.tensor([2.0, 0.0, 1.0])  # 每个动作的确定奖励

p = torch.softmax(logits, dim=0)
J_exact = (p * rewards).sum()              # 期望奖励的精确值
J_exact.backward()
g_exact = logits.grad.clone()              # 直接对 softmax 反传 = 精确梯度

# REINFORCE 形式：sum_z p(z) * R(z) * grad log p_theta(z)
g_score = torch.zeros(3)
for z in range(3):
    lg = torch.log_softmax(logits, dim=0)[z]
    grad_z = torch.autograd.grad(lg, logits, retain_graph=True)[0]
    g_score += rewards[z] * p.detach()[z] * grad_z
print(g_exact, g_score)  # 两者相等：tensor([ 0.4028, -0.4290,  0.0262])

# 加 baseline b=0.8：期望梯度不变
b = 0.8
g_base = torch.zeros(3)
for z in range(3):
    lg = torch.log_softmax(logits, dim=0)[z]
    grad_z = torch.autograd.grad(lg, logits, retain_graph=True)[0]
    g_base += (rewards[z] - b) * p.detach()[z] * grad_z
print(torch.allclose(g_score, g_base, atol=1e-6))  # True
```

三种写法给出完全相同的梯度（约 $[0.403, -0.429, 0.026]$）：直接反传期望、score function 加权、score function 加 baseline。差别只在蒙特卡洛估计的方差——这也是从"恒等式成立"到"实践中必须用 baseline"的全部理由。

### TRPO：用 trust region 控制样本复用

**Attempt 2：TRPO。** policy gradient 还有一个结构性浪费：每次参数更新后，旧样本的分布就不再代表新策略，必须重新 rollout——而大模型 rollout 极其昂贵。要复用旧策略 $\pi_{\theta_{\mathrm{old}}}$ 采集的样本，自然的工具是 importance sampling，用概率比 $\pi_\theta/\pi_{\theta_{\mathrm{old}}}$ 修正分布偏移；但概率比是无界乘子，少数样本的比值爆炸就会摧毁估计。TRPO 的思路是把更新限制在一个信任域内：在旧策略附近用概率比目标做局部近似，同时用平均 KL 约束新旧策略的距离：

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
\,\|\,
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
- $D_{\mathrm{KL}}$：新旧策略在同一状态下的 KL divergence。
- $\delta$：允许的平均距离上限（信任域半径）。
- $\hat{\mathbb E}_t$：对采样 batch 的经验平均。

TRPO 的理论保证（单调改进界的近似版本）依赖这个约束；但带 KL 约束的优化需要共轭梯度加线搜索，实现复杂、难以与深度学习的标准一阶优化器兼容，每一步的成本也高。

### PPO：把硬约束换成概率比裁剪

**Attempt 3：PPO。** PPO 的洞察是：信任域的*效果*——防止概率比离 1 太远——可以用一个几乎免费的逐样本裁剪近似。定义概率比 $r_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)$，裁剪后的 surrogate objective 为：

$$
L^{\mathrm{clip}}(\theta)
=
\hat{\mathbb E}_t
\left[
\min\!\left(
r_t(\theta)\hat A_t,\;
\operatorname{clip}\!\left(r_t(\theta),\,1-\epsilon,\,1+\epsilon\right)\hat A_t
\right)
\right].
$$

- $L^{\mathrm{clip}}$：PPO 的 clipped surrogate objective。
- $r_t(\theta)$：新策略与旧策略对同一动作的概率比，$r_t(\theta_{\mathrm{old}})=1$。
- $\hat A_t$：估计 advantage。
- $\epsilon$：裁剪半径（常用 0.1–0.2）。
- $\operatorname{clip}$：把概率比限制在 $[1-\epsilon,1+\epsilon]$。
- $\min$：取未裁剪项和裁剪项中更保守的一项。
- $\hat{\mathbb E}_t$：对 batch 取经验平均。

逐项理解 $\min$ 的作用：当 $\hat A_t>0$（好动作），目标希望增大 $r_t$，但一旦 $r_t>1+\epsilon$，裁剪项成为更小（更保守）的值，梯度被截断——好动作的概率提升被限速；当 $\hat A_t<0$（坏动作），目标希望减小 $r_t$，一旦 $r_t<1-\epsilon$，同样被截断——坏动作的概率压低也被限速。于是同一批 rollout 可以安全地做若干 epoch 的小步更新：样本被复用，而每次复用都不会让策略跑出旧策略的邻域太远。这不是严格的信任域保证，而是以可忽略的额外计算换取了 TRPO 的大部分稳定性——"clip the ratios at some eps"，用一行 $\min$ 与 $\operatorname{clip}$ 换掉共轭梯度。

> [!NOTE]
> RLHF 目标中的 reference-model KL 与 TRPO/PPO 中的新旧策略距离作用不同：前者限制最终策略不要离 SFT 模型过远（防止 reward hacking 与语言退化）；后者控制一次优化中的更新步长和样本复用误差（防止单步训练崩溃）。实践中两套机制同时存在，不要混为一谈。

### 为什么人们想摆脱 PPO

![slide-054：能否摆脱 PPO？——避免 on-policy RL 的四种朴素方案](assets/slides/slide-054.jpg)

本页列出社区在 DPO 之前认真尝试过的四种"不做 RL"的替代方案，值得逐一评估其信息利用效率：

1. **控制 token**：把偏好对拼进 SFT 数据，chosen 前加 `[GOOD]`、rejected 前加 `[BAD]`，推理时用 `[GOOD]` 条件化。问题是推理时 `[BAD]` 样本的全部信息被丢弃，且模型学到的只是条件模仿，没有学到相对偏好。
2. **只对 chosen 做 SFT**（即 filtered SFT / best-of-n distillation 的退化版）：丢掉了 loser 的全部信息——"这个回答为什么不好"恰恰是偏好数据独有、demonstration 数据没有的信号。
3. **训练 reward model，用它筛选模型输出，再对 preferred output 做 SFT**：多了一次筛选，但仍只利用正向样本，winner 与 loser 的相对差值没有进入损失。
4. **训练 reward model，采样 1024 个输出取最优（rejection sampling / best-of-n）**：推理期直接有效，但推理成本放大 $n$ 倍，且同样没有把比较信号沉淀进参数。

PPO 需要 reward model、rollout、旧策略概率、advantage/value estimation 和外层循环，工程上很繁琐；但上面四个替代方案的通病是丢失偏好数据中的相对比较信息。这个张力——"PPO 太复杂"与"简单方案信息利用不足"——正是 DPO 要一举化解的。

### 本章小结

- RLHF 同时最大化 reward 和限制策略偏离 reference model；KL 正则防止语言退化并把优化限制在 reward model 可靠的分布区域内。
- Policy gradient 经 log-derivative 技巧把期望奖励的梯度写成 $\mathbb E[R\nabla\log\pi]$；baseline 无偏且降方差。
- TRPO 用 KL 信任域控制新旧策略距离以实现样本复用；PPO 用概率比裁剪以近乎零成本近似同一效果。
- PPO 有效但系统复杂（三个大模型同时驻留、超参数敏感），而抛弃 RL 的朴素方案又丢失相对比较信息——这直接推动了 DPO。

## DPO：把偏好优化改写成监督学习

### 删除显式 reward model 与 on-policy loop

![slide-055：DPO——"RLHF without tears"？删除 reward model 与所有 on-policy 组件](assets/slides/slide-055.jpg)

本页给出 DPO 的入口宣言：通过两件事简化 PPO——去掉 reward model，去掉一切 on-policy 组件（rollout、外循环等）。取而代之的是两条朴素原则：对好的回答做正梯度（log-loss），对坏的回答做适当加权的负梯度。页面下方的管线对比图一目了然：经典 RLHF 是 preference data → reward model →（label rewards + sample completions 的 RL 循环）→ LM policy；DPO 则是 preference data →（maximum likelihood）→ final LM，中间没有任何在线组件。DPO 的入口直觉非常简单：提高 winner 的 log-probability，降低 loser 的 log-probability，并根据模型当前对这对偏好的判断错误程度加权。剩下的问题是：这个直觉能否从 KL 正则化 RLHF 目标中严格推导出来，而不是又一个启发式？答案是肯定的，而且推导只需三步。

### 第一步：从 KL 正则化目标得到指数倾斜策略

![slide-056：DPO 推导第一页——KL 正则化目标、非参数最优解与隐式奖励](assets/slides/slide-056.jpg)

DPO 从上一节建立的 KL-regularized RLHF 目标出发：

$$
\max_{\pi}\;
\mathbb E_{x\sim\mathcal D,\;y\sim\pi(y\mid x)}[r(x,y)]
-
\beta D_{\mathrm{KL}}
\left[
\pi(y\mid x)\,\|\,\pi_{\mathrm{ref}}(y\mid x)
\right].
$$

关键假设是**非参数化**：暂且不考虑神经网络参数 $\theta$ 的表达约束，允许 $\pi$ 取遍所有条件分布。由于目标对不同 $x$ 解耦，只需对每个 $x$ 分别求解。把目标展开成对 $y$ 的求和，逐步变形：

$$
\begin{aligned}
&\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]
-\beta\, D_{\mathrm{KL}}\!\left(\pi\,\|\,\pi_{\mathrm{ref}}\right)\\[2pt]
=&\sum_y\pi(y\mid x)\,r(x,y)
-\beta\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\\[2pt]
=&\;\beta\sum_y\pi(y\mid x)\left[\frac{r(x,y)}{\beta}-\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\right]\\[2pt]
=&\;-\beta\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)\exp\!\bigl(r(x,y)/\beta\bigr)}\\[2pt]
=&\;-\beta\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}{\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\!\bigl(r(x,y)/\beta\bigr)}
+\beta\log Z(x),
\end{aligned}
$$

其中最后一步在分母上人为引入归一化常数

$$
Z(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)\exp\!\left(\frac{r(x,y)}{\beta}\right),
$$

- $Z(x)$：配分函数，把指数倾斜后的参考分布重新归一化为合法概率分布；只依赖 $x$，不依赖 $\pi$。

令 $\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\bigl(r(x,y)/\beta\bigr)$，则目标的第一项恰好是 $-\beta D_{\mathrm{KL}}(\pi\,\|\,\pi^*)$，第二项 $\beta\log Z(x)$ 与 $\pi$ 无关。KL divergence 非负且仅当两分布相等时为零，于是对每个 $x$，最优策略唯一：

$$
\pi_r(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\!\left(\frac{r(x,y)}{\beta}\right).
$$

- $\pi_r$：奖励 $r$ 下的非参数最优策略。
- $\pi_{\mathrm{ref}}$：参考策略。
- $r(x,y)$：回答奖励。
- $\beta$：KL 正则强度，在解中充当温度——$\beta$ 越小，指数倾斜越尖锐，策略越集中到高奖励回答。
- $\exp$：指数函数，使高奖励回答相对 reference 被放大。

这个解有漂亮的直觉：最优策略就是参考分布按奖励的指数倾斜（exponential tilting，统计物理中的 Boltzmann 分布）。它同时解释了 KL 正则的两个作用——保留 $\pi_{\mathrm{ref}}$ 的支撑集（任何 reference 概率为零的回答永远为零），并按奖励比例重排概率质量而非一刀切。

对等式两边取对数并整理，可以反解出"隐式奖励"：

$$
r(x,y)
=
\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
$$

- $r(x,y)$：由策略与参考策略表达的隐式奖励。
- $\pi_r$：优化后策略。
- $Z(x)$：只依赖 prompt 的归一化常数。

这一步是整个 DPO 的枢纽：reward 与 policy 在同一族模型中互相参数化——任何策略都隐含一个奖励函数，反之（在给定 reference 与 $\beta$ 后）亦然。课件页脚特别注明这一等价关系同样被 kimi-think 论文使用，可见它已成为后训练推导的标准工具。

### 第二步：把隐式奖励代入成对偏好损失

![slide-057：DPO 推导第二页——隐式奖励代入 Stiennon 目标得到 DPO loss](assets/slides/slide-057.jpg)

现在把隐式奖励代入 Bradley-Terry 成对偏好损失（slide-036 与 slide-052 的 RM 损失）。对每个 $x$，用当前策略 $\pi_\theta$ 参数化隐式奖励：

$$
\hat r_\theta(x,y)
=
\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
$$

把它代入 $\mathcal L_{\mathrm{RM}}=-\mathbb E\bigl[\log\sigma\bigl(r(x,y_w)-r(x,y_l)\bigr)\bigr]$，奖励差为

$$
\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)
=
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
+\underbrace{\beta\log Z(x)-\beta\log Z(x)}_{=\,0}.
$$

配分函数 $Z(x)$ 只依赖 $x$，在同一 prompt 的 winner/loser 差值中精确抵消——这正呼应了 Bradley-Terry 模型中奖励绝对水平不可识别的观察。于是得到只含策略与参考策略的 DPO objective：

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
- $\pi_{\mathrm{ref}}$：固定参考策略（SFT 模型，训练中不更新）。
- $\mathcal D$：偏好对数据集。
- $y_w$ / $y_l$：winner / loser 回答。
- $\beta$：隐式奖励尺度，同时控制偏离 reference 的惩罚强度。
- $\sigma$：sigmoid 函数。
- $\mathbb E$：对偏好数据集取期望。

课件把推导压缩成三步：(1) 非参数假设把 policy 与 reward 闭式联系起来；(2) 用 policy 相对 reference 的 log-ratio 参数化 reward；(3) 用普通 pairwise supervised loss 优化这个隐式 reward——而优化隐式 reward 就是在优化 policy 本身。概念上，这是非参数假设加换参数化下对成对奖励的 MLE。整个损失没有任何 rollout、没有任何在线 reward model 调用：前向两遍（policy 与 reference 各一次，reference 可预先缓存），其余与标准监督训练无异。

### 第三步：理解梯度在机械上做什么

![slide-058：DPO 的更新构成——隐式奖励预测误差加权的 winner 上拉与 loser 下压](assets/slides/slide-058.jpg)

损失函数的形式优雅，但真正理解一个算法要看它的梯度。对单个偏好对，记 logit 差

$$
u_\theta
=
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.
$$

利用 $\frac{d}{du}\bigl[-\log\sigma(u)\bigr]=\sigma(u)-1=-\sigma(-u)$ 与链式法则（$\pi_{\mathrm{ref}}$ 是常数，梯度为零），逐步展开：

$$
\begin{aligned}
\nabla_\theta\mathcal L_{\mathrm{DPO}}
&=
\nabla_\theta\bigl[-\log\sigma(u_\theta)\bigr]
=
-\sigma(-u_\theta)\,\nabla_\theta u_\theta\\[2pt]
&=
-\sigma\!\left(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\right)
\cdot
\beta\left(
\nabla_\theta\log\pi_\theta(y_w\mid x)
-
\nabla_\theta\log\pi_\theta(y_l\mid x)
\right).
\end{aligned}
$$

对数据集取期望即得课件中的形式：

$$
\nabla_\theta\mathcal L_{\mathrm{DPO}}
=
-\beta\,\mathbb E
\left[
\underbrace{\sigma\!\left(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\right)}_{\text{隐式奖励估计越错，权重越大}}
\left(
\underbrace{\nabla_\theta\log\pi_\theta(y_w\mid x)}_{\text{提高 winner 似然}}
-
\underbrace{\nabla_\theta\log\pi_\theta(y_l\mid x)}_{\text{降低 loser 似然}}
\right)
\right].
$$

- $\nabla_\theta\mathcal L_{\mathrm{DPO}}$：DPO loss 对策略参数的梯度。
- $\beta$：整体更新尺度。
- $\sigma\bigl(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\bigr)$：sigmoid 权重——当隐式奖励把 loser 排在 winner 之前（排序错误）时趋近 1，排序正确且差距大时趋近 0。
- $\hat r_\theta$：由策略相对 reference 的 log-ratio 定义的隐式奖励估计。
- $\nabla_\theta\log\pi_\theta(y\mid x)$：回答的条件 log-probability 梯度。

机械上，这个更新同时做三件事：上拉 winner 的 log-probability、下压 loser 的 log-probability、按隐式奖励模型当前的"预测误差"自适应加权。如果模型已经把 winner 明显排在 loser 前面，这一 pair 的更新几乎为零；如果两者概率接近或顺序颠倒，更新达到满权重。这比"chosen 做 SFT、rejected 做负 SFT"更准确，因为两侧更新不是机械等权——误差大的 pair 主导梯度，误差小的 pair 自动退火。

下面用玩具 logit 表手算一遍 DPO 损失与梯度方向，验证上述机制：

```python
import torch

beta = 0.5
# 玩具词表大小 4；winner 与 loser 各为单 token 序列，便于手算
# policy 与 reference 的 logits
pi_logits  = torch.tensor([2.0, 1.0, 0.5, 0.0], requires_grad=True)
ref_logits = torch.tensor([1.5, 1.5, 0.5, 0.5])
y_w, y_l = 0, 1  # winner 是 token 0，loser 是 token 1

logp_pi  = torch.log_softmax(pi_logits, dim=0)
logp_ref = torch.log_softmax(ref_logits, dim=0)
logratio_w = logp_pi[y_w] - logp_ref[y_w]   # beta * log pi/ref for winner
logratio_l = logp_pi[y_l] - logp_ref[y_l]
u = beta * (logratio_w - logratio_l)
loss = -torch.nn.functional.logsigmoid(u)
loss.backward()
print(f"u = {u.item():.4f}, loss = {loss.item():.4f}")
# 权重 sigma(-u) 与梯度方向：token0 为负梯度(上拉), token1 为正梯度(下压)
w = torch.sigmoid(-u).item()
print(f"sigmoid weight = {w:.4f}")
print("d loss / d logits:", pi_logits.grad)
```

数值结果：$\log\pi_\theta(y_w)-\log\pi_{\mathrm{ref}}(y_w)\approx0.46$，$\log\pi_\theta(y_l)-\log\pi_{\mathrm{ref}}(y_l)\approx-0.54$，$u=0.5\times(0.46-(-0.54))=0.50$，loss $\approx0.474$，sigmoid 权重约 $0.38$——排序已正确但差距尚小，权重仍接近 $1/2$，梯度继续拉大 winner 与 loser 的 logits 差距；当 $u$ 增长到 $3$ 以上时，权重衰减到 $0.05$ 以下，该样本基本退出训练。这正是"按预测误差自适应加权"的数值体现。

### DPO 只是 primitive，完整系统仍可有外循环

![slide-059：LLaMA 及其他开放模型中的 DPO——DPO + expert iteration 后训练](assets/slides/slide-059.jpg)

本页展示 LLaMA 式开放模型的完整后训练回路，提醒读者 DPO 的核心优化器很像标准梯度训练，但完整系统仍可能采用 expert iteration：收集 prompt → 每个 prompt 生成 $K$ 个候选 → reward model 做 rejection sampling → 筛出的数据作为 SFT data 训练 SFT model → DPO training 得到 final DPO model → 最佳模型进入下一轮（best model for next round），同时 specialized per-capability 的 pairwise preference data 与 SFT data 持续回流用于 reward model training。也就是说，"DPO 不需要显式 reward model"描述的是 DPO 训练*原语*；它不排斥在更大的数据生成或 expert-iteration 外循环中使用 reward model——rejection sampling 造数据、DPO 消化数据，各司其职。

> [!NOTE]
> "DPO 不需要显式 reward model"描述的是 DPO 训练原语；它不排斥在更大的数据生成或 expert-iteration 外循环中使用 reward model。

### SimPO、长度归一化和经验脆弱性

![slide-060：DPO 变体——SimPO（无 reference）与长度归一化 DPO](assets/slides/slide-060.jpg)

本页给出 Tulu 3 论文实验中两个值得注意的变体。**SimPO** 去掉 reference model，用按序列长度归一化的平均 log-probability 之差再加一个目标 margin $\gamma$：

$$
\mathcal L_{\mathrm{SimPO}}(\pi_\theta)
=
-\mathbb E
\left[
\log\sigma\!\left(
\frac{\beta}{|y_w|}\log\pi_\theta(y_w\mid x)
-
\frac{\beta}{|y_l|}\log\pi_\theta(y_l\mid x)
-\gamma
\right)
\right].
$$

- $\mathcal L_{\mathrm{SimPO}}$：SimPO 损失。
- $|y_w|$ / $|y_l|$：winner / loser 回答的 token 长度。
- $\frac{1}{|y|}\log\pi_\theta(y\mid x)$：平均 per-token log-probability，即长度归一化后的隐式奖励。
- $\beta$：奖励差尺度。
- $\gamma$：目标 margin——要求 winner 的归一化对数概率不仅高于 loser，还要高出至少 $\gamma/\beta$ 的裕量。

**长度归一化 DPO** 则保留 reference，但对两侧 log-ratio 分别按长度归一化：

$$
\max_{\pi_\theta}
\mathbb E_{y_c,y_r\sim\mathcal D}
\left[
\log\sigma\!\left(
\frac{\beta}{|y_c|}\log\frac{\pi_\theta(y_c\mid x)}{\pi_{\mathrm{ref}}(y_c\mid x)}
-
\frac{\beta}{|y_r|}\log\frac{\pi_\theta(y_r\mid x)}{\pi_{\mathrm{ref}}(y_r\mid x)}
\right)
\right].
$$

两者的动机相同：序列级 log-probability 是逐 token 对数概率之和，天然随长度下降，于是 DPO 的隐式奖励会系统性惩罚长回答；而 slide-017 与 slide-048 已经表明评审又偏好长回答——长度成为偏好分数中的混杂变量。除以 $|y|$ 把比较单位从"整段回答的总对数概率"换成"平均每 token 的对数概率"，长度本身不再直接贡献奖励差。SimPO 进一步发现：有了长度归一化，隐式奖励不再需要相对 reference 的 log-ratio 来校准尺度，于是可以彻底删掉 reference model，省下一半前向计算与显存；margin $\gamma$ 则替代 KL 约束提供正则——它要求偏好裕量达到一定阈值才停止更新，防止模型对噪声 pair 过拟合。

经验脆弱性必须讲清楚。去掉 reference 后，没有任何机制阻止 $\pi_\theta$ 漂离预训练分布：长度归一化只校正了尺度，不约束方向；margin $\gamma$ 只控制何时停止放大偏好差距，不限制模型整体的分布漂移。实践中 SimPO 对学习率、$\beta$、$\gamma$ 的组合相当敏感，训练过长会导致生成质量崩坏（重复、格式退化），这些失败模式在 DPO 中被 reference 的锚定作用部分抑制。变体之间的差异是真实工程权衡：省去的每个组件都以某种稳定性为代价。

![slide-061：但 PPO 也行（有时甚至更好？）——RL 实证工作的条件依赖性](assets/slides/slide-061.jpg)

本页是讲者刻意安排的去魅时刻：标题 "But PPO does too (and sometimes better?)"，副标题 "the trickiness of RL-related empirical work"——大量结果高度依赖于实验设置的具体细节。左侧是 Ivison et al.（"Unpacking DPO and PPO"）的柱状图：从 initial SFT 的 56.8 出发，DPO 配弱偏好数据只有 58.1，DPO 配更好偏好数据 61.0，而 DPO 加一轮 PPO 达 62.2，PPO 配更大 reward model 62.8，PPO 配混合 prompt 62.4——最好的结果全部来自 PPO 变体，但注意每一步提升的主要因素似乎是*数据与 reward model 质量*而非算法本身。右侧是 Tulu 3 的消融表：SimPO 51.8–52.9、DPO 55.2、PPO 54.5–55.5、DPO-norm 在不同 $\beta$ 下从 46.8 到 57.3 剧烈波动——同一算法换个 $\beta$ 或 epoch 数，排名就翻转。讲者不认为"DPO 一定优于 PPO"或反过来是一个脱离实验设置的定律。更稳健的认识是：这类方法都接近"相对地上拉好回答、下压坏回答"的有效核心机制，普通开放模型场景中 DPO 往往已经足够好；而当有高质量 reward model 和工程资源时，PPO 的上限可能更高。

### 本章小结

- DPO 用策略相对 reference 的 log-ratio 表示隐式奖励；非参数最优策略是 reference policy 的 reward 指数倾斜 $\pi^*\propto\pi_{\mathrm{ref}}\exp(r/\beta)$，由把目标改写为 $-\beta D_{\mathrm{KL}}(\pi\|\pi^*)+\text{常数}$ 严格解出。
- 将隐式奖励代入 Bradley-Terry 损失后，配分函数 $Z(x)$ 在 winner/loser 差值中精确抵消，得到纯监督的 DPO loss。
- 梯度机械上是"提高 winner、降低 loser、按隐式奖励预测误差自适应加权"；排序已正确的 pair 自动退火。
- DPO primitive 简单，但完整后训练管线仍可能包含生成、筛选和 expert iteration。
- SimPO 与长度归一化 DPO 试图消除长度混杂；去 reference 以稳定性为代价换取效率。DPO 与 PPO 的优劣高度依赖实验设置，没有脱离数据的定律。

## RLHF 的边界：过优化、模式坍缩与失校准

![slide-062：RLHF 中需要警惕的两类问题——奖励过优化与模式坍缩/熵下降](assets/slides/slide-062.jpg)

本页把最后一节的两大风险并置。左图（Gao et al. 的 scaling laws for reward model overoptimization）：横轴是 RL 后策略与初始策略的 KL 距离，纵轴是 RM score；虚线（proxy reward）随 KL 单调上升，实线（gold reward）则在不同 reward model 尺寸下全部先升后降——RM 越大，拐点越晚、峰值越高，但没有任何尺寸的 RM 能免去过优化。右图是 52B 模型在 MMLU 上的 RLHF 校准曲线：$T=1$ 时模型概率与经验频率严重背离（自信但不一定正确），把温度调到 2.5 后才贴近对角线——RLHF 后模型的原始输出概率已不再具有频率含义。两张图预告：过优化侵蚀的是"reward 信号的真实性"，模式坍缩与失校准侵蚀的是"输出分布的统计含义"。

### Proxy reward 会被优化器反向利用

![slide-063：过优化——跨多种 RLHF 优化器，优化 proxy reward 超过某点后必然过拟合](assets/slides/slide-063.jpg)

本页给出过优化现象最系统的证据（同样来自 Gao et al.），三个子图横跨三种评测协议与三种优化器（expert iteration、best-of-n、PPO）：(a) 用人类偏好评测（$p_{\mathrm{human}}$）：三种优化器的真实胜率全部随 proxy reward 先升后降，PPO 在 proxy 约 1.3 处见顶后急转直下；(b) 用带噪模拟评审评测（AlpacaFarm 式）：同样先升后降；(c) 用无噪 single-prompt GPT-4 评审评测：曲线在所测范围内近似单调上升。结论写成页底一行：对 human pref（左）与 noisy LM pref（中）成立，对 noiseless LM pref（右）不成立。机制一目了然：reward model 只是人类偏好的近似代理，其中含有拟合误差与系统性偏差；优化强度较低时，策略主要利用 proxy 中与真实偏好一致的分量，二者同升；超过某个 KL 预算后，继续榨取 proxy 收益只能靠利用 proxy 的误差分量——于是 proxy 越高、真实越差。这也是为什么 KL regularization 不是装饰：优化器越强，越需要限制策略走出 reward model 可靠的分布区域。

> [!WARNING]
> 右图的 noiseless 条件是特殊实验设置（单 prompt、无标注噪声的 GPT-4 评审），不能概括为"模型评审 reward 永远不会被过优化"。关键变量是 reward 中是否存在噪声和可利用误差；真实部署中的模型评审同样带有长度偏好等系统偏差，过优化照样会发生。

### 模式坍缩与概率失校准

![slide-064：模式坍缩——RLHF 使模型不再是"概率模型"，默认失去校准](assets/slides/slide-064.jpg)

本页汇总模式坍缩与失校准的三组证据。左上是 52B 模型 MMLU 的 RLHF 校准曲线（同 slide-062）：$T=1$ 时概率明显偏离频率，需要人为升温到 2.5 才恢复对角线——模型输出 0.9 概率的选项，实际正确率可能只有 0.7。右上是 GPT-4 报告中的校准对比：预训练模型的校准曲线几乎贴合对角线（accuracy 0.82，ECE 0.007），PPO 后的模型则严重偏离（accuracy 0.78，ECE 0.014 且中间概率段大面积平坦）——RLHF 把本来接近完美校准的预训练分布压得失真。下方是 Santurkar et al. 的熵分布直方图：人类回答的熵分布在较宽区间，各代 davinci 模型逐代向低熵移动，text-davinci-003 在熵接近 0 处出现巨大尖峰——大量回答变得几乎确定、千篇一律。

把机制讲透：生成建模（预训练与 SFT）试图拟合带内在多样性的分布，其最优解要求概率质量按数据频率铺开，校准是 MLE 的副产品；奖励优化只要求找到高分回答，其最优解是 delta 分布——KL 正则只是延缓这一集中，不改变优化压力的方向。RLHF 因而容易降低输出熵，让概率质量集中到少数模式。两个概念相关但不同：

- **Mode collapse**：输出多样性和熵下降，许多可行回答消失——分布的*支撑集*收缩。
- **Miscalibration**：模型概率与经验正确率不匹配——分布的*数值*失去频率含义。

这一问题不只影响 confidence 的可读性。下一讲讨论 RLVR 时，模型必须探索多种潜在解法才能找到新的成功轨迹；熵过低会直接限制探索——一个已经坍缩到单一解法的模型，在可验证奖励下也没有样本多样性可供筛选。

### 讲者的实质性结语

![slide-065：本讲回顾——数据难、算法复杂、警惕过优化](assets/slides/slide-065.jpg)

讲者最后留下三条总结。第一，RLHF 数据采集同样困难，而且充满混杂因素：指南设计、标注者构成、专业度、长度与格式偏好，全部会渗入标签。第二，RLHF 算法比 SFT 更复杂，PPO 尤其难；课程作业会采用更简单的 GRPO。第三，必须警惕对 reward 的过优化：proxy 上升不代表真实目标上升，KL 预算、外部评测与多样性监控必须配套。结尾没有把 RLVR 当作已经证明的万能解，而是提出下一讲的问题：能否找到更不容易被过优化、可验证、噪声更低的 reward，使增加 compute 能更稳定地带来进步？这也是 RLVR 近年来影响力巨大的原因之一。

### 本章小结

- Learned reward 是 proxy，不是最终目标；优化足够强时会利用它的误差——真实偏好先升后降，拐点位置由 reward model 质量决定。
- RLHF 可能降低熵、造成模式坍缩，并破坏概率校准；预训练模型近乎完美的校准在 PPO 后显著失真。
- 探索不足会影响后续 reasoning/RLVR 找到新解法——坍缩的分布没有多样性可供可验证奖励筛选。
- 下一步问题不是"怎样无限优化任何 reward"，而是"什么 reward 足够可靠，值得持续投入计算"。

## 总结与延伸

### 一条统一主线

整堂课可以压缩成一个控制问题：预训练给模型一个广阔但难以直接使用的行为分布，后训练逐步把它变成可部署的 policy。

1. **SFT 选择模式**：用 demonstration 教模型以期望的语气、格式和协议回答；它最擅长抽取预训练已有的行为，而非注入新知识。
2. **Midtraining 扩大规模**：在预训练尾部提高高质量、指令型和合成数据比例，以训练日程换取规模与抗遗忘。
3. **Preference data 定义目标**：通过指南、标注者和比较任务，把"好回答"操作化；generation-verification gap 使比较比生成更可靠。
4. **PPO 或 DPO 优化目标**：提高偏好奖励，同时用 reference policy 抑制无约束漂移；二者的推导核心在 DPO 一节被统一为同一 KL 正则目标。
5. **风险控制**：检查 reward hacking、长度偏差、过优化、熵下降和失校准。

### SFT、PPO 与 DPO 的概念对照

| 维度 | SFT | PPO-RLHF | DPO |
|---|---|---|---|
| 直接训练信号 | 完整参考回答 token | Reward model + rollout | Winner/loser 偏好对 |
| 核心目标 | 拟合参考分布 | 最大化 reward 并限制漂移 | 提高相对偏好概率 |
| 是否需要显式 reward model | 否 | 是 | 核心训练不需要 |
| 是否需要 on-policy rollout | 否 | 是 | 否 |
| 主要优点 | 简单稳定 | 能直接优化可测目标、上限可能更高 | 工程简单、离线训练方便 |
| 主要风险 | 模仿错误、知识与格式纠缠 | 系统复杂、reward hacking | 依赖偏好数据与 reference，仍会过拟合偏好 |

### 实践中的四个判断

- 如果目标行为已在强底座中存在，少量高质量 SFT 往往足以 steering（约 500 条安全数据即可显著改变粗粒度行为）。
- 如果目标是注入模型原本没有的专家知识，仅靠合成数据或未知事实 SFT 不可靠，需要真实专家和验证；在未知事实上训练会同时提高训练准确率和幻觉率。
- 如果评价指标容易被长度、格式或其他代理变量操纵，先修正评审和数据，再选择更强优化器——更强的优化器只会更快地撞上过优化拐点。
- 如果采用 DPO、PPO 或其变体，必须把 KL、熵、校准和真实外部评测一起监控，不能只看 proxy reward。

### 开放问题

- 如何可靠判断某种行为或知识是否已存在于预训练模型中（从而区分"抽取"与"注入"）？
- 如何设计既能覆盖长尾安全场景、又不过度误拒的持续数据管线？
- 如何区分"更详细"与"仅仅更长"，并建立不易被 reward hacking 的评价？
- 如何在偏好优化后恢复或保留多样性与概率校准？
- 可验证 reward 能否真正支持随 compute 增加而近似单调改进？

> [!IMPORTANT]
> 本讲最值得保留的不是"哪一种后训练算法最好"，而是一套诊断框架：先问数据在教什么，再问评价者在奖励什么，最后问优化器会怎样利用这些信号的漏洞。

### 本章小结

后训练把语言模型从"会续写的概率模型"变成"按目标行动的策略"。SFT、midtraining、RLHF、PPO 与 DPO 分别解决控制、规模、偏好和优化问题，但每一步都可能把数据与评价中的偏差写入模型。完整 pipeline 因此必须同时设计数据、反馈、优化约束和外部验证，而不能把任何单一算法当作秘密武器。




