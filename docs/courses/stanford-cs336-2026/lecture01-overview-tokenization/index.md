# Stanford CS336 2026 Lecture 1：Overview, Tokenization

![课程封面](assets/cover.jpg)

- **视频标题**：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 1: Overview, Tokenization
- **主讲 / 频道**：Stanford Online
- **视频链接**：<https://www.youtube.com/watch?v=JuoVZkPBiKk>
- **时长**：01:19:21
- **材料范围**：人工英文字幕、1080p 课程视频与官方 `lecture_01.py`
- **课程定位**：先解释为何要从零构建语言模型、全课五个单元如何围绕效率联结，再以 Tokenization 作为第一个完整技术案例

> [!IMPORTANT]
> 这堂课的统一问题不是“最新模型用了什么技巧”，而是：**给定数据与计算预算，怎样构造效果最好的模型？** 课程把答案拆成可迁移的 mechanics、mindset 与需要警惕尺度变化的 intuitions；Tokenization 则把这套思路第一次落到可运行代码上。

## 1. 为什么要从零构建语言模型

### 1.1 抽象提高生产力，也会遮住研究对象

过去十年，使用语言模型的抽象层不断升高：早期研究者自己实现和训练模型，后来下载 BERT 等预训练权重微调，如今大量工作直接调用 GPT、Claude 或 Gemini 一类模型。抽象让应用开发更快，但语言模型是一种 **leaky abstraction**：当问题涉及训练数据、模型结构、优化器或并行方式时，API 无法替代对底层机制的理解。

因此课程选择“通过构建来理解”。这里的“from scratch”不是复刻一个闭源前沿模型，而是亲手完成一条规模较小但结构完整的链路：

```text
raw text
  → tokenizer
  → Transformer
  → loss / optimizer / training loop
  → kernels / parallelism
  → scaling recipe
  → data pipeline
  → alignment
```

讲者强调，课程不会真的从晶体管开始：PyTorch、GPU 和集群仍然是已有抽象。关键在于把抽象下钻到足以回答研究问题的层级，并让代码、测量与讲义彼此对应。

### 1.2 工业化造成尺度与透明度鸿沟

现代前沿模型的训练已经工业化。讲者把 GPT-4 约一亿美元的成本称作“据称”，把当前约十亿美元量级明确标为推测；这些数字的作用是提示数量级，而不是提供可审计的成本表。更棘手的是，前沿模型通常不公开完整架构、规模、硬件、训练计算、数据构造和训练方法。

![GPT-4 技术报告未披露的训练细节](assets/gpt4-no-details.png)
*图 1：官方课件截取 GPT-4 技术报告对未公开项目的说明。这里能支持“关键训练细节不透明”，不能扩写成“公众对模型一无所知”。（字幕区间：00:05:18--00:05:33）*

课堂里的模型小于 1B 参数，而产业模型大得多。为什么小模型结论不能机械外推？第一个例子是计算构成会变：在课件引用的 OPT 分析中，较小模型的 MLP/FFN 只占约 44% FLOPs，到 175B 时上升到约 80%。在小模型上极力优化 attention，不一定能给大模型带来同等收益。

![模型规模改变计算构成](assets/roller-flops.png)
*图 2：OPT 不同规模中 embedding、attention、FFN 等部分的 FLOPs 占比；规模增长后优化重点发生变化。（字幕区间：00:06:01--00:06:33）*

第二个例子是行为曲线可能非线性。若某项 zero-shot 或 few-shot 能力只在更大训练 FLOPs 后明显出现，小模型实验就观察不到它。图中“突然上升”是经验现象，不自动证明存在严格的物理相变。

![部分任务随训练 FLOPs 出现非线性提升](assets/wei-emergence-plot.png)
*图 3：八项任务的性能随训练 FLOPs 变化；它提醒我们，小尺度无提升不等于大尺度仍无提升。（字幕区间：00:06:33--00:07:09）*

### 1.3 三类知识的可迁移边界

课程把能学到的知识分成三类：

- **Mechanics**：Transformer、优化器、模型并行、kernel 如何工作。只要实现和数学没有变，通常能跨尺度迁移。
- **Mindset**：profile、benchmark、认真计算资源账单，并持续追问单位资源是否用在最有价值的地方。它也能跨尺度迁移。
- **Intuitions**：某种数据、结构或超参数“应该更好”的经验判断。它最容易随模型、数据和计算规模改变。

SwiGLU 论文的幽默结论正好展示第三类知识的局限：实验显示结构有效，却没有完整理论解释。

![SwiGLU 论文中的 divine benevolence](assets/divine-benevolence.png)
*图 4：实验发现可以先于完整解释；这不是说理论无用，而是要求研究者诚实地区分测量结果与因果解释。（字幕区间：00:08:49--00:09:15）*

### 1.4 Bitter Lesson 的正确落点是可扩展算法

常见误读是“规模决定一切，算法不重要”。课程给出的解释恰好相反：重要的是 **能随规模继续工作的算法**。讲者用一句纲领式关系表达：

$$
\text{accuracy}\approx \text{efficiency}\times \text{resources}.
$$

- `accuracy`：这里泛指模型最终效果，并非单一分类准确率。
- `efficiency`：单位数据、计算、内存或通信带来的有效进展。
- `resources`：可用数据与硬件预算。

这不是量纲严格的物理定律，而是研究 framing。在小实验里慢两倍可能只是多等一会儿；在昂贵训练中，5% 的改进就可能意味着巨额节省。课程最终问题因此是：**给定数据与计算预算，能构建怎样的最佳模型？**

### 本章小结

- API 抽象很有价值，但基础研究常需要进入训练与系统内部。
- 小模型能教 mechanics 与 mindset，却不能保证所有经验直觉外推到大模型。
- 规模会改变计算构成，也可能改变可观察行为。
- Bitter Lesson 强调可扩展算法；效率不是附属优化，而是模型能力的一部分。

## 2. 技术谱系、开放生态与课程方法

### 2.1 从 N-gram 到 agent：变化的是能力界面

课程用一条历史线把许多论文放到同一问题里：

1. Shannon 熵与 N-gram 建立统计语言建模基础。
2. LSTM、seq2seq、Adam、attention、Transformer、MoE 和模型并行提供神经网络组件。
3. ELMo、BERT、T5 让预训练模型成为通用表示或任务起点。
4. GPT-2、scaling laws、GPT-3、PaLM 与 Chinchilla 把规模、in-context learning 和 compute-optimal 训练推到中心。
5. ChatGPT 把语言模型变成对话界面；更长上下文与工具调用进一步把它变成 agent。

这个历史不是“旧模型都被淘汰”。模型的使用规格变了，但 attention、kernel、optimization、data processing 等 fundamentals 仍然贯穿其中。

### 2.2 Open-weight 不等于 fully open

开放生态至少有不同层级：

| 开放内容 | 能回答的问题 | 仍缺什么 |
|---|---|---|
| 只有 API | 模型会输出什么 | 权重、训练与数据细节 |
| 权重 + 论文 | 可部署、可测量部分结构 | 完整训练代码与数据配方 |
| 权重 + 论文 + 代码 + 数据 | 更接近可复现与可审计 | 仍可能缺集群环境、数据混合细节 |

讲者指出，Llama、Mistral、DeepSeek、Qwen、Kimi、GLM、MiniMax、MIMO 等 open-weight 系列正接近闭源模型；OLMo、Nemotron 与 Marin 等项目则尝试进一步开放代码和数据。这里的判断是授课时点的生态概览，不应把“approaching”理解成每个模型在每项任务都追平。

### 2.3 Executable lecture：讲义也是程序

官方课件本身是可执行 Python，而非静态幻灯片。例如：

```python
total = 0
for x in [1, 2, 3]:
    total += x
```

运行时会展示代码、变量值和层级结构。这个形式与课程方法一致：不要把实现当作不可修改的截图；应该运行它、检查中间状态、测量速度并验证不变量。

### 2.4 五项作业是一条完整流水线

五项作业依次对应 Basics、Systems、Scaling Laws、Data、Alignment。作业不给大量 scaffolding code，但提供单元测试和 adapter interface：先在本地证明正确，再到集群上测 accuracy 与 speed。Leaderboard 不是单纯竞速；不同任务会在固定预算下考察 perplexity、准确率或吞吐。

课程允许 AI 做解释和 tutoring，但核心目标是形成自己的机制理解。若 coding agent 直接代写而学生无法解释代码，作业即使通过测试也没有完成学习目标。具体政策仍应以正式课程规则为准。

### 本章小结

- 语言模型的交互形态不断变化，但底层训练与系统机制有延续性。
- Open-weight、open code 与 open data 是不同承诺，不能混为一谈。
- Executable lecture 把“能运行、能检查”变成课程材料的组成部分。
- 五项作业沿完整 LM 栈展开，正确性与效率都必须被实测。

## 3. Basics：从 token 到可训练的 Transformer

### 3.1 三个组件与三项约束

Basics 的最小闭环是：tokenize 数据、定义模型、实现训练。课程并不把它们视为互不相关的模块，而是要求同时平衡：

- **Expressivity**：能否表示复杂依赖；
- **Stability**：激活、参数与梯度是否保持可训练；
- **Efficiency**：计算、内存和推理代价是否可承受。

Tokenization 决定模型看到的“原子”；架构决定信息如何交互；训练目标与优化器决定怎样从数据中更新参数。改变任何一项都会反馈到另外两项。

### 3.2 现代 Transformer 是一组连续的工程选择

原始 Transformer 只是起点。现代语言模型会在以下维度做选择：

- activation：ReLU、SwiGLU；
- position：绝对位置、sinusoidal、RoPE；
- normalization：LayerNorm、RMSNorm、QK norm、pre-norm / post-norm；
- attention：full、local/sparse、GQA、MLA；
- sequence model：attention、linear attention、Mamba、Gated DeltaNet 及混合结构；
- MLP：dense 或 MoE；
- shape：宽度、深度、head 数、expert 数。

![Decoder-only Transformer 总体结构](assets/transformer-architecture.png)
*图 5：token/position embedding 进入堆叠 block；每层由 causal self-attention、FFN、residual 与 normalization 组成，最后经线性层和 softmax 预测 token。（字幕区间：00:29:54--00:31:48）*

标准 self-attention 对序列长度 $L$ 的两两交互会形成 $L\times L$ 的 attention matrix，因此主要计算与存储常按 $O(L^2)$ 增长。Tokenization 若把 byte 序列压短，就会直接影响这里的成本。

### 3.3 训练稳定性不是“调几个参数”

课程列出的训练选择包括：

- next-token 或 multi-token prediction；
- AdamW、SOAP、Muon 等优化器；
- Xavier、$\mu$P 等初始化/参数化方法；
- cosine、WSD 等学习率日程；
- dropout、weight decay、batch size；
- MoE 的负载均衡。

这些选项表面上都是超参数，组合起来却可能决定训练是稳定收敛，还是直接发散。Assignment 1 因此要求完成 BPE、Transformer、loss、optimizer 和完整 training step，而不是只拼装现成模块。

### 本章小结

- Basics 是 tokenizer、模型结构与训练算法组成的闭环。
- 架构设计同时服务表达能力、稳定性和效率。
- attention 的序列长度代价解释了为什么 token compression 重要。
- “超参数”在大规模训练中可能决定整次运行是否可用。

## 4. Systems 与 Scaling Laws：把资源变成可预测的能力

### 4.1 先做资源账本：$C\approx6ND$

对 dense Transformer，课程使用训练 FLOPs 的近似：

$$
C\approx 6ND.
$$

- $C$：训练总浮点运算次数（FLOPs）。
- $N$：模型参数量。
- $D$：训练 token 数。
- 常数 6：把每个参数、每个 token 的前向与反向主要矩阵计算压缩成课程级估算。

例如 70B 参数训练 1T token：

```python
total_flops = 6 * 70e9 * 1e12  # 4.2e23 FLOPs
```

得到约 $4.2\times10^{23}$ FLOPs。它是做数量级规划的公式，不包含所有 kernel、稀疏结构、通信和硬件利用率细节。

### 4.2 峰值算力不等于实际速度

GPU 必须在 compute 与 memory 间搬运数据。课堂用 B200 举例：bf16 峰值约 2.25 PFLOP/s，内存带宽约 8 TB/s。若每次运算需要从 HBM 搬大量数据，计算单元会等待内存，程序成为 memory-bound。

![计算与内存带宽之间的瓶颈](assets/compute-memory.png)
*图 6：计算单元与显存之间的通路决定实际吞吐；硬件峰值只有在数据能及时送达时才可利用。（字幕区间：00:36:58--00:37:57）*

Systems 的统一原则是减少 data movement：

- fusion：一次读入后完成多项操作，再写回；
- tiling / FlashAttention：把适合的数据块留在更快的片上存储；
- parallelism：在 data、tensor、pipeline、sequence、expert 等维度切分；
- distributed collectives：用 gather、reduce、all-reduce 协调多 GPU 状态。

### 4.3 Prefill 与 decode 的瓶颈不同

推理有两个阶段：prefill 一次处理完整 prompt，建立各层 KV cache；decode 每次只生成一个新 token，并反复读取权重与缓存。前者更像训练，通常更容易使用并行计算；后者常受内存带宽和请求调度限制。

![Prefill、逐 token decode 与 KV cache](assets/prefill-decode.png)
*图 7：已知 prompt 可并行 prefill，后续 token 必须自回归生成；KV cache 避免重复计算历史状态。（字幕区间：00:42:35--00:43:03）*

可用 pruning、quantization、distillation、speculative decoding、专用 fused kernel 和 continuous batching 提速。Speculative decoding 的直觉是：便宜模型先猜一段 token，大模型并行核验；猜对时一次接受多个 token。

### 4.4 Scaling recipe：把预算映射为配置

当预算达到 $10^{24}$ 或 $10^{25}$ FLOPs 时，不可能在目标规模上反复调参。课程要求把思维从“训练一个模型”改成：

$$
\text{FLOP budget}
\longmapsto
\{N,D,\text{learning rate, batch size, shape, ...}\}.
$$

这就是 scaling recipe。研究者在较小预算上运行实验、拟合 loss 与规模关系，再预测目标规模。一个重要认识是：**predictability 至少和 optimality 同样重要。** 某个小规模点略优但无法稳定外推，可能不如一个稍逊却可预测的 recipe。

Scaling law 也不是自然界自动存在的定律。学习率、batch size、参数化和模型 shape 必须随规模以可预测方式变化；$\mu$P 等方法的价值之一，就是改善超参数跨尺度迁移。

### 4.5 Chinchilla：参数与 token 的预算分配

在固定 $C=6ND$ 下，多参数、少 token 与少参数、多 token 都可能浪费预算。Chinchilla 方法对每个 FLOPs budget 扫描不同模型大小，得到 U 形 ISOFLOP 曲线的最优点，再拟合最优 $N$ 与 $D$ 随预算的变化。

![Chinchilla ISOFLOP 曲线与外推](assets/chinchilla-isoflop.png)
*图 8：左图在各预算上寻找最低 loss 的参数量，中、右图分别拟合最优模型大小与训练 token 数。（字幕区间：00:49:36--00:50:50）*

课程给出粗略规则：

$$
D\approx20N.
$$

- $D$：训练 token 数。
- $N$：模型参数量。
- 例子：$N=70\text{B}$ 时，$D\approx1.4\text{T}$。

这只是经验 rule of thumb，会随数据和架构变化，并且没有纳入部署推理成本。若推理很贵，工程上可能偏向更小模型、更多训练 token，即便这不是纯训练 FLOPs 下的最优点。

### 本章小结

- $C\approx6ND$ 是规划 dense Transformer 训练数量级的近似账本。
- 实际速度取决于计算、内存和通信，而非只看峰值 FLOPs。
- prefill 与 decode 的并行性和瓶颈不同。
- Scaling recipe 用小实验决定大预算配置；可预测性本身是一项核心目标。
- $D\approx20N$ 是有边界的教学近似，不是跨模型永久成立的定律。

## 5. Data、Alignment 与全课的效率主线

### 5.1 先定义能力，再决定数据

“训练什么数据”不能脱离“希望模型会什么”。课程从 evaluation 开始：

- **Internal eval** 服务开发决策，重视小尺度趋势平滑、相对比较稳定；
- **External eval** 衡量真实用途，重视绝对质量和 ecological validity。

私有文档 perplexity、GPQA、HLE、SWE-Bench、Terminal-Bench 测量的能力不同，任何单一分数都不能完整代表模型。目标确定后，数据依次经历：

```text
curation
→ transformation
→ filtering
→ deduplication
→ mixing
→ rewriting / synthetic data
```

过滤的效率逻辑非常直接：固定 compute budget 下，在坏数据上花更多时间，就意味着在好数据上花更少时间。与此同时，训练者还必须处理许可、版权、隐私和数据污染，而不能把“公开可访问”自动视为“可随意训练”。

### 5.2 Alignment：评价往往比生成更容易

预训练给出 next-token 模型，但并不保证回答符合人类目标。Alignment 利用弱监督：模型先生成多个 response，再由 human、verifier 或 LM judge 给反馈，最后提高较好回答的概率。

```text
prompt
  → model responses
  → human / verifier / LM judge
  → preference or reward signal
  → PPO / GRPO / DPO update
```

PPO、GRPO 属于 RL 路线；DPO 可直接从 preference pairs 学习。RL 的困难不只在算法不稳定，还在 rollout 需要高吞吐推理系统；为了保持 on-policy，样本又不能无限复用。系统效率与统计目标在这里再次耦合。

### 5.3 五个单元其实在解同一道题

课程在进入 Tokenization 前再次把全课压缩成效率问题：

| 单元 | 主要资源瓶颈 | 核心效率动作 |
|---|---|---|
| Basics | 序列、参数、优化稳定性 | 选表达力足够且可训练的表示与架构 |
| Systems | HBM 与跨 GPU 搬运 | fusion、tiling、parallelism |
| Scaling | 大预算无法穷举 | 用小实验建立可外推 recipe |
| Data | 训练 token 价值不同 | 把 FLOPs 留给目标相关、高价值数据 |
| Alignment | rollout 昂贵且反馈稀疏 | 用弱监督与高效推理放大评价信号 |

这也解释了 Lecture 1 为什么最后详细讲 Tokenization：它同时影响数据表示、序列长度、attention 成本、模型容量分配和训练吞吐，是一个缩小版的全课程问题。

### 本章小结

- Evaluation 定义目标能力，数据 pipeline 才能围绕目标构造。
- 数据质量、合法性与污染都属于训练系统的一部分。
- Alignment 用“更容易评价”这一不对称性提供弱监督。
- Basics、Systems、Scaling、Data、Alignment 都在优化固定资源下的有效学习。

## 6. Tokenizer：在字符串与整数序列之间建立可逆接口

### 6.1 问题定义

语言模型对 token index 序列建模，而输入通常是 Unicode string。Tokenizer 至少要提供：

```python
class Tokenizer(ABC):
    def encode(self, string: str) -> list[int]: ...
    def decode(self, indices: list[int]) -> str: ...
```

最基本的正确性条件是 roundtrip：

$$
\operatorname{decode}(\operatorname{encode}(s))=s.
$$

- $s$：任意支持的输入字符串。
- `encode`：把字符串映射为 token ID 序列。
- `decode`：把 ID 序列还原为字符串。

![字符串、token 与整数 ID 的关系](assets/tokenized-example.png)
*图 9：文本先被分成可变边界的 token，再映射为整数；解码应恢复原字符串。（字幕区间：01:05:31--01:05:50）*

Token ID 的数值大小没有语义，只是 vocabulary index。Roundtrip 也只证明可逆，不证明序列短、词表合理或训练高效。

### 6.2 真实 tokenizer 的边界并不像“单词”

课堂使用 `tiktoken` 的 `o200k_base`，并称其为 GPT-5 tokenizer。对 `"Hello, 🌍! 你好!"` 的现场运行得到 8 个 token ID：

```text
[13225, 11, 130321, 235, 0, 220, 177519, 0]
```

随后解码回原字符串。这个例子展示：

- 前导空格常与后面的词合并成同一 token；
- 同一个词出现在句首和句中，可能是完全不同的 ID；
- 数字可能每若干位一组，不保证与人类十进制分组一致；
- emoji 或多语言字符不一定对应一个 token。

![o200k_base 的现场 roundtrip 结果](assets/tokenizer-roundtrip-runtime.jpg)
*图 10：现场变量检查显示 8 个 token ID，并成功还原多语言字符串。（字幕区间：01:07:00--01:07:25）*

### 6.3 Compression ratio 与 vocabulary trade-off

课程定义：

$$
\text{compression ratio}
=\frac{\text{input UTF-8 bytes}}{\text{number of tokens}}.
$$

- 分子：输入字符串编码为 UTF-8 后的 byte 数。
- 分母：token 数量。
- 比率越大：平均每个 token 承载更多 bytes，序列越短。

例子中有 20 bytes、8 tokens，所以比率是 $20/8=2.5$ bytes/token。实现为：

```python
def get_compression_ratio(string, indices):
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens
```

![压缩率与词表大小的现场检查](assets/tokenizer-compression-runtime.jpg)
*图 11：同一示例的 `compression_ratio=2.5`，`o200k_base` 的现场 `vocabulary_size=200019`；扩大词表可缩短序列，却会增加类别稀疏性。（字幕区间：01:07:25--01:08:25）*

更短序列可降低标准 attention 的二次成本，但不能无限扩大词表。更大的 vocabulary 会增加 embedding 和输出层参数，让大量 token 更稀疏、训练样本不足。现实 tokenizer 常有 100K 或 200K 词表，正是在序列长度与词表利用率之间折中。

### 本章小结

- Tokenizer 是 string 与 integer sequence 之间的可逆接口。
- Roundtrip 是必要条件，但不是效率或质量的充分条件。
- Token 边界由训练与规则决定，不等于词、字符或可见符号。
- Compression ratio 越高通常序列越短，但更大词表会带来参数与稀疏性成本。

## 7. 字符、字节与词：三种朴素方案为何都不够好

### 7.1 Character tokenizer：可逆但“大词表 + 长序列”

Python `str` 可按 Unicode code point 迭代，于是最直接的 tokenizer 是：

```python
class CharacterTokenizer(Tokenizer):
    def encode(self, string):
        return list(map(ord, string))

    def decode(self, indices):
        return "".join(map(chr, indices))
```

例如 `ord("a") == 97`，`ord("🌍") == 127757`。现场示例能 roundtrip，但观察到的最大 ID 推出的 `127758` 只是该示例所需词表的下界，不是 Unicode 字符总数。课程用约 150K characters 说明：词表很大，且许多 code point 极少出现；与此同时，一个 code point 通常仍占一个 token，压缩率并不高。

![Character tokenizer 的现场输出](assets/character-tokenizer-runtime.jpg)
*图 12：同一字符串被映射为 code point ID；现场下界 vocabulary size 为 127758，compression ratio 约 1.5385。（字幕区间：01:08:34--01:09:45）*

还要注意，Unicode code point 不总等于用户看到的 grapheme cluster。组合音标或某些 emoji 可能由多个 code point 组成。

### 7.2 Byte tokenizer：固定 256 词表但序列最长

UTF-8 把任意 Unicode string 编码为 0–255 的 bytes：

```python
bytes("a", encoding="utf-8")   # b"a"
bytes("🌍", encoding="utf-8")  # b"\xf0\x9f\x8c\x8d"
```

Byte tokenizer 的词表固定为 256，覆盖所有合法 UTF-8 文本，也没有 OOV。但一个 byte 就是一个 token，因此：

$$
\text{compression ratio}=1.
$$

![Byte tokenizer 的现场输出](assets/byte-tokenizer-runtime.jpg)
*图 13：emoji 和中文字符展开为多个 UTF-8 bytes；vocabulary size 固定为 256，但 compression ratio 恰为 1。（字幕区间：01:09:47--01:10:45）*

小词表不自动等于高效：序列更长会放大 attention 成本。课程认为端到端 byte-level 模型值得探索，但授课时点尚未扩展到前沿模型规模。

### 7.3 Word tokenizer：语义清楚却无法封闭词表

教学实现用正则切分：

```python
chunks = regex.findall(r"\w+|.", string)
```

连续字母数字形成 chunk，其余字符单独保留。词具有较稳定语义，示例 compression ratio 可达到 5.5；但词表等于训练集中的 distinct chunks，长尾会非常大。测试时遇到新词，只能映射为 `UNK`，信息被抹掉，还会扭曲 perplexity。

![Word tokenizer 的现场切分与失败模式](assets/word-tokenizer-runtime.jpg)
*图 14：教学正则给出可读 chunk，但新词必须落入 UNK，词表也没有自然上界。（字幕区间：01:10:45--01:11:58）*

这段正则只是教学例子，不等于现代 production tokenizer 的完整 pre-tokenization。

### 7.4 三种方案的共同矛盾

| 方案 | 词表 | 序列 | 未见文本 | 主要失败 |
|---|---:|---:|---|---|
| Character | 大且含大量稀有项 | 较长 | 通常可表示 | 词表利用率低 |
| Byte | 固定 256 | 最长 | 完全覆盖 | compression ratio = 1 |
| Word | 巨大、开放 | 较短 | 依赖 UNK | 长尾、OOV、概率失真 |

理想方案应同时做到：从 byte 获得全覆盖；让常见片段合并以缩短序列；遇到罕见片段仍能退回更小单元。这正是 BPE 的设计位置。

### 本章小结

- Character tokenizer 可逆，却浪费大词表并保留较长序列。
- Byte tokenizer 只有 256 个 token，但 compression ratio 为 1。
- Word tokenizer 压缩好，却有巨大长尾和不可避免的 OOV/UNK。
- 关键矛盾是全覆盖、词表大小与序列长度无法由固定粒度同时解决。

## 8. Byte Pair Encoding：用数据学习可变粒度 chunk

### 8.1 BPE 的核心思想

BPE 最初是数据压缩算法，后来进入神经机器翻译，再被 GPT-2 用于语言模型。它从 256 个单 byte token 开始，反复合并训练语料中最常见的相邻 pair：

- 常见序列逐步变成一个 token；
- 罕见序列保留为多个较小 token；
- 任意 UTF-8 文本仍能退回 bytes 表示，不需要 UNK。

训练参数可写成：

```python
@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]
```

- `vocab`：token ID 到 byte sequence 的映射。
- `merges`：旧 token pair 到新 token ID 的映射。
- merge 的顺序本身也是参数；交换应用顺序可能改变编码结果。

### 8.2 单次 merge 必须替换不重叠 pair

```python
def merge(indices, pair, new_index):
    new_indices = []
    i = 0
    while i < len(indices):
        if (
            i + 1 < len(indices)
            and indices[i] == pair[0]
            and indices[i + 1] == pair[1]
        ):
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices
```

匹配成功后 `i += 2`，表示两个旧 token 同时被消费。Pair 计数允许相邻窗口重叠，但一次替换不能重叠。例如 `[x,x,x]` 中 pair `(x,x)` 在计数窗口里出现两次，左到右 merge 一轮只能合并其中一对。

### 8.3 训练循环：统计、选择、建词、替换

课程用 `"the cat in the hat"` 做三轮 merge：

```python
indices = list(map(int, string.encode("utf-8")))
vocab = {x: bytes([x]) for x in range(256)}
merges = {}

for i in range(num_merges):
    counts = count_adjacent_pairs(indices)
    pair = max(counts, key=counts.get)
    new_index = 256 + i
    merges[pair] = new_index
    vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
    indices = merge(indices, pair, new_index)
```

第一轮 `(116,104)` 即 bytes `t,h` 出现两次，创建 token 256 表示 `b"th"`。第二轮将 `(256,101)` 合并为 257，即 `b"the"`。第三轮因同频 pair 的插入顺序选择 `(257,32)`，创建 258，表示 `b"the "`。课堂也明确指出有 ties；生产实现必须规定确定的 tie-breaking，不能依赖偶然字典顺序。

![BPE 第一轮 merge 后的现场状态](assets/bpe-first-merge-runtime.jpg)
*图 15：`th` 已由 256 代替；pair 统计仍显示多组同频候选，解释了 tie-breaking 为什么必须显式定义。（字幕区间：01:13:20--01:15:07）*

相邻 pair 的统计很直接：

```python
def count_adjacent_pairs(indices):
    counts = defaultdict(int)
    for index1, index2 in zip(indices, indices[1:]):
        counts[(index1, index2)] += 1
    return counts
```

每一轮都会让 vocabulary 增加 1，让训练序列缩短若干位置。这个 toy corpus 三轮后 compression ratio 为 1.5。

### 8.4 编码与解码

训练完成后，编码新文本从 UTF-8 bytes 开始，按训练顺序应用 merge：

```python
class BPETokenizer(Tokenizer):
    def encode(self, string):
        indices = list(map(int, string.encode("utf-8")))
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices):
        bytes_list = list(map(self.params.vocab.get, indices))
        return b"".join(bytes_list).decode("utf-8")
```

对 `"the quick brown fox"`，最初的 `the ` 被 token 258 表示，其余未学到的部分退回单 bytes，仍然可以完整解码。

![已训练 BPE 对新字符串编码](assets/bpe-encode-runtime.jpg)
*图 16：`the ` 被压成 ID 258，其余罕见片段仍由 byte ID 组成，体现“常见片段合并、罕见片段退化”的原则。（字幕区间：01:15:21--01:16:06）*

### 8.5 教学实现正确，但工程上极慢

课程代码会遍历所有 merge rules 并反复扫描整个序列。若 vocabulary 有 $V$ 项，规则数量约为 $V-256$；大词表下代价很高。Assignment 1 要求：

1. 只处理当前序列真正相关的 merges，并建立合适索引；
2. 识别并保护 `<|endoftext|>` 等 special token；
3. 使用 GPT-2 tokenizer regex 一类 pre-tokenization，先把长文本切成 chunk；
4. 尽可能优化速度，必要时可使用 Rust、C 等实现。

容易忽略的陷阱包括：

- `merges` 必须保留训练顺序；
- special token 不能当普通 byte sequence 被拆散；
- 任意 token ID 拼出的 bytes 未必是合法 UTF-8，只有合法 `encode` 输出可保证 roundtrip；
- BPE 优化频率与序列长度，不保证边界符合人类语义；
- pre-tokenization 不只是加速，也限制不希望跨越的 merge 边界。

### 8.6 BPE 是有效启发式，不是终点

![课程对 Tokenization 的最终要求](assets/tokenization-final-properties.jpg)
*图 17：即使未来端到端从 bytes 学习，模型仍需要序列抽象，并让 chunk 粒度随信息内容变化。（字幕区间：01:17:31--01:18:58）*

课程对未来方案提出两个比“必须使用 BPE”更本质的要求：

1. Transformer 应在序列的 chunks / abstractions 上工作，而不是平等对待每一个低层 byte；
2. chunk 应可变，把更多模型容量分配给有信息、难预测或“有趣”的位置。

视频、DNA 等序列更能说明这一点：原始单位可能低信噪比，建模前需要抽象；不同区段又不应获得相同计算。BPE 用频率启发式近似这种 variable computation，但未来可以由端到端模型学得更动态的分块。

### 本章小结

- BPE 从单 bytes 出发，通过频繁相邻 pair 的迭代合并学习词表。
- 常见序列变成单 token，罕见序列退回小单元，因此同时获得压缩与全覆盖。
- Merge order、tie-breaking、不重叠替换和 special token 都是正确性要求。
- 朴素实现虽然完整，却需要索引、pre-tokenization 与底层优化才能实用。
- BPE 是数据驱动的有效启发式；更长期目标是可学习、可变粒度的序列抽象。

## 总结与延伸

Lecture 1 建立了一条贯穿全课的因果链：

```text
前沿模型规模巨大且细节不透明
        ↓
不能只复制结果，要学习可迁移的机制与方法
        ↓
固定资源下，效率决定能做多少有效学习
        ↓
Basics / Systems / Scaling / Data / Alignment
分别优化表示、搬运、决策、样本与反馈
        ↓
Tokenization 成为第一个完整案例
```

真正值得保留的结论有四层：

1. **研究方法**：构建、运行、profile、benchmark，比只记住某个模型名称更可迁移。
2. **尺度意识**：小模型适合验证 mechanics，却可能给出错误的大模型 intuition。
3. **资源意识**：计算、内存、通信和数据都有限；局部算法改进只有进入完整资源账本才有意义。
4. **表示意识**：Tokenizer 不是文本预处理的边角步骤，它决定序列长度、词表容量、计算分配和可表达范围。

沿这条线继续学习时，可以带着三个检查问题：

- 这个结论是可证明的 mechanics，经过测量的经验，还是可能随规模变化的 intuition？
- 它节省的是 FLOPs、memory bytes、通信、训练 token，还是推理 latency？
- 局部优化是否在新的规模、数据分布和部署目标下仍然成立？

下一讲从 resource accounting 开始，把这里的效率 framing 进一步变成可计算的 Transformer 资源账本。

### 本章小结

- 本讲用 tokenization 建立了“机制理解、资源核算与可执行验证”这条全课主线。
- Tokenizer 同时影响表示粒度、序列长度、词表容量与后续 Transformer 成本。
- 学习任何规模化结论时，都应同时检查事实边界、资源口径和尺度迁移条件。
