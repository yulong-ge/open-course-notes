# CS336 2026 Lecture 4：Attention Alternatives 与稀疏专家模型

![视频封面](assets/cover.jpg)

- **原视频标题**：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 4: Attention Alternatives
- **讲者**：Tatsu Hashimoto
- **频道**：Stanford Online
- **视频链接**：<https://www.youtube.com/watch?v=cKSwj_qZ8Jg>
- **时长**：01:26:20
- **资料范围**：人工英文字幕、完整 1080p 视频、60 页官方课件

这节课研究一个贯穿现代大模型设计的问题：**模型怎样在容量继续增长时，不让每个 token 的计算与内存访问同步爆炸？** 前半场从 attention 的二次成本出发，依次讨论线性 attention、循环状态模型、Mamba-2、Gated Delta Net（GDN）、混合架构和 DeepSeek Sparse Attention（DSA）；后半场转向 Mixture of Experts（MoE），解释路由、负载均衡、并行通信、数值稳定性、微调、upcycling，以及 DeepSeek V1 到 V3 的演化。

两条主线表面上不同，实质上都属于 **conditional computation**：attention alternatives 决定“从哪些历史状态读取”，MoE 决定“激活哪些参数”。

## 1. 为什么需要 Attention Alternatives

### 1.1 阅读前需要的四个概念

- **Attention**：query 与一组 keys 计算相似度，再用所得权重聚合 values；因果语言模型只能读取当前位置及其之前的 tokens。
- **FFN**：Transformer block 中对每个 token 独立应用的前馈网络，通常占据大量参数和计算。
- **KV cache**：自回归推理时缓存历史 tokens 的 keys/values，避免每一步重新计算；上下文越长，缓存越大。
- **FLOPs 与带宽**：FLOPs 衡量算术工作量，带宽衡量数据搬运速度。理论乘法次数更少，不代表在真实 GPU 上必然更快。

### 1.2 Softmax attention 到底算了什么

先固定一个 query 位置 `t`。它与每个 key 位置 `s` 做缩放内积，加入 causal mask，再沿 **key 位置这一维** 做 softmax；最后用所得概率加权 values：

$$
\begin{aligned}
z_{t,s}&=\frac{q_t^\top k_s}{\sqrt{d_k}}+m_{t,s},\\
m_{t,s}&=\begin{cases}0,&s\le t,\\-\infty,&s>t,\end{cases}\\
a_{t,s}&=\frac{\exp(z_{t,s})}{\sum_{r=1}^{n}\exp(z_{t,r})},\qquad
y_t=\sum_{s=1}^{n}a_{t,s}v_s.
\end{aligned}
$$

- `t`：当前 query 的位置。
- `s,r`：被读取的 key/value 位置索引。
- `n`：序列长度。
- `q_t`：位置 `t` 的 query 向量。
- `k_s`：位置 `s` 的 key 向量。
- `v_s`：位置 `s` 的 value 向量。
- `d_k`：query/key 维度；平方根缩放用于控制 logit 尺度。
- `m_{t,s}`：causal mask；未来位置取负无穷。
- `z_{t,s}`：加入缩放与 mask 后的 attention logit。
- `a_{t,s}`：位置 `t` 分给位置 `s` 的 attention 权重。
- `y_t`：位置 `t` 聚合后的 attention 输出。
- `exp`：指数函数。
- `T`：向量转置。

做一个可手算的例子。假设序列有 3 个位置，当前 query 位于 `t=2`；即使未来位置 3 的原始 logit 高达 99，causal mask 也会把它完全排除：

$$
\begin{aligned}
z_{2,:}&=[1,2,99]+[0,0,-\infty]=[1,2,-\infty],\\
a_{2,:}&=\operatorname{softmax}(z_{2,:})\approx[0.269,0.731,0],\\
[v_1,v_2,v_3]&=[10,0,-10]\quad\Longrightarrow\quad y_2\approx0.269\times10+0.731\times0+0\times(-10)=2.69.
\end{aligned}
$$

- `z_{2,:}`：query 2 对全部三个 key 位置的 masked logits。
- `a_{2,:}`：沿 key 位置维归一化后的权重，三项之和为 1。
- `v_1,v_2,v_3`：为便于手算而设为标量的三个 values。
- `y_2`：query 2 的加权输出。
- `softmax`：对同一个 query 的全部允许 key logits 做指数归一化。
- `∞`：无穷大；`exp(-∞)=0`。
- `≈`：数值近似。

在实际多头实现中，attention logits 常写成 `[B,H,T_q,T_k]`；softmax 沿最后的 `T_k` 维执行。它不会跨 batch、head 或不同 query 相互归一化。训练时 causal mask 阻止看到未来 token；自回归推理时，KV cache 中本来就只有已经生成的历史位置。

> [!IMPORTANT]
> softmax 同时做两件事：把 logits 变成非负且和为 1 的读取权重，并让每个 query 的权重依赖同一行全部 keys。后一个全局耦合正是简单结合律不能直接穿过 softmax 的原因。

### 1.3 长上下文首先是系统问题

标准自注意力让每个 query 与所有历史 key 比较。序列长度从 `n` 增长到 `2n` 时，注意力矩阵面积约增至四倍；KV cache、显存带宽和 kernel 调度也随之变成瓶颈。长上下文并不只是“把 position embedding 拉长”，而是要重新安排：

- 哪些历史 token 值得被读取；
- 是否必须保留所有历史 token 的显式表示；
- 训练时的并行性与推理时的逐 token 延迟如何兼顾；
- 理论 FLOPs 的改善能否转化成真实硬件收益。

![上下文长度增长带来的 attention 成本](assets/attention-cost.jpg)

*图：上下文窗口与 attention/FFN 成本的增长趋势。官方课件第 2 页；视频对应讲解区间：`00:01:35--00:02:49`。*

讲者先给出两个直观基线。**局部 attention** 只看固定窗口，成本可控但可能漏掉远距离依赖；**少量全局层 + 大量局部层** 能恢复部分长程信息，却仍需要为“哪些层做全局”付出架构和系统复杂度。**FlashAttention** 则通过避免显式物化巨大 attention matrix、重排计算并减少显存搬运，能获得显著的常数项收益，却没有消除 `n²`。当目标走向数百万 tokens 时，课程因此继续追问：能否改变 attention 本身的计算形态？

> [!IMPORTANT]
> “支持更长 context window”与“渐近复杂度更优”是两件事。前者是产品或模型能力，后者是算法成本；必须结合实际 kernel、精度、通信与质量一起判断。

### 1.4 三种节省计算的基本策略

本讲出现的方案可以先按信息保存方式分类：

1. **压缩历史**：把过去累积进固定大小状态，如 linear attention、SSM、Mamba、GDN；
2. **稀疏读取历史**：仍保留显式 token memory，但先用轻量 indexer 选出少量位置，如 DSA；
3. **稀疏激活参数**：每个 token 只经过少数专家，如 MoE。

这三类方法都不是无条件胜出。固定状态可能丢失细节；稀疏检索仍需索引成本；MoE 又把计算问题转化成离散路由、通信和负载问题。

### 本章小结

- 标准 attention 的主要压力来自随序列长度增长的成对交互和 KV cache。
- FlashAttention 与局部/全局混合仍然很重要；渐近复杂度和常数项优化必须同时看。
- Attention alternatives 的目标不是单纯减少一个 Big-O，而是在质量、并行性、带宽、内存和延迟之间重新分配成本。
- 后续所有方法都可理解为“只让当前 token 读取必要的信息或激活必要的参数”。

## 2. Linear Attention：结合律与循环状态的双重视角

### 2.1 先移除无法交换的非线性

标准 attention 先形成 `QK^T`，再对权重施加 `ρ`（通常是 softmax），最后与 `V` 相乘。若 `ρ` 是恒等映射，矩阵乘法结合律允许先算 `K^T V`，避免显式构造 `n×n` 矩阵：

$$
\begin{aligned}
Q&\in\mathbb{R}^{n\times d_k},\quad K\in\mathbb{R}^{n\times d_k},\quad V\in\mathbb{R}^{n\times d_v},\\
\operatorname{Attn}(Q,K,V)&=\rho(QK^\top)V,\\
\rho=\operatorname{Id}\quad\Longrightarrow\quad (QK^\top)V&=Q(K^\top V).
\end{aligned}
$$

- `n`：序列长度。
- `d_k`：query/key 维度。
- `d_v`：value 维度。
- `Q`：所有位置的 query 矩阵。
- `K`：所有位置的 key 矩阵。
- `V`：所有位置的 value 矩阵。
- `ρ`：施加在 attention logits 或 weights 上的变换。
- `Id`：恒等映射。
- `T`：矩阵转置。
- `Attn`：attention 运算。

左结合大约需要 `n²d_k+n²d_v` 次乘加，右结合约为 `2nd_kd_v`。当 `d_k,d_v` 远小于 `n` 时，序列长度方向从二次变为线性。

![线性 attention 的结合律改写](assets/linear-attention.jpg)

*图：从 `(QK^T)V` 改写为 `Q(K^T V)` 及两种成本。官方课件第 4 页；视频对应讲解区间：`00:05:47--00:08:04`。*

> [!WARNING]
> 这里最关键的隐藏条件是 `ρ` 必须允许这种分解。softmax 不能直接“穿过”结合律，因此从 softmax attention 改成纯线性 attention 是有损的架构改变，不是等价加速。

### 2.2 同一运算可以写成 RNN

对因果序列逐位置展开，`K^T V` 就是历史 key-value 外积的累计状态：

$$
S_t=S_{t-1}+k_tv_t^\top,\qquad y_t=q_t^\top S_t.
$$

- `t`：当前位置。
- `k_t`：当前位置的 key 向量。
- `v_t`：当前位置的 value 向量。
- `q_t`：当前位置的 query 向量。
- `S_t`：截至位置 `t` 的累计 key-value 状态，形状约为 `d_k×d_v`。
- `S_{t-1}`：上一位置的累计状态。
- `y_t`：当前位置输出。
- `T`：向量转置。

![线性 attention 的循环形式](assets/recurrent-linear-attention.jpg)

*图：parallel form 与 recurrent form 的精确对应。官方课件第 5 页；视频对应讲解区间：`00:08:04--00:09:49`。*

这产生一个极有价值的 duality：训练时可以用并行矩阵形式吞吐整段序列，推理时用串行 recurrence，只维护固定大小状态。课堂问答专门澄清了两种“损失”不能混为一谈：

- 去掉 softmax 会改变模型，是潜在质量损失；
- 在已经线性化之后，把 parallel form 改写成 recurrent form 是代数等价，不再增加近似误差。

> [!IMPORTANT]
> 训练/推理双形态是本讲前半场的核心：同一线性运算可在训练时并行、在自回归推理时递推。

### 2.3 为什么纯线性状态还不够

简单加法会让所有历史以相同规则进入状态，表达力不足。现实模型因此加入输入相关的衰减、选择性写入、局部卷积或少量 full attention 层。问题从“能否压缩历史”转为“怎样有选择地保留、覆盖与遗忘”。

### 本章小结

- 线性 attention 的速度来自移除/分解 softmax 后改变乘法顺序。
- 其 parallel 与 recurrent 两种形式在代数上等价，分别适配训练和推理。
- 真正困难不在累计状态，而在让有限状态具备选择性记忆与遗忘能力。

## 3. 从 Mamba-2 到 GDN：给状态加门，并与 Attention 混合

### 3.1 Mamba-2 的教学性简化

在简单 recurrence 上加入输入相关衰减门和直接通路，可以得到课程用来解释 Mamba-2 的简化力学：

$$
S_t=\gamma_tS_{t-1}+k_tv_t^\top,\qquad
y_t=q_t^\top S_t+v_t^\top D,\qquad
\gamma_t=f(x_t).
$$

- `S_t`：当前位置更新后的状态。
- `S_{t-1}`：上一位置状态。
- `γ_t`：由当前输入决定的状态保留/衰减门。
- `k_t`：当前位置 key。
- `v_t`：当前位置 value。
- `q_t`：当前位置 query。
- `y_t`：当前位置输出。
- `D`：直接或 skip 路径参数。
- `x_t`：当前位置输入。
- `f`：把输入映射为门值的函数。
- `T`：向量转置。

![Mamba-2 的门控状态视角](assets/mamba2-gating.jpg)

*图：从 linear attention 到带衰减门与 skip path 的 Mamba-2 简化式。官方课件第 7 页；视频对应讲解区间：`00:11:11--00:13:25`。*

> [!NOTE]
> 这不是完整 Mamba-2 block。实际模型还包含 projection、卷积、规范化和更具体的 SSM 参数化；课程这里只保留与 linear attention 对照所需的状态更新骨架。

课堂提问还指出 `v_t^T D` 看似绕过状态。讲者的回答是：它给当前输入一条直接的局部通路，状态负责跨位置记忆，两者承担不同职责。

### 3.2 Gated Delta Net：写入前先沿 key 方向擦除

如果只衰减整个状态，模型不能精确改写某一类记忆。GDN 在当前 key 方向上先擦除旧内容，再写入新 value：

$$
\begin{aligned}
S_t&=\gamma_t\bigl(I-\beta_tk_tk_t^\top\bigr)S_{t-1}+\beta_tk_tv_t^\top,\\
y_t&=q_t^\top S_t,\qquad \gamma_t=f_\gamma(x_t),\quad \beta_t=f_\beta(x_t).
\end{aligned}
$$

- `S_t`：更新后的状态，`S_t∈R^{d_k×d_v}`。
- `S_{t-1}`：旧状态，形状同样是 `d_k×d_v`。
- `I`：`d_k×d_k` 单位矩阵。
- `β_t`：输入相关的写入/擦除门；为零时不写入。
- `k_tk_t^T`：当前 key 方向的外积。
- `γ_t`：旧状态整体保留门。
- `k_t`：当前 key，按 `d_k×1` 列向量理解。
- `v_t`：当前 value，按 `d_v×1` 列向量理解。
- `q_t`：当前 query，按 `d_k×1` 列向量理解。
- `y_t`：当前位置输出，形状为 `d_v×1`；公式中的 `q_t^T S_t` 横向写作 `1×d_v`。
- `x_t`：当前位置输入。
- `f_γ,f_β`：生成两个门值的输入相关函数。
- `T`：向量转置。

逐项做一次形状检查，就能看出“擦除旧状态”和“写入新状态”确实可以相加：

| 表达式 | 形状 | 检查结果 |
|---|---:|---|
| `S_{t-1}`、`S_t` | `d_k×d_v` | 状态以 key 维为行、value 维为列 |
| `k_t`、`q_t` | `d_k×1` | 两者都是列向量 |
| `v_t` | `d_v×1` | value 列向量 |
| `I`、`k_tk_t^T` | `d_k×d_k` | 二者可做减法 |
| `(I-β_tk_tk_t^T)S_{t-1}` | `(d_k×d_k)(d_k×d_v)=d_k×d_v` | 擦除项仍是状态形状 |
| `k_tv_t^T` | `(d_k×1)(1×d_v)=d_k×d_v` | 写入项也是状态形状 |
| `q_t^T S_t` | `(1×d_k)(d_k×d_v)=1×d_v` | 得到一个 value 维输出 |

所以第一项和第二项都落在 `R^{d_k×d_v}` 中，状态更新维度闭合；最后的 query 读取再把 key 维消掉，留下 value 维。

![Gated Delta Net 的擦除与写入](assets/gated-delta-net.jpg)

*图：GDN 通过 `I-β_tk_tk_t^T` 擦除当前 key 方向的旧内容。官方课件第 9 页；视频对应讲解区间：`00:15:22--00:18:21`。*

> [!WARNING]
> 只有在额外归一化条件下，`I-β_tk_tk_t^T` 才能严格解释成正交投影；更稳妥的说法是“沿 key 方向进行可学习擦除”。

### 3.3 混合架构为何成为主流折中

课程列举的 MiniMax M1、Nemotron 3、Qwen 3.5 / Qwen Next 都不是完全移除 attention，而是把线性/循环层与 full attention 层按比例混合。示例比例包括 7:1 或 3:1（若干线性层配一个 full attention 层）。这样做的直觉是：

- 大多数层用固定状态实现便宜的局部和中程传播；
- 少量 full attention 层提供高带宽的显式历史访问；
- 推理成本接近线性模型，同时降低纯固定状态的容量瓶颈。

讲者强调，现实证据常来自不同模型之间的横向比较，控制严格的同预算 ablation 仍稀缺。因此“混合架构效果不错”是强经验信号，却不能自动推出某个比例是普适最优。

### 本章小结

- Mamba-2 用输入相关衰减与直接通路增强有限状态。
- GDN 再加入定向擦除与选择性写入，使状态能被改写而非只累加。
- 现代模型常用“多数线性/SSM 层 + 少量 full attention 层”折中效率与长程读取能力。

## 4. DeepSeek Sparse Attention：保留显式历史，但只读 Top-k

### 4.1 与固定状态路线的根本区别

DSA 不把全部历史压缩进一个矩阵状态。它保留 token 级 memory，先用轻量 lightning indexer 为每个历史位置打分，再只对 top-k 候选执行较贵的标准 attention：

$$
\begin{aligned}
I_{t,s}&=\sum_{j=1}^{H^I}w^I_{t,j}\,\operatorname{ReLU}\!\left((q^I_{t,j})^\top k^I_s\right),\\
u_t&=\operatorname{Attn}\!\left(h_t,\{c_s\mid s\in\operatorname{TopK}(I_{t,:},k)\}\right).
\end{aligned}
$$

- `t`：当前 query 位置。
- `s`：候选历史位置。
- `I_{t,s}`：位置 `t` 对历史位置 `s` 的索引分数。
- `H^I`：indexer head 数量。
- `j`：indexer head 索引。
- `w^I_{t,j}`：第 `j` 个 indexer head 的权重。
- `q^I_{t,j}`：第 `j` 个 indexer query。
- `k^I_s`：历史位置 `s` 的 indexer key。
- `ReLU`：逐元素非线性。
- `h_t`：当前 token hidden state。
- `c_s`：最终 attention 使用的历史 key-value entry。
- `TopK`：选出最大分数集合的操作。
- `k`：保留的历史位置数。
- `u_t`：稀疏 attention 输出。
- `Attn`：对选中条目执行的精细 attention。
- `T`：向量转置。

![DeepSeek Sparse Attention 的 lightning indexer](assets/deepseek-sparse-attention.jpg)

*图：轻量 indexer 打分、top-k token selection 与精细 attention。官方课件第 12 页；视频对应讲解区间：`00:23:13--00:24:45`。*

### 4.2 为什么实际会快，但不应误称“严格线性”

课堂追问了一个关键问题：如果 indexer 仍要看全部历史，它不还是全局扫描吗？答案是肯定的。收益来自 indexer 的常数项极小：低维 query/key、少量 heads、ReLU、FP8 等；真正昂贵的 value 聚合只发生在 `k` 个选中位置上。

对长度 `n` 的整段训练，可用近似分解理解成本：indexer 仍可能达到 `O(n²d_I)`，随后精细 attention 约为 `O(nkd)`。若把选出的 `k` 个 token 当作独立 self-attention 子问题，则会看到 `O(k²d)`。因此不能简单写成“DSA 已变成线性时间”。

> [!IMPORTANT]
> DSA 的价值是把高精度、高维的读取集中到少量位置；它优化的是昂贵计算的覆盖范围，而不是让索引本身凭空消失。

### 4.3 Post-hoc 长上下文扩展与精度分工

DSA 的另一吸引力是可以在 dense 短上下文预训练之后，增加 indexer 并做长上下文适配。讲者把这看作架构设计与 context management / retrieval 的融合：模型内部开始显式学习“什么值得读”。

在低精度讨论中，indexer 可用 FP4/FP8 做候选选择，选中后的 attention 再用相对更高精度计算。这里的“高精度”是相对筛选阶段而言，不应自动等同于 FP32。

固定状态与 DSA 的取舍由信息容量决定：SSM 的状态大小固定，序列极长时可能形成容量瓶颈；DSA 保留显式历史，内存更高，但能回看细节。课程将这表述为工程 trade-off，而非一个已证明的容量定理。

### 本章小结

- DSA 用轻量 indexer 选出历史 top-k，再执行标准 attention。
- 它仍有全局索引成本，实际优势依赖低维、低精度和较小 `k`。
- 与固定状态模型相比，DSA 用更多显式 memory 换取更精确的长程回看能力。

## 5. MoE：让参数容量增长快于每 Token 计算量

### 5.1 Dense FFN 到 Sparse FFN

Transformer 中大量参数和 FLOPs 位于 FFN。MoE 把一个大 FFN 换成多个 expert FFN，并让 router 对每个 token 只选择少数 experts。于是模型的 **total parameters** 可以远大于 **active parameters**。

![Dense FFN 与 Sparse MoE FFN](assets/moe-definition.jpg)

*图：稠密模型与带 selector 的 sparse expert layer。官方课件第 15 页；视频对应讲解区间：`00:35:27--00:36:57`。*

例如课程引用 Qwen1.5-MoE：约 14.3B total parameters，但每个 token 只激活约 2.7B。比较模型时必须说明是哪种口径；只写一个“参数量”会产生严重误导。

### 5.2 为什么更多未激活参数仍然有用

在固定 active FLOPs 下，增加 experts 往往能改善训练 loss：不同 tokens 可调用不同子网络，参数容量增大而每个 token 的矩阵乘法量近似不变。

![固定计算下增加专家的经验收益](assets/moe-scaling.jpg)

*图：相近 active compute 下，更多 total parameters 带来更好训练曲线的经验结果。官方课件第 16 页；视频对应讲解区间：`00:36:57--00:38:18`。*

但这只是一种 **active-compute 视角的 free win**。总参数仍要存储；训练时要更新；token 要在设备间 all-to-all；路由不均还会造成 straggler。MoE 的真正成本从单卡 GEMM 转移到了分布式系统。

### 5.3 Expert parallelism 与部分反馈

把不同 experts 放在不同设备上，可获得额外的 expert parallelism。一次 MoE 层通常包括：router 打分 → token dispatch → 各设备 expert 计算 → token combine。网络拓扑和跨机带宽因此决定真实吞吐。

![MoE 的 expert parallelism](assets/expert-parallelism.jpg)

*图：不同 experts 分布到设备后形成的新并行维度。官方课件第 19 页；视频对应讲解区间：`00:39:50--00:40:42`。*

路由又带来部分反馈：一个 token 只经过被选 expert，模型无法直接观察“如果送到其他 expert 会怎样”。这很像 contextual bandit，但主流系统通常不用完整 RL，而采用可微 router、top-k 选中路径、噪声和辅助损失等启发式方案。

### 本章小结

- MoE 用稀疏激活把 total capacity 与 per-token active compute 解耦。
- total parameters、active parameters 与 active FLOPs 是三个不同口径。
- MoE 的收益伴随新的通信、路由、存储和负载均衡成本。

## 6. 路由设计：谁选择谁，以及如何组合专家输出

### 6.1 三类 assignment

课程区分三类路由：

- **token-choice**：每个 token 选自己的 top-k experts，最常用；
- **expert-choice**：每个 expert 选一批 tokens，负载更好控制，但因果与在线推理更麻烦；
- **global assignment**：在 batch 级做匹配，目标清楚却计算和实现复杂。

![Token-choice、expert-choice 与全局匹配](assets/routing-types.jpg)

*图：三种 token-expert assignment 的对照。官方课件第 27 页；视频对应讲解区间：`00:48:15--00:49:00`。*

经验上，简单 token-choice top-k 仍是主流。RL 路由和线性匹配在理论上更直接，却因高方差、复杂度与在线约束没有成为默认。

### 6.2 标准 token-choice top-k

router 先算 token 与每个 expert 的 affinity，再保留 top-k，最后加权专家输出并走 residual：

$$
\begin{aligned}
s_{i,t}&=\operatorname{Softmax}_i\!\left((u_t^l)^\top e_i^l\right),\\
g_{i,t}^l&=\begin{cases}
s_{i,t},&s_{i,t}\in\operatorname{TopK}(\{s_{j,t}\}_{j=1}^{N},K),\\
0,&\text{otherwise},
\end{cases}\\
h_t^l&=u_t^l+\sum_{i=1}^{N}g_{i,t}^l\operatorname{FFN}_i^l(u_t^l).
\end{aligned}
$$

- `l`：层索引。
- `t`：token 位置。
- `i,j`：expert 索引。
- `u_t^l`：进入第 `l` 个 MoE 层的 hidden state。
- `e_i^l`：expert `i` 的 router embedding 或分类权重。
- `s_{i,t}`：router 对 expert `i` 的 soft probability。
- `TopK`：选出最大 `K` 个分数的操作。
- `K`：每 token 选择的 routed experts 数。
- `N`：routed experts 总数。
- `g_{i,t}^l`：top-k mask 后的 gate。
- `FFN_i^l`：第 `i` 个 expert FFN。
- `h_t^l`：MoE residual 输出。
- `Softmax_i`：沿 expert 维度的 softmax。
- `⊤`：向量转置。

#### 一个 `T=3, N=4, K=2` 的手算例子

为了把“affinity → top-k → gate”具体化，令每个 token 的 hidden dimension 为 `d=2`。三行 token states 与四列 expert router vectors 相乘，得到每个 token 对每个 expert 的 logits；softmax 必须逐行、沿 expert 维归一化：

$$
\begin{aligned}
U&=\begin{bmatrix}1&0\\0&1\\1&2\end{bmatrix}\in\mathbb R^{3\times2},\qquad
E=\begin{bmatrix}2&1&0&-1\\0&1&3&-1\end{bmatrix}\in\mathbb R^{2\times4},\\
A&=UE=\begin{bmatrix}2&1&0&-1\\0&1&3&-1\\2&3&6&-3\end{bmatrix}\in\mathbb R^{3\times4},\\
S&=\operatorname{Softmax}_{\rm row}(A)\approx
\begin{bmatrix}
0.644&0.237&0.087&0.032\\
0.041&0.112&0.831&0.015\\
0.017&0.047&0.936&0.000
\end{bmatrix},\\
G&=\operatorname{KeepTop2}(S)=
\begin{bmatrix}
0.644&0.237&0&0\\
0&0.112&0.831&0\\
0&0.047&0.936&0
\end{bmatrix}\in\mathbb R^{3\times4}.
\end{aligned}
$$

- `T=3`：batch/sequence 中参与路由的 token 数量，即 `U` 的行数。
- `N=4`：routed experts 数量，即 `E`、`A`、`S`、`G` 的列数。
- `K=2`：每个 token 保留的 expert 数量。
- `d=2`：token hidden state 与 router vector 的维度。
- `U`：token state 矩阵，形状 `[T,d]=[3,2]`。
- `E`：由四个 expert router vectors 按列组成的矩阵，形状 `[d,N]=[2,4]`。
- `A`：affinity/logit 矩阵 `UE`，形状 `[T,N]=[3,4]`。
- `S`：对 `A` 每一行做 softmax 后的稠密 router probability，形状仍为 `[3,4]`。
- `G`：每行只保留两个最大 probability 后的 sparse gate，形状仍为 `[3,4]`。
- `Softmax_row`：固定一个 token，沿四个 experts 归一化。
- `KeepTop2`：每行保留最大两项，其余置零。
- `R`：实数域。
- `≈`：三位小数近似。

三行的选择集合分别是 `{expert 1, expert 2}`、`{expert 3, expert 2}`、`{expert 3, expert 2}`。按照课件这版“softmax 后 top-k”的定义，未再次归一化，所以每行 gate 之和可以小于 1。现在给六个实际被选中的 expert outputs 指定简单二维值，就能手算 residual combine：

$$
\begin{aligned}
&F_{1,1}=[1,0],\quad F_{1,2}=[0,2],\qquad
F_{2,2}=[1,1],\quad F_{2,3}=[-1,2],\\
&F_{3,2}=[2,0],\quad F_{3,3}=[0,1],\\
H&=U+\begin{bmatrix}
0.644F_{1,1}+0.237F_{1,2}\\
0.112F_{2,2}+0.831F_{2,3}\\
0.047F_{3,2}+0.936F_{3,3}
\end{bmatrix}
\approx
\begin{bmatrix}
1.644&0.474\\
-0.719&2.774\\
1.094&2.936
\end{bmatrix}\in\mathbb R^{3\times2}.
\end{aligned}
$$

- `F_{t,i}`：expert `i` 对 token `t` 的二维 FFN 输出。
- `H`：加上稀疏 expert 加权和后的 residual 输出，形状 `[T,d]=[3,2]`。
- `U`：原 token states，作为 residual 分支逐行加回。
- `t`：token 行索引。
- `i`：expert 列索引。
- 数字系数：来自 sparse gate `G` 中对应的非零项。
- `R`：实数域。
- `≈`：使用前三位 gate 近似计算。

完整的 tensor shape 流如下：

| 步骤 | 运算 | 输入形状 | 输出形状 |
|---:|---|---|---|
| 0 | 准备 token states / router vectors | `U:[3,2]`，`E:[2,4]` | — |
| 1 | affinity `A=UE` | `[3,2]×[2,4]` | `A:[3,4]` |
| 2 | row-wise softmax | `A:[3,4]` | `S:[3,4]` |
| 3 | 每行 top-2 与置零 | `S:[3,4]` | indices `[3,2]`，`G:[3,4]` |
| 4 | 执行被选 experts | 概念张量 `F:[3,4,2]` | 实际只计算 `T×K=6` 个二维输出 |
| 5 | gate 加权并沿 expert 维求和 | `G:[3,4]`，`F:[3,4,2]` | expert sum `[3,2]` |
| 6 | residual combine | `U:[3,2]` + expert sum `[3,2]` | `H:[3,2]` |

> [!NOTE]
> 这里为了匹配课件第 31 页，保留的是 softmax-before-top-k gate。若实现采用 top-k 后对 selected scores 重新归一化，选择集合不变，但上面六个非零权重和最终 `H` 会改变。

![标准 top-k router](assets/topk-routing.jpg)

*图：router score、top-k mask 与加权 expert 输出。官方课件第 31 页；视频对应讲解区间：`00:52:57--00:54:20`。*

不同模型的归一化顺序并不完全相同。DeepSeek V1–V2、Grok、Qwen 使用接近图中“softmax 后 top-k”的形式；Mixtral、DBRX、DeepSeek V3 常在 top-k 后对 selected scores 再归一化。不能把两种实现混成同一公式。

### 6.3 Fine-grained 与 shared experts

把一个大 expert 切成更多小 expert，可在相近 active parameter budget 下增加组合数；shared experts 则始终激活，用于承接跨 token 的共通知识。

![细粒度 routed experts 与 shared experts](assets/fine-grained-shared-experts.jpg)

*图：常规 top-2、细粒度分割与 shared-expert isolation。官方课件第 32 页；视频对应讲解区间：`00:54:20--00:55:43`。*

DeepSeek 的 ablation 支持细粒度与 shared expert，但 OLMoE 的受控实验没有复现 shared expert 的稳定收益。这种冲突很重要：shared expert 是合理设计假设，不是已被普遍证明的定律。

课堂还澄清，shared expert 可复制到多设备以减少通信，但复制会用更多显存换带宽；是否值得取决于拓扑与参数规模。

### 6.4 专家并不等于人类语义领域

可视化常显示某些 experts 偏好标点、特殊符号、非英语字符或局部 token pattern，但课程没有证据支持“一个医学专家、一个法律专家”这样的整洁分工。expert specialization 更可能是高维、分布式和上下文相关的。

### 本章小结

- Token-choice top-k 以简单和在线友好成为主流。
- 路由包含打分、离散选择与加权聚合；softmax 的位置因模型而异。
- Fine-grained/shared experts 是容量设计手段，但经验效果必须以具体 ablation 判断。

## 7. 训练 MoE：探索、均衡与不稳定性

### 7.1 离散 top-k 怎样得到学习信号

hard top-k 对“未入选的集合变化”不可微，但被选 expert 的 gate 和输出仍可反向传播；soft router probability、噪声与辅助损失又为边界附近提供信号。课程讨论了三类探索手段：

- noisy top-k：给 logits 加可学习尺度的高斯噪声；
- multiplicative jitter：训练时对 logits 乘小幅均匀噪声；
- REINFORCE：理论上直接优化离散策略，但方差和复杂度通常不划算。

![Noisy top-k gating](assets/noisy-topk.jpg)

*图：确定性 router logits、噪声尺度与 KeepTopK。官方课件第 38 页；视频对应讲解区间：`01:00:33--01:02:25`。*

### 7.2 Switch balancing loss 的梯度直觉

如果许多 tokens 都选同一 expert，其他设备空闲而热门设备成为瓶颈。Switch loss 把实际 hard load 与 soft probability mass 相乘：

$$
\begin{aligned}
\mathcal L_{\mathrm{bal}}&=\alpha N\sum_{i=1}^{N}f_iP_i,\\
f_i&=\frac1T\sum_{x\in\mathcal B}\mathbf1\{\arg\max_j p_j(x)=i\},\\
P_i&=\frac1T\sum_{x\in\mathcal B}p_i(x).
\end{aligned}
$$

- `L_bal`：负载均衡辅助损失。
- `α`：辅助损失权重。
- `N`：experts 数量。
- `i,j`：expert 索引。
- `T`：batch 中 token 数。
- `𝔅`：token batch。
- `x`：一个 token 表示。
- `p_i(x)`：token `x` 分给 expert `i` 的 router probability。
- `argmax`：选择最大概率 expert。
- `1{·}`：条件成立时为 1 的指示函数。
- `f_i`：实际发送给 expert `i` 的 token 比例。
- `P_i`：expert `i` 得到的平均 probability mass。

![Switch Transformer 的均衡损失](assets/switch-balancing-loss.jpg)

*图：hard load `f_i` 与 soft mass `P_i` 的点积。官方课件第 40 页；视频对应讲解区间：`01:03:13--01:06:00`。*

看公式本身不容易直觉化，观察梯度更清楚：某 expert 的 `f_i` 越大，损失对其 probability mass 的正梯度越强；梯度下降会压低热门 expert 的后续分配概率。

### 7.3 Expert balance、device balance 与“aux-loss-free”

完美均衡每个 expert 可能过度限制模型，所以 DeepSeek V1–V2 还在设备层聚合负载；只要同一设备上的总流量平衡，设备内 experts 可以保留更自由的分工。DeepSeek V3 更进一步，用在线更新的 expert bias 影响 top-k 集合选择：

$$
g'_{i,t}=\begin{cases}
s_{i,t},&s_{i,t}+b_i\in\operatorname{TopK}(\{s_{j,t}+b_j\}_{j=1}^{N_r},K_r),\\
0,&\text{otherwise}.
\end{cases}
$$

- `g'_{i,t}`：token `t` 对 expert `i` 的稀疏 gate。
- `s_{i,t}`：不含 bias 的 router score。
- `b_i`：根据负载在线更新的 expert-selection bias。
- `TopK`：选出最大分数集合的操作。
- `N_r`：routed experts 数。
- `K_r`：每 token 激活的 routed experts 数。
- `i,j`：expert 索引。
- `t`：token 位置。

![DeepSeek V3 的 per-expert bias](assets/deepseek-v3-expert-bias.jpg)

*图：bias 参与集合排名，但 selected gate 仍使用原始 score。官方课件第 42 页；视频对应讲解区间：`01:07:12--01:07:49`。*

因此“aux-loss-free”应谨慎理解：它主要指减少传统全局辅助均衡损失的依赖；同页仍保留 complementary sequence-wise auxiliary protection。

OLMoE 的实验则给出反例：移除 balancing loss 后训练没有立即崩溃，但 expert 使用会明显失衡。这说明均衡是否必要与 router、初始化、规模和训练配方有关。

![移除负载均衡损失的 OLMoE 实验](assets/no-balancing-ablation.jpg)

*图：有/无 load balancing 时的 loss 与 expert load。官方课件第 43 页；视频对应讲解区间：`01:07:53--01:09:08`。*

### 本章小结

- top-k 的离散性通过选中路径梯度、soft probability、噪声和辅助目标共同处理。
- Switch loss 的本质是按拥挤程度压低热门 expert 的分配概率。
- “均衡”可以发生在 expert、device、sequence 多个尺度；V3 的 aux-loss-free 也不是完全没有保护项。

## 8. 系统与工程：通信、Sparse Kernels、稳定性、微调和 Upcycling

### 8.1 All-to-all 才是 MoE 的隐藏主循环

MoE 层要把 tokens dispatch 到 expert 所在设备，再将输出 combine 回原位置。它与 data、tensor、pipeline、sequence parallelism 叠加后，形成复杂的通信图。

![MoE 的多维并行与通信](assets/moe-system-parallelism.jpg)

*图：expert parallelism 与其他并行方式的组合。官方课件第 44 页；视频对应讲解区间：`01:11:39--01:12:47`。*

token 按 expert 重排后，矩阵形状不再规则。MegaBlocks 一类实现把小矩阵乘合并成 block-sparse GEMM，减少 padding 与 kernel launch；更现代的 dropless 实现也避免因固定 capacity 直接丢 token。

![Block-sparse expert matrix multiplication](assets/block-sparse-matmul.jpg)

*图：逐 expert 小 GEMM 与 block-sparse 合并。官方课件第 45 页；视频对应讲解区间：`01:12:47--01:13:55`。*

LatentMoE / Nemotron-3 进一步在通信前下投影 activation、到 expert 侧再上投影，以额外计算换更少传输字节。这与后面的 DeepSeek MLA 都使用“latent compression”思想，但一个压缩 expert communication，另一个压缩 KV cache，不应混为同一模块。

### 8.2 Token dropping 为什么增加随机性

早期 capacity-limited MoE 为每个 expert 设定 batch capacity；溢出的 token 会被丢弃。于是一个 token 是否被执行不仅取决于自身 router，还取决于同 batch 其他 tokens 的选择，带来额外随机性。

![Capacity-limited MoE 的 token dropping](assets/token-dropping.jpg)

*图：batch composition 可改变某个 token 是否被 expert 接收。官方课件第 47 页；视频对应讲解区间：`01:15:08--01:16:36`。*

> [!WARNING]
> 这种随机性针对带 capacity/drop 的实现；不能推广到所有 MoE。Dropless routing 和更好的 sparse kernels 正是为避免这一问题而发展。

### 8.3 Router 用更高精度，z-loss 控制 logit 尺度

router softmax 对舍入很敏感，小数值误差可能改变 top-k 集合。常见做法是让 expert 计算保持低精度，但 router logits/softmax 使用 FP32，并加入 z-loss 抑制 log-normalizer 过大：

$$
L_z(x)=\frac1B\sum_{i=1}^{B}\left(\log\sum_{j=1}^{N}e^{x_j^{(i)}}\right)^2.
$$

- `L_z`：router z-loss。
- `B`：batch 中 router 样本或 tokens 数。
- `i`：token 索引。
- `N`：experts 数量。
- `j`：expert 索引。
- `x_j^{(i)}`：第 `i` 个 token 对 expert `j` 的 router logit。
- `log Σ exp`：softmax 的 log-normalizer。

![Router FP32 与 z-loss](assets/router-zloss.jpg)

*图：低精度 softmax 敏感性、router FP32 与 z-loss。官方课件第 48 页；视频对应讲解区间：`01:16:36--01:17:58`。*

课程展示的 ablation 表明 z-loss 在部分设置改善稳定性，但它不是所有模型都必需的万能修复；精度、初始化、clipping 与 router 设计要一起看。

### 8.4 微调与 Upcycling

MoE 在小规模 fine-tuning 数据上可能收敛，因为预训练已学到 router 与 expert；实践中既可只调非-MoE MLP/共享部分，也可使用全量或大规模 SFT。结论不是“MoE 天生难微调”，而是更容易受数据规模和路由漂移影响。

Upcycling 则把已有 dense checkpoint 的 FFN 复制或拆分成 experts，再继续训练 sparse MoE。它能复用已有表示，但通常仍需数百亿至数千亿 tokens 让 experts 分化，不能理解为零成本扩容。

![Dense checkpoint 到 MoE 的 upcycling](assets/upcycling.jpg)

*图：从 dense FFN 初始化多个 experts，以及额外预训练成本。官方课件第 51 页；视频对应讲解区间：`01:19:41--01:20:59`。*

MiniCPM 与 Qwen 的案例说明 upcycling 可以成功；但如果目标从一开始就是超大 MoE，直接稀疏预训练往往更经济，因为无需先完整支付 dense pretraining。

### 本章小结

- MoE 的端到端效率取决于 dispatch/combine、拓扑、稀疏 kernel 和负载，而不只是 expert FLOPs。
- Token dropping 与 router 数值问题都是具体实现造成的风险，可由 dropless kernels、FP32 router 和稳定化损失缓解。
- 微调和 upcycling 可行，但都需要额外训练与路由适配，不能当作免费转换。

## 9. DeepSeek V1→V3：模型结构与系统约束共同演化

### 9.1 V1：shared + fine-grained 的现代原型

V1 的课程配置为 16B total / 2.8B active，2 个 shared experts，并使用细粒度 routed experts、标准 top-k、expert/device auxiliary balancing。输出可抽象为：

$$
h'_t=u_t+\sum_{i=1}^{N_s}\operatorname{FFN}^{(s)}_i(u_t)+\sum_{i=1}^{N_r}g_{i,t}\operatorname{FFN}^{(r)}_i(u_t).
$$

- `t`：token 位置。
- `u_t`：进入 MoE FFN 的 hidden state。
- `h'_t`：加入 shared/routed expert 输出后的 hidden state。
- `N_s`：shared experts 数量。
- `N_r`：routed experts 数量。
- `i`：expert 索引。
- `FFN_i^(s)`：始终启用的 shared expert。
- `FFN_i^(r)`：由 router 选择的 routed expert。
- `g_{i,t}`：top-k 后的 expert gate。

![DeepSeek MoE V1](assets/deepseek-moe-v1.jpg)

*图：V1 的 shared/routed experts、标准 top-k 与均衡目标。官方课件第 54 页；视频对应讲解区间：`01:22:33--01:22:55`。*

### 9.2 V2：把通信写进路由目标

V2 的课程配置为 236B total / 21B active。它引入 top-M device routing：先限制 token 访问的设备集合，再从这些设备上的 experts 中选择；communication balancing 同时约束通信流入和流出。

![DeepSeek MoE V2](assets/deepseek-moe-v2.jpg)

*图：V2 的 top-M device routing 与 communication balancing。官方课件第 55 页；视频对应讲解区间：`01:22:55--01:23:25`。*

这里的关键原则是“respect your systems”：架构目标函数不仅优化语言建模 loss，也要反映设备拓扑和通信代价。

### 9.3 V3：解耦 affinity、集合选择和权重归一化

V3 的课程配置为 671B total / 37B active、1 个 shared expert、每 token 激活 8 个 routed experts。router 不是对所有 experts 做一次 softmax，而是 sigmoid 独立打分、top-k 选集合，再只在 selected scores 间归一化：

$$
\begin{aligned}
s_{i,t}&=\operatorname{Sigmoid}(u_t^\top e_i),\\
g'_{i,t}&=\begin{cases}s_{i,t},&i\in\operatorname{TopK}(\{s_{j,t}\}_{j=1}^{N_r},K_r),\\0,&\text{otherwise},\end{cases}\\
g_{i,t}&=\frac{g'_{i,t}}{\sum_{j=1}^{N_r}g'_{j,t}}.
\end{aligned}
$$

- `t`：token 位置。
- `i,j`：expert 索引。
- `u_t`：token hidden state。
- `e_i`：expert `i` 的 router vector。
- `s_{i,t}`：sigmoid affinity，尚未跨 experts 归一化。
- `Sigmoid`：独立压缩每个 affinity 的函数。
- `TopK`：选择最大 `K_r` 个分数的操作。
- `N_r`：routed experts 总数。
- `K_r`：激活 routed experts 数量。
- `g'_{i,t}`：top-k mask 后的 score。
- `g_{i,t}`：只在 selected experts 间归一化的权重。

![DeepSeek MoE V3](assets/deepseek-moe-v3.jpg)

*图：V3 的 sigmoid score、top-k、selected normalization 与 seq-wise protection。官方课件第 56 页；视频对应讲解区间：`01:23:25--01:23:50`。*

> [!WARNING]
> 官方课件第 56 页标题为 V3，但参数行误写 V2；expert 总数又与课程前面的配置表不一致。这里保留讲者明确给出的 total/active/active-expert 口径，不静默补写有冲突的 expert 总数。

### 9.4 三代变化的主线

| 版本 | 路由/结构重点 | 系统意识 |
|---|---|---|
| V1 | shared + fine-grained experts；标准 top-k | expert/device auxiliary balance |
| V2 | top-M device routing | communication in/out balance |
| V3 | sigmoid affinity、selected normalization、online expert bias | 减少全局 aux 干扰，同时保留 sequence-wise protection |

三代不是反复推倒重来，而是在相似 sparse FFN 骨架上逐步重新安排 routing、balancing 与 communication 的耦合关系。

### 本章小结

- V1 建立 shared + fine-grained + top-k 的现代 MoE 原型。
- V2 把设备选择和通信平衡纳入模型设计。
- V3 将 affinity、top-k 集合与 selected weights 拆开，并用在线 bias 辅助均衡。

## 10. DeepSeek V3 的两个补充组件：MLA 与 MTP

### 10.1 MLA：缓存低维 latent，而不是完整 KV

Multi-head Latent Attention（MLA）先把 hidden state 压到低维 latent，再从 latent 恢复 content Q/K/V。推理时主要缓存 `c_t^{KV}`，因此 KV cache 显著缩小：

$$
\begin{aligned}
c_t^{KV}&=W^{DKV}h_t,&k_t^C&=W^{UK}c_t^{KV},&v_t^C&=W^{UV}c_t^{KV},\\
c_t^Q&=W^{DQ}h_t,&q_t^C&=W^{UQ}c_t^Q.
\end{aligned}
$$

- `t`：token 位置。
- `h_t`：输入 hidden state。
- `c_t^(KV)`：低维 KV latent，推理时缓存。
- `c_t^Q`：低维 query latent，主要降低训练内存。
- `W^(DKV)`：KV down-projection。
- `W^(DQ)`：query down-projection。
- `W^(UK)`：key up-projection。
- `W^(UV)`：value up-projection。
- `W^(UQ)`：query up-projection。
- `k_t^C`：恢复出的 content key。
- `v_t^C`：恢复出的 content value。
- `q_t^C`：恢复出的 content query。

![MLA 的 latent KV cache 架构](assets/mla-architecture.jpg)

*图：MLA 从低维 latent 生成 Q/K/V，并标出推理时缓存位置。官方课件第 57 页；视频对应讲解区间：`01:23:50--01:24:42`。*

不带位置旋转时，key up-projection 可以吸收到 query 侧；带 RoPE 后，位置相关旋转夹在矩阵之间，无法预先合并：

$$
\begin{aligned}
\langle Q,K\rangle&=\langle hW^Q,W^{UK}c_t^{KV}\rangle=\langle hW^QW^{UK},c_t^{KV}\rangle,\\
\langle QR_q,R_kK\rangle&=\langle hW^QR_q,R_kW^{UK}c_t^{KV}\rangle=\langle hW^QR_qR_kW^{UK},c_t^{KV}\rangle.
\end{aligned}
$$

- `Q,K`：未压缩表述下的 query 与 key。
- `h`：query 侧 hidden state。
- `W^Q`：query projection。
- `W^(UK)`：latent key up-projection。
- `c_t^(KV)`：位置 `t` 的 KV latent。
- `R_q`：query 位置对应的 RoPE 旋转矩阵。
- `R_k`：key 位置对应的 RoPE 旋转矩阵。
- `⟨·,·⟩`：attention 内积。

![MLA 的矩阵吸收与 RoPE 冲突](assets/mla-derivation.jpg)

*图：无 RoPE 时可合并 projection，加入位置旋转后不再成立。官方课件第 58 页；视频对应讲解区间：`01:24:42--01:25:05`。*

课件给出的直觉性解法是保留少量 non-latent key dimensions 专门承载 RoPE，content 部分继续使用 latent cache。讲者没有展开完整多头实现，因此这里也不超出课程范围补全论文细节。

### 10.2 MTP：让模型同时学习更远的 token

Multi-Token Prediction（MTP）用轻量模块把上一预测深度的 hidden representation 与未来偏移 token embedding 合并，再预测更远 token：

$$
\begin{aligned}
h_i^k&=M_k\!\left[\operatorname{RMSNorm}(h_i^{k-1});\operatorname{RMSNorm}(\operatorname{Emb}(t_{i+k}))\right],\\
h_{1:T-k}^{\prime k}&=\operatorname{TRM}_k(h_{1:T-k}^k),\qquad
p_{i+k+1}^k=\operatorname{OutHead}(h_i^{\prime k}).
\end{aligned}
$$

- `i`：序列位置。
- `k`：MTP module 或预测深度索引。
- `t_{i+k}`：偏移到未来位置的 token。
- `Emb`：token embedding 函数。
- `h_i^(k-1)`：上一预测深度的 hidden representation。
- `M_k`：拼接两个 RMSNorm 表示后的 projection module。
- `RMSNorm`：均方根归一化。
- `T`：序列长度。
- `1:T-k`：从位置 1 到 `T-k` 的序列切片。
- `TRM_k`：第 `k` 个轻量 Transformer block。
- `h'`：经轻量 Transformer 更新后的表示。
- `OutHead`：输出 token distribution 的 head。
- `p_(i+k+1)^k`：第 `k` 个模块对更远 token 的预测分布。
- 分号 `;`：向量拼接。

![DeepSeek 的 Multi-Token Prediction](assets/multi-token-prediction.jpg)

*图：DeepSeek MTP、EAGLE 对照与多步预测模块。官方课件第 59 页；视频对应讲解区间：`01:25:05--01:25:41`。*

MTP 有两种动机：额外未来目标迫使共享表示捕捉更长程的可预测结构；推理时多个候选又可作为内置 speculative decoding 草稿。课件的“only do MTP with one token ahead”更可能指只使用一个额外预测深度，不能据此断言它退化成普通 next-token prediction。

### 本章小结

- MLA 用低维 KV latent 降低 cache，但必须专门处理 RoPE 破坏 projection 合并的问题。
- MTP 既提供更远期训练监督，也可能为 speculative decoding 提供候选。
- 两者说明架构优化往往同时面向统计效率与推理系统，而不是只改一个 layer 名称。

## 总结与延伸

### 讲者的结尾结论

在 `01:25:42--01:26:14`，讲者用三点收束 MoE：

1. 稀疏性允许模型拥有远多于每个 token 实际计算量所对应的参数；
2. 离散 top-k routing 很难优化，但简单启发式方法已经足够有效；
3. 现有大量经验结果表明 MoE 具有成本效益，短期内不会退出主流架构。

![课程的 MoE 总结页](assets/moe-summary.jpg)

*图：稀疏激活、top-k 路由与经验有效性三条结论。官方课件第 60 页；视频对应讲解区间：`01:25:43--01:26:14`。*

### 统一视角：对历史与参数做条件访问

把整节课压缩成一张概念地图，可以看到两个对偶问题：

| 设计对象 | 稠密基线 | 条件计算方案 | 新增代价 |
|---|---|---|---|
| 历史信息 | 每个 query 读取全部 tokens | 线性状态、SSM/GDN、混合层、DSA top-k | 状态容量、索引、质量损失 |
| 参数容量 | 每个 token 激活完整 FFN | MoE top-k experts | 路由、均衡、all-to-all、稀疏 kernel |

这两条线共享同一工程规律：**删掉的计算并不会免费消失，而会以选择器、状态、通信或优化难度的形式回来。** 因此评估新架构至少要同时报告：模型质量、active compute、total parameters、内存、通信、吞吐、延迟和训练稳定性。

### 可迁移的设计原则

- **先定位真正昂贵的轴**：长序列的瓶颈可能是 attention matrix，也可能是 KV cache 或带宽；MoE 的瓶颈可能是通信而非 FLOPs。
- **让便宜的选择器负责缩小昂贵操作的作用域**：DSA 的 indexer 与 MoE router 是同一类模式。
- **训练与推理可以使用等价但不同形态**：linear attention 的 parallel/recurrent duality 是最清楚的例子。
- **统计模型必须尊重硬件拓扑**：V2 的 device routing、communication balance 和 block-sparse kernels 都把系统约束写进架构。
- **不把单篇 ablation 当普适定律**：shared experts、balancing loss、z-loss 和混合比例都存在模型依赖的反例。

### 开放问题与下一步

1. 能否把 DSA 的检索、外部 memory 和 context management 统一成端到端可训练的读取策略？
2. 固定状态模型怎样扩大可寻址容量，同时保留 linear-time recurrent inference？
3. MoE router 能否在不依赖强辅助损失的情况下兼顾 specialization、稳定性和拓扑感知？
4. 在统一硬件和数据预算下，hybrid attention、DSA 与纯 SSM 的高质量受控比较仍然不足。
5. MLA、MTP 与 speculative decoding 的组合，是否能让大 MoE 的 memory、compute 和 latency 同时受益？

### 本章小结

- Attention alternatives 稀疏化“读哪些历史”，MoE 稀疏化“用哪些参数”。
- 真正成功的架构需要算法、优化、精度、kernel 与分布式系统协同设计。
- 本讲最值得带走的不是某个固定配方，而是 capacity、active compute 与 information access 可以被分别设计。
