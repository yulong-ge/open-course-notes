# Stanford CS336 2026 Lecture 3：Architectures

![课程封面](assets/cover.jpg)

> [!NOTE]
> **课程**：Stanford CS336 — Language Modeling from Scratch（Spring 2026）  
> **讲次**：Lecture 3: Architectures  
> **讲者**：Tatsunori Hashimoto  
> **视频**：[YouTube 原视频](https://www.youtube.com/watch?v=lVynu4bo1rY)  
> **时长**：约 1 小时 29 分钟  
> **材料说明**：讲义基于公开视频、人工英文字幕与官方课件交叉整理；公开仓库不再分发这些原始文件。

这节课不是要寻找一个永远正确的“最佳 Transformer”，而是训练一种更有用的判断能力：看到一项架构改动时，先问它在解决什么约束——优化稳定性、泛化、训练吞吐，还是推理解码成本；再看证据是否真的支持它成为默认选择。整堂课可以压缩成一条主线：**现代语言模型的架构，是模型质量、优化稳定性和硬件效率共同塑造的工程折中。**

> [!IMPORTANT]
> 阅读本讲需要知道 token、向量、矩阵乘法、softmax、残差连接和自注意力的基本概念。若暂时不熟悉 Jacobian、arithmetic intensity 或 KV cache，不影响开始学习；对应概念会在第一次出现时补足。

## 0. 阅读前的五分钟桥接：一个 token 怎样穿过注意力层

### 0.1 从 token 到 embedding

语言模型先用 tokenizer 把文本切成 token，再把每个 token 映射成整数 ID。例如，“猫爱鱼”可能被切成 `['猫', '爱', '鱼']`，对应 ID `[17, 42, 9]`。ID 本身没有大小或距离意义；模型只把它当作 embedding 表中的行号。

若词表有 $V$ 行、每个 token 用 $d_{\text{model}}$ 个数表示，那么查表过程可写成：

$$
X=E[\text{ids}],
\qquad
E\in\mathbb{R}^{V\times d_{\text{model}}},
\qquad
X\in\mathbb{R}^{n\times d_{\text{model}}}.
$$

- $E$：token-embedding 查找表；
- $V$：词表大小；
- $d_{\text{model}}$：每个 token 的隐藏向量维度；
- $\text{ids}$：长度为 $n$ 的 token-ID 序列；
- $n$：当前序列的 token 数；
- $X$：查表后按序排列的 token 表示矩阵。

一个便于观察 shape 的玩具例子是 $n=3,d_{\text{model}}=4$：三个 token 查表后得到 `X.shape = [3, 4]`，也就是三行 token、每行四个特征。真实模型只是把 4 换成数千，并增加 batch 维。

### 0.2 Q、K、V 各自负责什么

自注意力把同一份输入 $X$ 线性投影成三种角色：**Q（query）表示当前 token 想找什么，K（key）表示每个 token 可以用什么特征被匹配，V（value）表示匹配成功后真正取回什么内容。**

为了先看清数据流，暂时忽略多头拆分：

$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
$$

- $X\in\mathbb{R}^{n\times d_{\text{model}}}$：输入 token 表示；
- $W_Q,W_K,W_V$：可学习的 query、key、value 投影矩阵；
- $Q$：每个位置的查询向量；
- $K$：每个位置的匹配键向量；
- $V$：每个位置可被聚合的内容向量。

在上面的 `[3, 4]` 玩具输入中，若有两个 head、每个 head 两维，那么投影结果可从 `[n, d_model] = [3, 4]` 重排为 `[heads, seq, head_dim] = [2, 3, 2]`。每个 head 都会独立做一次“Q 找 K，再按权重读取 V”。

### 0.3 causal mask 与 softmax

decoder-only 语言模型只能使用当前位置及其左侧 token，不能偷看未来答案。它先算 Q/K 相似度，再把未来位置的 score 加上 $-\infty$，最后沿每一行做 softmax：

$$
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),
\qquad
O=AV.
$$

- $Q,K,V$：query、key、value；
- $K^\top$：把 key 的 token 维转置以计算两两内积；
- $d_h$：一个 attention head 的维度；
- $M$：causal mask，对允许位置填 0，对未来位置填 $-\infty$；
- $A$：每个 query 对可见 token 的注意力概率；
- $O$：对 value 加权求和后的输出；
- $\operatorname{softmax}$：把一行有限 scores 转成总和为 1 的非负权重。

三个 token 的 mask 是一个下三角结构：第一行只能看第一个 token，第二行能看前两个，第三行能看全部三个。

| query 位置 | K₁ | K₂ | K₃ |
|---|---:|---:|---:|
| 1 | 0 | $-\infty$ | $-\infty$ |
| 2 | 0 | 0 | $-\infty$ |
| 3 | 0 | 0 | 0 |

例如第二个 token 在 mask 后的 scores 若为 `[2, 1, −∞]`，softmax 约为 `[0.731, 0.269, 0]`：未来的第三个 token 权重严格为零，其余权重总和为一。softmax 因而不是“选择唯一 token”，而是产生一个可微的加权读取方案。

### 0.4 residual block 为什么要把输入加回来

注意力或 FFN 不必从头重写整条表示；它只计算需要添加的修正量，再与原输入相加：

$$
Y=X+F(X).
$$

- $X$：子层输入；
- $F$：attention 或 FFN 子层；
- $F(X)$：子层建议写入 residual stream 的增量；
- $Y$：相加后的输出，shape 与 $X$ 完全相同。

若某个二维 token 表示是 `[1, −1]`，子层给出的修正是 `[0.2, 0.3]`，残差输出就是 `[1.2, −0.7]`。这种“保留旧信息、逐层写增量”的结构，正是后文讨论 pre-norm、干净 residual path 和训练稳定性的起点。

### 本章小结

- embedding 把离散 token ID 查表成连续向量，序列因此成为 `[seq, d_model]` 矩阵。
- Q 用来查询，K 用来匹配，V 是实际被加权读取的内容。
- causal mask 阻止未来信息进入当前 token；softmax 把可见位置的 scores 变成概率权重。
- residual block 让每个子层只学习增量，并保持输入输出 shape 一致。

## 1. 如何读懂架构演化：从原始 Transformer 到现代配方

### 1.1 先确定评价轴，而不是背模型名字

讲者先声明本讲采用 survey 视角：比较公开模型的架构选择，再结合论文中的消融实验判断哪些变化较稳健。一个改动是否“好”，至少要同时看三条轴：

- **泛化与最终质量**：给定训练计算量，验证损失或下游能力是否更好；
- **GPU 效率**：每秒能处理多少 token，数据搬运是否成为瓶颈；
- **训练稳定性**：大规模训练是否容易出现 loss spike、溢出或发散。

这三条轴经常互相冲突。例如，某个归一化可能几乎不改变参数量，却增加低 arithmetic-intensity 的数据移动；某个注意力变体可能稍损质量，却显著降低在线服务时的 KV cache 带宽。因而“更新”不等于“更优”，默认值也不等于定律。

### 1.2 原始 Transformer 与现代 decoder-only 模型

2017 年的 Transformer 同时包含 encoder 与 decoder；每个子层后做 LayerNorm，FFN 使用 ReLU，位置由绝对位置嵌入提供。其历史结构如下。

![原始 Transformer 编码器—解码器结构](assets/original-transformer.jpg)
*图 1：原始 Transformer 的 encoder–decoder、post-norm、ReLU 与绝对位置编码（视频 00:01:47–00:02:54）。*

现代大语言模型通常只保留因果 decoder 堆栈，并常见以下组合：pre-norm 或 RMSNorm、无 bias 的线性层、SwiGLU/GeGLU、RoPE，以及针对解码吞吐设计的 GQA。课件用一张总览图把这些差异放在同一页面。

![现代 Transformer 的常见架构选择](assets/modern-transformer.jpg)
*图 2：现代 decoder-only Transformer 的常见组件（视频 00:01:47–00:04:53）。*

这种收敛并非终点。讲者强调，LLaMA 一类公开配方曾使社区暂时趋同，但更大上下文、更低推理成本和更稳定训练又带回 QK norm、局部—全局混合注意力等设计。因此应把“现代架构”理解为一个随约束变化的前沿，而不是固定模板。

### 本章小结

- 架构选择必须放在质量、硬件效率和稳定性三条轴上评价。
- 从原始 Transformer 到现代 LLM，变化集中在 block 排列、归一化、FFN、位置表示和注意力服务成本。
- 公开模型的共同做法是经验起点，不是无需验证的真理。

## 2. Residual stream 与 normalization：让深层网络可训练

### 2.1 Post-norm 与 pre-norm 的真正差别

残差连接的目标，是让信息和梯度能沿一条接近恒等映射的路径跨越很多层。原始 Transformer 在“残差相加之后”归一化，即 post-norm：

$$
x_{\ell+1}=\operatorname{LN}\!\left(x_\ell+F_\ell(x_\ell)\right).
$$

- $x_\ell$：第 $\ell$ 层输入的 residual-stream 表示；
- $x_{\ell+1}$：该子层输出；
- $F_\ell$：第 $\ell$ 层的注意力或 FFN 变换；
- $\operatorname{LN}$：LayerNorm；
- $\ell$：层索引。

现代模型常把归一化移到子层之前，即 pre-norm：

$$
x_{\ell+1}=x_\ell+F_\ell\!\left(\operatorname{N}(x_\ell)\right).
$$

- $x_\ell$、$x_{\ell+1}$：相邻两层的 residual-stream 表示；
- $F_\ell$：注意力或 FFN 子层；
- $\operatorname{N}$：LayerNorm、RMSNorm 等归一化；
- $\ell$：层索引。

![视频中的 pre-norm 与 post-norm 对照](assets/video-pre-post-norm.jpg)
*图 3：讲者比较“先残差后归一化”和“先归一化后残差”（视频 00:07:31–00:09:29）。*

pre-norm 更稳定的直觉来自它保留的“干净残差路径”。对一层求 Jacobian：

$$
\frac{\partial x_{\ell+1}}{\partial x_\ell}
=I+\frac{\partial F_\ell(\operatorname{N}(x_\ell))}{\partial x_\ell}.
$$

- $\partial x_{\ell+1}/\partial x_\ell$：输出对输入的 Jacobian；
- $I$：恒等矩阵，对应残差直通路径；
- $F_\ell$：子层变换；
- $\operatorname{N}$：子层输入上的归一化；
- $x_\ell$：该层输入。

即使第二项的尺度不理想，梯度仍有恒等项 $I$ 可以传播。post-norm 则让归一化处在整条残差路径上，深层网络更依赖 warmup 和初始化细节。课件中的实验显示，post-norm 在缺少足够 warmup 时更容易失稳；pre-norm 更像一个稳健默认值。

> [!WARNING]
> “pre-norm 更稳定”不等于“任何条件下最终质量都更高”。它主要改善优化条件；一些工作会用 sandwich norm、额外缩放或其他 post-norm 变体尝试换取更好表示能力。

### 2.2 LayerNorm 与 RMSNorm

LayerNorm 同时减去均值并按标准差缩放。对单个 token 的 $d$ 维隐藏向量，标准写法是：

$$
\mu(x)=\frac{1}{d}\sum_{i=1}^{d}x_i,
\qquad
\operatorname{LN}(x)_i
=\gamma_i\frac{x_i-\mu(x)}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}(x_j-\mu(x))^2+\varepsilon}}+\beta_i.
$$

- $x\in\mathbb{R}^d$：一个 token 的隐藏向量；
- $x_i$：第 $i$ 个通道；
- $d$：隐藏维度；
- $\mu(x)$：通道均值；
- $\gamma_i$、$\beta_i$：可学习的缩放与平移参数；
- $\varepsilon$：防止分母为零的数值稳定项；
- $i,j$：通道索引。

RMSNorm 删除中心化，只按均方根缩放：

$$
\operatorname{RMSNorm}(x)_i
=\gamma_i\frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\varepsilon}}.
$$

- $x_i$：输入向量第 $i$ 个通道；
- $d$：隐藏维度；
- $\gamma_i$：可学习缩放参数；
- $\varepsilon$：数值稳定项；
- $j$：均方根统计中的通道索引。

![LayerNorm 与 RMSNorm](assets/layernorm-rmsnorm.jpg)
*图 4：课件对照两种归一化；正文采用含 $1/d$ 的标准 RMSNorm 定义（视频 00:13:56–00:14:50）。*

> [!WARNING]
> 课件的 RMSNorm 简写容易被读成 $\sqrt{\sum_i x_i^2}$，漏掉均值因子 $1/d$。标准 RMS 是 root **mean** square；实现还可能在累积精度、epsilon 位置和是否保留可学习缩放上不同。

RMSNorm 少了均值计算和 bias，表达更简洁，但不能只凭 FLOP 数判断它一定更快。归一化的算术量很小，实际可能由读取、写回整条 hidden state 的数据移动主导。

![FLOPs、运行时间与数据移动并不等价](assets/flops-runtime-data-movement.jpg)
*图 5：低 FLOP 操作也可能因数据搬运而占据可观运行时间（视频 00:15:34–00:17:31）。*

这里需要一个系统概念：**arithmetic intensity** 是每搬运一个字节完成多少次算术操作。矩阵乘法能复用数据，强度高；norm、逐元素激活往往强度低。所以现代架构也常去掉线性层 bias：质量上通常无明显收益，同时减少参数、内核分支和数据流复杂度。

### 2.3 一次 block 的信息流

典型 pre-norm decoder block 可写成两步：

$$
h_\ell=x_\ell+\operatorname{Attn}(\operatorname{N}_1(x_\ell)),
\qquad
x_{\ell+1}=h_\ell+\operatorname{FFN}(\operatorname{N}_2(h_\ell)).
$$

- $x_\ell$：block 输入；
- $h_\ell$：注意力残差相加后的中间状态；
- $x_{\ell+1}$：block 输出；
- $\operatorname{Attn}$：因果自注意力；
- $\operatorname{FFN}$：逐 token 前馈网络；
- $\operatorname{N}_1,\operatorname{N}_2$：两处独立的归一化层。

> [!IMPORTANT]
> residual stream 可以理解为贯穿网络深度的共享“信息总线”。注意力负责 token 间通信，FFN 负责每个 token 内的通道变换；两者把增量写回同一条总线。

### 本章小结

- pre-norm 的核心价值是保留带恒等 Jacobian 项的干净残差路径。
- RMSNorm 删除均值中心化，已成为常见默认值，但具体速度取决于内核与数据移动。
- 少 bias、少低强度操作是系统效率选择，不应被包装成纯粹的表示理论结论。
- 对课件公式要核对标准定义，特别是 RMSNorm 的 $1/d$。

## 3. FFN：从激活函数到 gated MLP

### 3.1 为什么不再满足于 ReLU

传统 FFN 先扩宽、激活，再投影回 residual 维度：

$$
\operatorname{FFN}(x)=W_2\,\sigma(W_1x).
$$

- $x\in\mathbb{R}^{d_{\text{model}}}$：输入 token 表示；
- $W_1\in\mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}$：上投影；
- $W_2\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}$：下投影；
- $d_{\text{model}}$：residual-stream 宽度；
- $d_{\text{ff}}$：FFN 中间宽度；
- $\sigma$：ReLU、GELU 等逐元素激活。

ReLU 是 $\max(0,x)$，计算便宜但负半轴梯度为零；GELU 以平滑概率门控近似保留小负值。现代模型更常使用显式 gated 结构，让一条分支生成内容，另一条分支决定每个通道通过多少。

ReGLU 的基本形式为：

$$
\operatorname{ReGLU}(x)=\bigl(\operatorname{ReLU}(xW_g)\odot xW_v\bigr)W_o.
$$

- $x$：输入 token 表示；
- $W_g$：gate 分支投影；
- $W_v$：value/content 分支投影；
- $W_o$：输出投影；
- $\operatorname{ReLU}$：门控分支的激活函数；
- $\odot$：逐元素乘法。

![ReGLU 的双分支门控](assets/reglu-gating.jpg)
*图 6：一条分支产生 gate，另一条分支携带内容，二者逐元素相乘（视频 00:21:50–00:23:23）。*

把 gate 激活替换为 GELU 或 SiLU，就得到 GeGLU 与 SwiGLU：

$$
\operatorname{GeGLU}(x)=\bigl(\operatorname{GELU}(xW_g)\odot xW_v\bigr)W_o,
\qquad
\operatorname{SwiGLU}(x)=\bigl(\operatorname{SiLU}(xW_g)\odot xW_v\bigr)W_o.
$$

- $x,W_g,W_v,W_o,\odot$：含义与 ReGLU 相同；
- $\operatorname{GELU}$：Gaussian Error Linear Unit；
- $\operatorname{SiLU}(z)=z\,\sigma(z)$：Sigmoid Linear Unit；
- $\sigma(z)$：sigmoid 函数。

![GeGLU 与 SwiGLU](assets/geglu-swiglu.jpg)
*图 7：GLU 家族只改变 gate 分支的非线性（视频 00:23:23–00:25:01）。*

### 3.2 为什么 gated FFN 常取传统宽度的 $2/3$

普通 FFN 有两个主要矩阵，参数量近似 $2d_{\text{model}}d_{\text{ff}}$；gated FFN 有 gate、value、output 三个矩阵，参数量近似 $3d_{\text{model}}d_{\text{gate}}$。若希望参数量和矩阵计算预算相近，令两者相等：

$$
3d_{\text{model}}d_{\text{gate}}
\approx 2d_{\text{model}}d_{\text{ff}}
\quad\Longrightarrow\quad
d_{\text{gate}}\approx\frac{2}{3}d_{\text{ff}}.
$$

- $d_{\text{model}}$：模型隐藏宽度；
- $d_{\text{ff}}$：普通两矩阵 FFN 的中间宽度；
- $d_{\text{gate}}$：三矩阵 gated FFN 的中间宽度；
- $2$、$3$：两类 FFN 的主矩阵数量。

> [!IMPORTANT]
> “$2/3$”不是激活函数的神秘常数，而是预算配平。若传统 FFN 取 $d_{\text{ff}}=4d_{\text{model}}$，预算相当的 gated FFN 就约为 $d_{\text{gate}}=\frac{8}{3}d_{\text{model}}$。

实验上，GLU 变体常在相似计算预算下优于普通 ReLU/GELU FFN；但差距、排序和最佳宽度都会随规模与训练设置改变。讲者借此反复强调：用消融支持默认值，而不是把模型家族的流行度当因果证据。

### 本章小结

- FFN 对每个 token 独立地做通道混合；注意力才负责跨 token 通信。
- GLU 用内容分支乘门控分支，使网络能按输入选择通过哪些通道。
- gated FFN 的 $2/3$ 宽度来自三矩阵与两矩阵之间的预算匹配。
- SwiGLU/GeGLU 是有经验支持的强默认值，但仍需在目标规模和系统上验证。

## 4. Serial 还是 parallel：block 排列也是系统设计

### 4.1 串行 block

上一章默认采用串行排列：注意力先更新 residual stream，FFN 再读取更新后的状态。为看清依赖关系，可写成：

$$
h=x+\operatorname{Attn}(\operatorname{N}_1(x)),
\qquad
y=h+\operatorname{FFN}(\operatorname{N}_2(h)).
$$

- $x$：block 输入；
- $h$：注意力后的中间 residual state；
- $y$：block 输出；
- $\operatorname{Attn}$：自注意力分支；
- $\operatorname{FFN}$：前馈分支；
- $\operatorname{N}_1,\operatorname{N}_2$：对应分支前的归一化。

因为 FFN 必须等待注意力结果，两个大分支不能完全并行执行。好处是 FFN 能直接处理注意力刚汇聚的信息，表达路径更深。

### 4.2 并行 block

parallel block 让两个分支读取同一个归一化输入，再把结果一起写回：

$$
y=x+\operatorname{Attn}(\operatorname{N}(x))
  +\operatorname{FFN}(\operatorname{N}(x)).
$$

- $x$、$y$：block 输入与输出；
- $\operatorname{N}$：共享或等价复用的归一化结果；
- $\operatorname{Attn}$：注意力分支；
- $\operatorname{FFN}$：前馈分支。

![串行与并行 Transformer block](assets/serial-parallel-layers.jpg)
*图 8：串行路径具有分支依赖，并行路径可同时计算 attention 与 FFN（视频 00:27:09–00:29:29）。*

并行形式减少了关键路径和一次 normalization/activation 读写机会，适合追求训练吞吐；代价是单个 block 内 FFN 看不到本层注意力刚生成的结果。实际质量差距可能很小，也可能依赖模型规模与并行实现，因此它是“系统—模型协同”的典型例子。

> [!NOTE]
> 这里的 parallel 指一个 block 内的 attention/FFN 分支并行，不是 tensor parallel、pipeline parallel 或 data parallel。后几者描述的是模型如何分布到设备上。

### 本章小结

- serial block 允许 FFN 使用同层注意力输出，但增加顺序依赖。
- parallel block 缩短关键路径，可能改善吞吐，但改变了单层的信息依赖。
- 架构图中一条连线的改变，可能同时影响表示能力、内核调度和通信重叠。

## 5. 从相对位置目标推导 RoPE

### 5.1 为什么内容向量本身不够

自注意力若只比较 token 内容，就无法区分“同一个词出现在第 2 位”和“出现在第 200 位”。位置方案大体包括：把绝对位置向量加到输入、在 attention score 上加相对位置 bias、或直接让 query/key 的几何关系编码相对距离。

![位置表示的主要方案](assets/position-embedding-variants.jpg)
*图 9：绝对位置、相对 bias 与旋转式位置编码的对照（视频 00:31:04–00:33:01）。*

RoPE 想实现的目标可写为：内容 $x,y$ 位于位置 $i,j$ 时，变换后的内积只通过相对位移 $i-j$ 依赖位置：

$$
\left\langle f(x,i),f(y,j)\right\rangle
=g(x,y,i-j).
$$

- $x,y$：两个 token 的内容表示，实际进入注意力时对应 query/key 子向量；
- $i,j$：两个 token 的绝对位置；
- $f(\cdot,\cdot)$：把内容与位置结合的变换；
- $g$：仅依赖内容和相对位移的打分函数；
- $\langle\cdot,\cdot\rangle$：向量内积。

这个目标比“记住一个固定长度的位置表”更贴合注意力：模型做匹配时通常更关心两个 token 相隔多远，而非它们从序列开头数到第几位。

### 5.2 二维旋转如何自动得到相对位置

从二维向量开始最直观。定义旋转矩阵：

$$
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix},
\qquad
f(x,i)=R(i\theta)x.
$$

- $R(\theta)$：逆时针旋转角度 $\theta$ 的二维矩阵；
- $\theta$：该二维通道对的基础旋转频率；
- $i$：位置索引；
- $x$：二维内容向量；
- $f(x,i)$：旋转后带位置信息的向量。

两个旋转后向量的内积为：

$$
\bigl(R(i\theta)x\bigr)^\top R(j\theta)y
=x^\top R((j-i)\theta)y.
$$

- $x,y$：两个二维内容向量；
- $i,j$：各自位置；
- $(\cdot)^\top$：转置；
- $R((j-i)\theta)$：由相对位移决定的旋转。

等式使用 $R(i\theta)^\top R(j\theta)=R((j-i)\theta)$。绝对位置因此在内积中消去，只剩相对位移。

#### 手算一次：平移绝对位置，内积保持不变

先明确真实实现里的 shape：投影并拆头后，query 通常是 `Q.shape = [B, H_q, n, d_h]`，key 是 `K.shape = [B, H_kv, n, d_h]`。在标准 MHA 中 $H_q=H_{kv}$；在 GQA/MQA 中，一个 KV head 会供一组或全部 query heads 使用。RoPE 沿最后的 $d_h$ 维，把它分成 $d_h/2$ 个二维对。下面只取 `B = H_q = H_kv = 1, n = 2, d_h = 2` 的单个二维对手算。

取基础角度 $\theta=\pi/2$，两个内容向量和位置为：

$$
q=(1,0)^\top,
\qquad
k=(0,1)^\top,
\qquad
i=1,
\qquad
j=2.
$$

- $q$：位置 $i$ 上的二维 query；
- $k$：位置 $j$ 上的二维 key；
- $i,j$：两个绝对位置；
- $\theta$：每前进一个位置增加的旋转角度；
- $(\cdot)^\top$：把坐标写成列向量。

分别旋转后：

$$
R(i\theta)q=R(\pi/2)(1,0)^\top=(0,1)^\top,
\qquad
R(j\theta)k=R(\pi)(0,1)^\top=(0,-1)^\top,
$$

- $R(i\theta)q$：位置 1 上旋转后的 query；
- $R(j\theta)k$：位置 2 上旋转后的 key；
- $R(\pi/2)$：逆时针旋转 $90^\circ$；
- $R(\pi)$：逆时针旋转 $180^\circ$。

所以 attention 内积为：

$$
\bigl(R(i\theta)q\bigr)^\top R(j\theta)k
=(0,1)\cdot(0,-1)=-1.
$$

- $\bigl(R(i\theta)q\bigr)^\top R(j\theta)k$：带位置旋转的 Q/K 内积；
- $\cdot$：二维点积；
- $-1$：本例得到的相似度。

现在把两个绝对位置同时向右平移 3，改成 $i'=4,j'=5$，内容向量不变。此时旋转后的 query 为 $(1,0)$，key 为 $(-1,0)$，内积仍为 $-1$。两组位置分别是 $(1,2)$ 与 $(4,5)$，但都有 $j-i=1$；结果相同，直接验证了内积只依赖相对位移。写成 $i-j$ 或 $j-i$ 只是旋转方向的符号约定，关键是绝对位置的共同平移会消去。

![视频中的 RoPE 旋转直觉](assets/video-rope-rotation.jpg)
*图 10：以二维旋转解释为何注意力内积只保留相对角度（视频 00:34:25–00:36:06）。*

### 5.3 扩展到高维：不同通道对使用不同频率

高维 RoPE 不是在“三维小块”上旋转，而是把维度分成许多**二维对**，每一对使用不同频率。对第 $m$ 对维度：

$$
\theta_m=\Theta^{-2m/d_h},
\qquad
\widetilde q_i^{(m)}=R(i\theta_m)q_i^{(m)},
\qquad
\widetilde k_j^{(m)}=R(j\theta_m)k_j^{(m)}.
$$

- $m$：二维通道对索引；
- $d_h$：单个 attention head 的维度；
- $\Theta$：频率基数，常见配方会选择一个较大的常数；
- $\theta_m$：第 $m$ 对通道的旋转频率；
- $q_i^{(m)},k_j^{(m)}$：位置 $i,j$ 上第 $m$ 个二维 query/key 子向量；
- $\widetilde q_i^{(m)},\widetilde k_j^{(m)}$：应用 RoPE 后的 query/key。

![RoPE 的分块旋转矩阵](assets/rope-matrix.jpg)
*图 11：高维向量由多个二维旋转块组成，而非一个高维“整体角度”（视频 00:36:06–00:38:15）。*

快频率通道对区分近距离位置，慢频率通道对承载更长周期结构，类似一组多尺度相位特征。实现时通常用 even/odd 通道重排和预计算的 sine/cosine 完成，不显式构造大矩阵。

![RoPE 的向量化实现](assets/rope-code.jpg)
*图 12：用逐元素 sine/cosine 与通道配对实现旋转（视频 00:38:15–00:39:02）。*

> [!IMPORTANT]
> RoPE 通常只作用于 **Q 和 K**，因为位置需要进入 attention score；V 不必旋转。把 RoPE 加到 residual stream 或 V 上，是不同的设计，不能与标准实现混为一谈。

> [!WARNING]
> 课件/口述中有两个易错点：一个位置示意把第二个 token 的位置也写成了 $i$，语义上应为 $j$；口头提到的“3D pairs”应理解为二维通道对。RoPE 的长上下文外推也并非自动保证，频率基数、训练长度和后续 scaling 方法都会影响结果。

### 本章小结

- 位置表示的关键目标是让 attention score 感知相对位移。
- 二维旋转的内积自然把绝对角度化为角度差，这是 RoPE 的数学核心。
- 高维 RoPE 将 head 维度拆成多个二维对，并使用多尺度频率。
- 标准 RoPE 旋转 Q/K，不旋转 V；它改善位置建模，但不自动解决所有长度外推问题。

## 6. 超参数默认值：经验盆地、预算约束与反例

### 6.1 FFN 宽度：从 $4\times$ 到 $8/3\times$，再到更宽

早期 Transformer 常令普通 FFN 宽度为模型宽度的四倍：

$$
d_{\text{ff}}=4d_{\text{model}}.
$$

- $d_{\text{ff}}$：FFN 中间维度；
- $d_{\text{model}}$：residual-stream 隐藏维度；
- $4$：历史经验倍数。

改用三矩阵 GLU 并保持相近参数预算时，常见换算是：

$$
d_{\text{gate}}\approx\frac{8}{3}d_{\text{model}}.
$$

- $d_{\text{gate}}$：gated FFN 的中间维度；
- $d_{\text{model}}$：模型隐藏维度；
- $8/3$：由 $4\times$ 普通 FFN 乘预算因子 $2/3$ 得到。

![普通 FFN 与 GLU 的宽度换算](assets/glu-ffn-ratios.jpg)
*图 13：两矩阵 FFN 与三矩阵 GLU 在相近预算下的中间宽度（视频 00:44:37–00:46:42）。*

但 T5 等实验也探索了远高于 $4\times$ 的中间维度，并不支持一个尖锐唯一最优值。课件展示的曲线更像宽阔的经验盆地：在固定预算下，一段范围内的性能相近。

![FFN 比例的宽阔经验盆地](assets/ffn-ratio-basin.jpg)
*图 14：FFN 宽度并非只有一个精确最优点（视频 00:47:54–00:49:55）。*

这给出更实用的选择法：先满足张量核友好的倍数与并行切分，再在小规模代理实验中扫一段合理范围；不要为复制论文里的小数比例破坏硬件效率。

### 6.2 Head 维度与模型 aspect ratio

多头注意力常满足：

$$
d_{\text{model}}=H_qd_h.
$$

- $d_{\text{model}}$：模型隐藏宽度；
- $H_q$：query head 数量；
- $d_h$：单个 head 的维度。

实践中 $d_h$ 常取 64、128 一类硬件友好值，但它不是尺度定律。增大 $H_q$ 或 $d_h$ 都能扩宽模型，却会影响 RoPE 频率布局、attention kernel 与 KV cache 形状。

给定参数预算，模型还要在“更深”和“更宽”之间选择。一个常用描述是 aspect ratio：

$$
\rho=\frac{d_{\text{model}}}{L}.
$$

- $\rho$：宽深比；
- $d_{\text{model}}$：隐藏宽度；
- $L$：Transformer block 数量。

![公开模型的宽度与深度选择](assets/aspect-ratio-models.jpg)
*图 15：不同模型族在层数与隐藏宽度之间采用不同折中（视频 00:51:31–00:53:12）。*

![宽深比的实验对照](assets/aspect-ratio-evidence.jpg)
*图 16：固定或近似固定预算下，宽深比存在较宽的可行区间（视频 00:53:12–00:55:08）。*

更深会增加串行关键路径、激活存储和 pipeline 切分难度；更宽能提高大矩阵效率，却加大每层参数、通信与激活。讲者的结论不是“固定某个比值”，而是公开模型落在一片相对宽的区域。

> [!WARNING]
> 课件中一个 PaLM 表格行存在内部不一致：按所列 $48\times258/18432$ 计算约为 $0.672$，不是 $1.48$；其中 258 也很可能是 256。不要把这行小数当作可靠经验常数。

### 6.3 Vocabulary size 与公平比较

词表变大可以缩短序列，却让 embedding 与输出 softmax 变大；词表变小则相反。近年来公开模型的词表常从数万扩到十万乃至更多，原因还包括多语言、代码和工具调用特殊 token。

![公开模型的词表大小](assets/vocabulary-sizes.jpg)
*图 17：不同模型族选择的 vocabulary size 差异很大（视频 00:55:11–00:57:12）。*

不同 tokenizer 下，per-token perplexity 不能直接横比，因为一个 token 覆盖的原始字符或字节数不同。更公平的归一化指标是 bits per byte：

$$
\operatorname{BPB}
=-\frac{1}{B\ln 2}\sum_{t=1}^{T}\log p(x_t\mid x_{<t}).
$$

- $B$：原始文本的字节数；
- $T$：token 数；
- $x_t$：第 $t$ 个 token；
- $x_{<t}$：它之前的 token 前缀；
- $p(x_t\mid x_{<t})$：模型赋予真实下一个 token 的概率；
- $\ln 2$：把自然对数损失换算成 bit 的因子。

> [!NOTE]
> 直观上，BPB 问的是“平均每个原始字节还剩多少不确定性”。它能削弱 tokenizer 粒度差异，但仍不替代同数据、同计算预算的严格比较；多模态 token 更不能简单当普通文本字节处理。

### 6.4 Dropout、weight decay 与训练尺度

大规模预训练通常数据充足、训练轮次少，传统 dropout 的收益较弱，还会妨碍 fused kernel 和确定性。公开配方常把 dropout 设为零，但“几乎无限数据”是前提；小数据微调或反复过拟合同一数据时，它仍可能有用。

weight decay 在 AdamW 中把参数衰减与梯度更新解耦：

$$
\theta_{t+1}=\theta_t-\eta_t\widehat m_t-\eta_t\lambda\theta_t.
$$

- $\theta_t$：第 $t$ 步参数；
- $\theta_{t+1}$：更新后参数；
- $\eta_t$：学习率；
- $\widehat m_t$：经 Adam 二阶统计缩放后的更新方向，此处用简写表示；
- $\lambda$：weight-decay 系数；
- $t$：优化步索引。

![Weight decay 与学习率的联动](assets/weight-decay-learning-rate.jpg)
*图 18：AdamW 中实际衰减强度同时受 $\lambda$ 与学习率调度影响（视频 01:01:39–01:03:24）。*

每步乘法因子近似为 $1-\eta_t\lambda$，因此同一个 $\lambda$ 在不同学习率、训练步数下含义不同。讲者建议把它视为与学习率计划耦合的优化超参数，而不是独立正则化旋钮。

> [!IMPORTANT]
> 选择默认值的顺序应是：先明确预算与服务约束，再找文献给出的可行区间，最后用本任务消融验证。模型表格能告诉你“大家试过什么”，不能证明“为什么有效”。

### 本章小结

- FFN 宽度、head 维度与宽深比通常存在宽阔可行区，而非单点最优。
- 硬件整除、通信和关键路径是选择维度时的一等约束。
- 不同 tokenizer 应用 BPB 等按原始单位归一化的指标比较。
- dropout 是否需要取决于数据复用程度；weight decay 必须与学习率和训练步数一起解释。

## 7. Softmax 稳定性的三个控制点

### 7.1 先定位风险：模型里有两处大 softmax

规模增大后，训练损失偶发尖峰并非小事：一次数值爆炸可能污染 optimizer state，浪费数千张 GPU 的计算。课件先定位两处风险点：输出 vocabulary softmax，以及 attention softmax。

![规模增大后的训练稳定性曲线](assets/training-stability-curves.jpg)
*图 19：更大训练运行可能出现突发 loss spike，需要架构级稳定措施（视频 01:05:01–01:06:24）。*

![语言模型中的两处 softmax](assets/two-softmax-risk-points.jpg)
*图 20：输出层 softmax 与 attention softmax 分别受 logit 尺度影响（视频 01:06:24–01:07:07）。*

softmax 本身可以用“先减最大值”稳定计算，但若上游 logit 范数持续变大，仍会造成极尖分布、低精度下溢、梯度恶化或训练动力学失控。三个常见控制点分别作用在输出 logits、Q/K 向量和 attention logits。

### 7.2 输出端：z-loss

交叉熵只关心各类别 logit 的相对差值：给所有 logit 加同一个常数，概率不变。这留下一个不受任务损失约束的整体漂移方向。z-loss 对 log-partition 加惩罚，让输出尺度不至于无界漂移。若总目标采用最小化约定，可写为：

$$
L=L_{\text{CE}}+\alpha\left(\log\sum_{v=1}^{V}e^{z_v}\right)^2.
$$

- $L$：加入稳定项后的总损失；
- $L_{\text{CE}}$：交叉熵损失；
- $\alpha$：z-loss 权重；
- $z_v$：词表第 $v$ 项的输出 logit；
- $V$：词表大小；
- $\log\sum_v e^{z_v}$：log-partition，也称 log-sum-exp。

![z-loss 约束输出 log-partition](assets/z-loss.jpg)
*图 21：z-loss 控制输出 softmax 前的整体 logit 尺度（视频 01:07:07–01:09:28）。*

> [!WARNING]
> 若论文把目标写成要**最大化**的 log-likelihood，正负号会与上式相反。判断 z-loss 的作用应看它是否惩罚过大的 log-partition，而不能脱离优化约定机械背符号。

### 7.3 注意力输入端：QK norm

标准 scaled dot-product attention 为：

$$
S=\frac{QK^\top}{\sqrt{d_h}},
\qquad
A=\operatorname{softmax}(S),
\qquad
O=AV.
$$

- $Q,K,V$：query、key、value 矩阵；
- $d_h$：单个 head 维度；
- $S$：softmax 前的 attention logits；
- $A$：归一化后的 attention 权重；
- $O$：attention 输出。

即使除以 $\sqrt{d_h}$，训练过程中 Q/K 范数仍可能增长，使 $S$ 变得极端。QK norm 在点积前分别归一化 Q 与 K：

$$
\widehat Q=\operatorname{N}_q(Q),
\qquad
\widehat K=\operatorname{N}_k(K),
\qquad
S=\frac{\widehat Q\widehat K^\top}{\sqrt{d_h}}.
$$

- $Q,K$：线性投影得到的原始 query/key；
- $\operatorname{N}_q,\operatorname{N}_k$：head 维度上的归一化，可带独立参数；
- $\widehat Q,\widehat K$：归一化后的 query/key；
- $S$：受控的 attention logits；
- $d_h$：单个 head 维度。

![QK norm 在注意力路径中的位置](assets/qk-norm-flow.svg)
*图 22：QK norm 作用于投影后的 Q/K、点积之前；不要与 block 输入的 pre-norm 混淆（视频 01:09:28–01:12:07）。*

> [!IMPORTANT]
> block pre-norm 控制进入整个 attention/FFN 子层的 residual state；QK norm 专门控制 attention score 的两个输入。二者解决的数值路径不同，可以同时存在。

### 7.4 注意力 logit 端：soft-cap

另一条路线是直接把 attention logits 压到有限范围。常见平滑裁剪为：

$$
\widetilde S=c\tanh\left(\frac{S}{c}\right),
\qquad
A=\operatorname{softmax}(\widetilde S).
$$

- $S$：原始 attention logits；
- $c>0$：soft-cap 阈值；
- $\tanh$：双曲正切函数；
- $\widetilde S$：平滑限制在约 $[-c,c]$ 的 logits；
- $A$：attention 权重。

![Attention logit soft-cap](assets/logit-soft-cap.jpg)
*图 23：用 $c\tanh(S/c)$ 平滑限制 attention logit（视频 01:12:07–01:13:56）。*

当 $|S|\ll c$ 时，$\tanh(S/c)\approx S/c$，变换近似恒等；当 $|S|$ 很大时才逐渐饱和。它能提高稳定性，却也可能阻止模型形成非常尖锐的注意力，课件中的结果提示 perplexity 可能受损。因而这是稳定性—表达力的直接交换。

### 本章小结

- 输出 softmax 与 attention softmax 是两条独立的尺度风险路径。
- z-loss 约束输出 log-partition；QK norm 控制点积输入；soft-cap 控制点积结果。
- 数值稳定措施并非免费，可能改变优化目标或限制注意力尖锐度。
- 实践中应先监控 logit/QK 范数与 loss spike，再选择最小必要干预。

## 8. Decode 为什么 memory-bound：KV cache、MQA 与 GQA

### 8.1 Prefill 和 decode 是两种不同工作负载

输入一个长度为 $n$ 的 prompt 时，prefill 会并行处理全部 token。单层投影和 attention 的主要量级可概括为：

$$
\operatorname{cost}_{\text{prefill}}
=O(nd_{\text{model}}^2)+O(n^2d_hH_q).
$$

- $n$：prompt token 数；
- $d_{\text{model}}$：模型隐藏宽度；
- $d_h$：单个 attention head 维度；
- $H_q$：query head 数；
- $O(nd_{\text{model}}^2)$：Q/K/V/O 与 FFN 等大线性投影的代表性量级；
- $O(n^2d_hH_q)$：全注意力 score 与 value 聚合的序列二次项。

![Prefill 阶段的注意力计算](assets/attention-prefill-cost.jpg)
*图 24：prefill 同时处理整段 prompt，矩阵较大、并行度高（视频 01:15:16–01:16:59）。*

大矩阵乘法有较高数据复用，prefill 往往更接近 compute-bound。注意不要把所有“平方”混为一谈：$d_{\text{model}}^2$ 来自宽度方向投影，$n^2$ 才来自 token 两两交互。

生成阶段每次只新增一个 token。若把历史 token 的 K/V 保存下来，第 $t$ 步只需计算新 token 的 Q/K/V，再让新 Q 读取历史 K/V：

$$
q_t=x_tW_Q,
\qquad
k_t=x_tW_K,
\qquad
v_t=x_tW_V,
$$

- $x_t$：第 $t$ 个新 token 的隐藏状态；
- $W_Q,W_K,W_V$：query、key、value 投影矩阵；
- $q_t,k_t,v_t$：新 token 的 query、key、value。

#### 从 hidden state 到 KV cache：逐步看 shape

用带 batch 的标准布局表示，一层输入 hidden state 的 shape 是 `[B, n, d_model]`。投影和拆头并不是“再创造一份序列”，而是把最后的特征维组织成 head：

1. 输入：`X.shape = [B, n, d_model]`。
2. Q 投影：先得到 `[B, n, H_q × d_h]`，再重排为 `[B, H_q, n, d_h]`。
3. K/V 投影：各先得到 `[B, n, H_kv × d_h]`，再重排为 `[B, H_kv, n, d_h]`。
4. cache：每一层分别保存 `K_cache` 与 `V_cache`，shape 都是 `[B, H_kv, cached_seq, d_h]`。

这里 $d_{\text{model}}=H_qd_h$ 是常见设置；GQA/MQA 中 $H_{kv}<H_q$，所以 K/V 投影宽度和 cache 都会变小。一个具体例子是 `B=1, H_q=8, H_kv=2, d_h=64, d_model=512, n=3`：

| 张量 | shape | 含义 |
|---|---|---|
| hidden `X` | `[1, 3, 512]` | 三个 token 的 residual 表示 |
| `Q` | `[1, 8, 3, 64]` | 八个 query heads |
| `K`、`V` | 各 `[1, 2, 3, 64]` | 两个共享 KV heads |
| 每层 KV cache | 两份 `[1, 2, 3, 64]` | 一份 K、一份 V |

decode 新增第 4 个 token 时，当前 hidden shape 是 `[1, 1, 512]`。它产生 `Q_new: [1, 8, 1, 64]` 与 `K_new/V_new: [1, 2, 1, 64]`；只把新 K/V 沿 `cached_seq` 轴追加，cache 从 `[1, 2, 3, 64]` 变为 `[1, 2, 4, 64]`。`Q_new` 只用于当前步，不写入 cache。随后每个 query head 与它所属 KV group 的四个历史 keys 做点积，score shape 为 `[1, 8, 1, 4]`。

随后计算：

$$
o_t=\operatorname{softmax}\!\left(\frac{q_tK_{\le t}^{\top}}{\sqrt{d_h}}\right)V_{\le t}.
$$

- $o_t$：第 $t$ 步 attention 输出；
- $q_t$：当前 token 的 query；
- $K_{\le t},V_{\le t}$：截至当前步缓存的全部 key/value；
- $d_h$：单个 head 维度；
- $\operatorname{softmax}$：沿历史位置归一化。

![Decode 阶段的 KV 读取成本](assets/attention-decode-cost.jpg)
*图 25：decode 每步 query 很小，却要反复读取随上下文增长的 K/V cache（视频 01:16:59–01:19:36）。*

> [!WARNING]
> **KV cache 缓存的是 K 和 V，不是 Q，也不是 QK attention matrix。** Q 只服务当前生成步，attention score 每步根据当前 Q 与历史 K 重新计算。课堂动画/口述在这里容易造成误解，正文采用标准实现解释。

每个请求、每层的缓存元素量近似为：

$$
N_{\text{KV}}=2nH_{kv}d_h.
$$

- $N_{\text{KV}}$：K 与 V 的缓存元素总数；
- $2$：分别对应 K 和 V；
- $n$：已缓存上下文长度；
- $H_{kv}$：key/value head 数；
- $d_h$：每个 KV head 维度。

decode 的单步矩阵很瘦，难以充分复用读入的 K/V，常由显存带宽而非峰值 FLOPs 限制。因此缩小 $H_{kv}$ 直接改善并发数、带宽和延迟。

### 8.2 MHA、MQA 与 GQA

标准 multi-head attention（MHA）让每个 query head 都有独立 K/V head：

$$
H_{kv}=H_q.
$$

- $H_{kv}$：key/value head 数；
- $H_q$：query head 数。

multi-query attention（MQA）让所有 query head 共享一组 K/V：

$$
H_{kv}=1.
$$

- $H_{kv}$：共享的 KV head 数；
- $1$：所有 query head 共用一组 K 和 V。

grouped-query attention（GQA）位于二者之间，让若干 query head 共享一个 KV head：

$$
1<H_{kv}<H_q,
\qquad
g=\frac{H_q}{H_{kv}}.
$$

- $H_q$：query head 数；
- $H_{kv}$：KV head 数；
- $g$：每个 KV head 服务的 query head 数，通常要求整除。

![视频中的 MHA、GQA 与 MQA 对照](assets/video-mha-gqa-mqa.jpg)
*图 26：从每个 Q head 独享 K/V，过渡到分组共享，再到全共享（视频 01:20:49–01:21:59）。*

> [!WARNING]
> MQA 的“one dimension”应理解为**一个共享 KV head**，不是把 key/value 压成单个标量维。$d_h$ 仍是完整 head 维度。

从 MHA 改成 MQA/GQA 并不是让 FLOPs 凭空消失，而是减少缓存读取，使每字节数据对应更多 query-head 计算，即提高 arithmetic intensity。讲者在现场先说反、随后自我纠正；正确方向是：共享 K/V 通常提高而非降低解码阶段的 arithmetic intensity。

![MQA/GQA 的质量与延迟结果](assets/mqa-gqa-quality-latency.jpg)
*图 27：GQA 常取得接近 MHA 的质量和接近 MQA 的服务收益，但差异并非恒为零（视频 01:21:59–01:23:14）。*

图中一个 MQA 结果从约 29.9 变化到 30.2，差距虽小却说明质量并非数学保证。选择 $H_{kv}$ 应结合目标 batch、上下文长度、显存容量和评测，而不能只看架构名称。

### 8.3 一个可操作的服务估算

若元素采用 $b$ 字节、模型有 $L$ 层，则单请求 KV cache 近似为：

$$
M_{\text{KV}}=2LnH_{kv}d_hb.
$$

- $M_{\text{KV}}$：单请求 KV cache 字节数；
- $L$：层数；
- $n$：上下文 token 数；
- $H_{kv}$：KV head 数；
- $d_h$：head 维度；
- $b$：每个缓存元素的字节数；
- $2$：K 与 V 两份缓存。

这个估算先忽略 allocator 分页、量化元数据和 padding，却足以做架构比较：在其他量不变时，GQA 相对 MHA 的 cache 比例约为 $H_{kv}/H_q$。

> [!IMPORTANT]
> 训练吞吐好不代表在线 decode 好。评估服务架构时至少分开测 prompt prefill 延迟、单 token decode 延迟、可并发 batch 和长上下文显存占用。

### 本章小结

- prefill 具有大矩阵和高并行度；decode 每步很小，却持续读取历史 K/V，常受显存带宽限制。
- KV cache 只保存 K/V，使生成免于重算历史表示，但内存随层数和上下文线性增长。
- MQA/GQA 通过减少 KV head 数降低 cache 与带宽；GQA 在质量和效率间提供连续旋钮。
- 质量接近不是理论保证，最终选择必须以目标服务负载实测。

## 9. 长上下文：稀疏、滑动窗口与局部—全局混合

### 9.1 全注意力为什么难扩展

长度为 $n$ 的 full causal attention 需要约 $n(n+1)/2$ 个可见连接，复杂度为 $O(n^2)$。当上下文扩到数十万 token，二次项会压过其他计算与内存成本。

sparse attention 的目标是让每个 token 只连接一部分位置，例如局部邻域、固定步长或少量全局 token。

![稀疏注意力连接模式](assets/sparse-attention-patterns.jpg)
*图 28：局部、跨步与全局连接可组合成稀疏模式（视频 01:25:08–01:25:57）。*

最简单的 sliding-window attention（SWA）只看最近 $w$ 个 token，连接数从二次降为线性量级：

$$
\operatorname{cost}_{\text{SWA}}=O(nwd_hH_q).
$$

- $n$：序列长度；
- $w$：滑动窗口大小；
- $d_h$：单个 head 维度；
- $H_q$：query head 数；
- $O(nwd_hH_q)$：窗口注意力 score 与聚合的量级。

当 $w\ll n$ 时节省巨大，但任意两个远距离 token 无法在一层直接通信。堆叠多层会扩大感受野，却仍可能使精确长程检索更困难。

### 9.2 交错 full 与 local attention

当前常见折中是多数层用 SWA，隔若干层插入一次 full attention。局部层便宜地处理邻近模式，全局层周期性地重新连通整个序列。

![视频中的局部—全局交错结构](assets/video-interleaved-attention.jpg)
*图 29：示例每四层插入一层 full attention，其余使用 sliding-window attention（视频 01:25:57–01:27:13）。*

若每 $r$ 层中有一层 full attention，粗略量级为：

$$
\operatorname{cost}_{\text{hybrid}}
\approx O\!\left(\left(L-\frac{L}{r}\right)nwd_hH_q
+\frac{L}{r}n^2d_hH_q\right).
$$

- $L$：总层数；
- $n$：序列长度；
- $w$：局部窗口大小；
- $r$：full-attention 层的间隔；
- $d_h$：head 维度；
- $H_q$：query head 数；
- 第一项：约 $L-L/r$ 个局部注意力层的成本；
- 第二项：约 $L/r$ 个全注意力层的成本。

这个公式是理解量级的教学近似；FlashAttention、padding、序列打包和并行策略都会改变常数。

![近期模型中的混合注意力配方](assets/recent-hybrid-attention.jpg)
*图 30：公开模型采用不同的 full/local 比例、位置编码和窗口方案（视频 01:27:13–01:28:38）。*

课件以 Command A 等模型说明：局部层可配 RoPE 以表达短程相对位置，全局层有时使用 NoPE；其他公开模型则采用 full RoPE。这里最重要的不是背某个版本列表，而是理解三个独立旋钮：**哪些层全局、窗口多大、位置编码作用在哪些层。** 模型版本更新很快，表格应视作授课时点的快照。

> [!IMPORTANT]
> GQA 与 SWA 优化不同维度：GQA 减少每个历史位置保存/读取的 KV head，SWA 减少需要访问的历史位置数。二者可以叠加。

### 9.3 如何验证长上下文架构

只测短序列 perplexity 会漏掉核心风险。合理验证至少包含：

- 不同长度下的 prefill/decode 延迟与峰值显存；
- 长程检索、跨段推理与局部语言建模分别评测；
- full 层间隔和窗口大小的消融；
- 训练长度内与长度外的 RoPE/位置外推稳定性；
- 相同服务吞吐或相同计算预算下的质量比较。

### 本章小结

- full attention 的序列二次项限制超长上下文扩展。
- SWA 将连接规模降到 $O(nw)$，代价是单层缺少远距离直接通信。
- 局部—全局交错用少量 full 层恢复全局信息流，是质量与成本的折中。
- GQA、SWA 和位置编码分别控制 KV 通道数、可见位置数与位置几何，可以联合设计。

## 总结与延伸

### 一张决策地图

本讲的组件不是一袋互不相关的 tricks。它们分别作用在 Transformer 的不同瓶颈上：

| 目标 | 主要组件 | 解决的问题 | 需要警惕的代价 |
|---|---|---|---|
| 深层优化 | pre-norm、RMSNorm、干净残差路径 | 梯度传播与训练稳定 | 最终质量并非总是占优 |
| FFN 表达 | SwiGLU/GeGLU、宽度比例 | 条件通道选择 | 三矩阵增加参数，需按预算缩宽 |
| 位置信息 | RoPE | attention score 的相对位置几何 | 长度外推依赖频率和训练范围 |
| 数值稳定 | z-loss、QK norm、soft-cap | 控制两处 softmax 的 logit 尺度 | 额外约束可能损失表达或 perplexity |
| 解码效率 | KV cache、MQA/GQA | 降低重复计算、缓存与带宽 | 过度共享 KV 可能损质量 |
| 长上下文 | SWA、full/local hybrid | 缓解 $n^2$ 注意力成本 | 远程信息路径变长、需专门评测 |

### 五个可以迁移到新模型的判断问题

1. **改动优化的是哪一个目标？** 是训练 loss、稳定性、训练吞吐，还是在线 decode？
2. **比较是否预算公平？** 参数、token、FLOPs、wall-clock 与服务吞吐不能混作一个尺度。
3. **收益来自数学结构还是硬件实现？** FLOP 更少不保证更快，低 arithmetic-intensity 操作可能由带宽主导。
4. **证据覆盖了目标工作负载吗？** 短序列预训练结果不能证明长上下文服务最优。
5. **所谓默认值有多宽的容忍区？** 若性能盆地平坦，应优先选择更利于整除、并行和内核融合的点。

### 建议的延伸练习

- 在同一小模型上实现 post-norm 与 pre-norm，记录不同 warmup 下的梯度范数和 loss spike。
- 以相同参数量比较 GELU FFN 与 SwiGLU，而不是直接使用相同中间宽度。
- 手写二维 RoPE，数值验证旋转后内积只依赖 $i-j$；再扩展到多频率通道对。
- 用 $M_{\text{KV}}=2LnH_{kv}d_hb$ 估算 MHA、GQA、MQA 的单请求缓存，并与真实服务 profiler 对照。
- 在 needle retrieval 与语言建模上分别扫描 SWA 窗口和 full-layer 间隔，观察效率—长程能力前沿。

### 最终结论

现代 Transformer 的主干之所以看似稳定，是因为许多选择已经在质量、稳定性和硬件效率之间找到宽阔可行区；它之所以仍在快速变化，是因为训练规模、上下文长度和在线服务约束不断移动。真正可迁移的知识不是“某模型用了什么”，而是能从 residual path、logit 尺度、数据移动和信息可达性四个层面解释：**一项设计为什么可能有效、在哪种负载下有效、又可能在哪里失败。**

### 本章小结

- 把架构组件映射到具体瓶颈，才能避免无目的堆叠技巧。
- 默认配方适合作为实验起点，不可替代预算公平的消融和目标负载测试。
- 模型、优化器和硬件共同决定最终系统；架构研究本质上也是系统研究。
