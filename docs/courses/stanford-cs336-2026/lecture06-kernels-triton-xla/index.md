# CS336 Lecture 6：GPU Kernel、Triton 与编译器视角

![课程视频封面](assets/cover.jpg)

> [!NOTE]
> **课程**：Stanford CS336 — Language Modeling from Scratch（Spring 2026）  
> **讲次**：Lecture 6: Kernels, Triton, XLA  
> **讲者**：Stanford CS336 课程团队  
> **视频**：[YouTube 原视频](https://www.youtube.com/watch?v=xnDHaNUvHBg)  
> **时长**：01:26:41  
> **材料**：完整视频、人工英文字幕、官方 `lecture_06.py` 与课堂图示

> [!IMPORTANT]
> 视频标题含有 **XLA**，但本次实际课堂、完整人工字幕和官方源码都没有展开 XLA/JAX。第 9 章会把 XLA 放回编译栈中，作为明确标注的课外补充；其余章节严格总结视频实际讲授的 GPU kernel、性能分析与 Triton。

在阅读本讲义之前，我们假设读者已经熟悉 PyTorch 的张量运算与自动求导，能够写出标准的矩阵乘法和逐元素表达式，但未必深入过 GPU 体系结构。本讲恰好位于"会写模型"与"会让模型快"之间的门槛上：它不要求读者手写 CUDA，但要求读者建立一套可以定量推理 GPU 性能的语言。我们将沿着课堂的实际顺序展开——先建立硬件词汇，再用测量工具定位瓶颈，然后用 Triton 写出四种结构递进的 kernel，最后把视野抬升到编译器栈。每一章都遵循同一条论证链：先给出动机，再建立数学或定量模型，然后用具体数字验算，最后讨论工程含义。

## 1. 本讲路线：从“能算对”到“能跑快”

训练语言模型时，我们通常先关注数学是否正确：矩阵乘法、归一化、激活函数能不能得到预期结果。进入系统层后，问题会变成：同一个数学表达式，为什么有时快十倍，有时却被内存访问、调度或固定开销拖住？

这个问题之所以困难，是因为 GPU 程序的运行时间并不由单一因素决定。一段 PyTorch 代码的时间可以分解为几个相互独立的成分：算术运算本身消耗的时间、数据在存储层级之间搬运的时间、kernel 启动与调度的固定开销，以及并行硬件未能被填满时的空闲。前两项随问题规模增长，后两项在小规模时占主导。不理解这个分解，就无法解释"为什么矩阵扩大一倍，时间却几乎不变"这类看似违反渐近复杂度的现象。

本讲给出的工作方式不是“背最快实现”，而是一个循环：

![性能优化闭环：测量、解释、改写、复测](assets/performance-loop.svg)

讲义重绘，依据课堂 `00:21:49--00:22:36` 的 “recipe for success”。

1. **建立基准**：先测真实形状、真实硬件上的运行时间。基准的意义在于为后续一切判断提供锚点；没有基准的优化只是猜测。
2. **使用 profiler**：确认时间花在计算、内存传输还是 kernel launch。Profiler 把一次前向或算子调用拆成逐个 kernel 的时间线，使"慢"从形容词变成可定位的结构。
3. **形成硬件解释**：用存储层级、warp、occupancy、coalescing 等概念解释瓶颈。解释必须能推出可检验的预测，例如"若瓶颈是 HBM 带宽，则增大 tile 应当有效；若瓶颈是 launch 开销，则增大 tile 无效"。
4. **改写 kernel**：通过 fusion、tiling、数据复用和合理映射减少浪费。改写不是盲目尝试，而是针对第三步定位的瓶颈选择对应的杠杆。
5. **重新测量**：优化是否有效，只能由新的测量回答。任何不经复测就宣称成功的优化都不可信。

这个闭环的精神与科学实验一致：测量是观察，硬件解释是假说，改写是干预，复测是检验。本讲后面的每一个案例——GeLU、softmax、row sum、matmul——都会完整走一遍这个循环，读者应当把注意力放在循环本身，而不是某个具体数字上。

这也解释了本讲的组织顺序：先建立 GPU 性能语言（第 2 章），再介绍测量方法论（第 3 章），然后用 GeLU 做性能侦探（第 4 章），最后逐步写出 Triton GeLU、softmax、row sum 和 tiled matmul（第 5 至 8 章）。四个 Triton 例子的选择并非随意：它们分别对应一维逐元素、单 tile 归约、跨 tile 归约、二维数据复用四种结构，覆盖了绝大多数实际 kernel 的骨架。

> [!WARNING]
> “用了 Triton”不等于“必然更快”。Triton 是表达高效 kernel 的工具；具体实现是否更快，仍取决于输入形状、GPU、编译器生成代码和已有库函数的质量。

### 本章小结

- Kernel 优化是一套“测量—解释—改写—复测”的闭环。
- 本讲的主线是用硬件知识指导实现，而不是凭语言或框架名称判断性能。
- 正确性是起点；数据移动、并行映射和 launch 次数常常决定最终速度。

## 2. GPU 性能语言：存储层级、warp 与调度

本章的目标是建立一套可以定量使用的词汇。我们将看到，GPU 性能的绝大多数反直觉现象，都可以归结为少数几个概念的推论：存储层级之间的带宽悬殊、warp 作为调度单位、寄存器与 shared memory 对驻留并发的约束、地址模式对内存事务的放大，以及 grid 尺寸相对硬件规模的取整效应。这些概念彼此独立，但在真实 kernel 中同时起作用，因此必须分别理解、组合使用。

### 2.1 为什么首先要理解存储层级

GPU 不是一块均匀的“并行计算器”。一个现代 GPU 包含许多 Streaming Multiprocessor（SM）；每个 SM 内有标量/向量执行单元、寄存器文件、shared memory 与 L1 cache，芯片上还有共享的 L2 cache，最外层才是容量最大的 HBM。

![GPU 硬件与存储层级（官方课件图）](assets/gpu-hardware.png)

课堂对应区间：`00:00:30--00:00:53`。

层级越靠近计算单元，通常容量越小、带宽越高、访问延迟越低。优化 kernel 的核心问题之一，就是尽可能让已经搬入片上存储的数据被重复利用，而不是一遍遍回到 HBM。

为了把这句话变成定量判断，我们需要一个统一的度量：**算术强度**（arithmetic intensity），定义为完成全部浮点运算所需搬运的字节数去除浮点运算总数，单位为 FLOP/byte：

$$
\mathrm{AI}=\frac{\mathrm{FLOPs}}{\mathrm{Bytes\ moved}}.
$$

- $\mathrm{AI}$：算术强度，每从内存搬运一个字节所能完成的浮点运算数。
- $\mathrm{FLOPs}$：kernel 的浮点运算总量。
- $\mathrm{Bytes\ moved}$：kernel 执行期间在目标存储层级（通常取 HBM）上搬运的总字节数。

给定硬件的峰值算力 $P$（FLOP/s）与峰值带宽 $W$（byte/s），一个 kernel 可达到的时间下界为：

$$
T\ge \max\!\left(\frac{\mathrm{FLOPs}}{P},\ \frac{\mathrm{Bytes\ moved}}{W}\right).
$$

- $T$：kernel 的实际运行时间。
- $P$：硬件峰值浮点算力。
- $W$：目标层级的峰值带宽。

当 $\mathrm{AI}<P/W$ 时，第二项主导，kernel 是**带宽受限**（memory-bound）的；反之则是**算力受限**（compute-bound）的。逐元素算子的算术强度通常极低——例如 $y=2x$ 对 `float32` 每搬运 8 字节（读 4 写 4）只做 1 次乘法，$\mathrm{AI}=0.125$ FLOP/byte——因此几乎永远被带宽卡住；而大矩阵乘法的算术强度随规模增长，才有机会吃满算力。这个分界就是著名的 roofline 模型，本讲后面所有"为什么慢"的分析本质上都是把具体 kernel 放进这张图里定位。

课堂用 A100、H100、B200 的数据直观展示了这种不均衡：

![A100、H100、B200 的 SM、容量与带宽对比](assets/gpu-generation-memory-bandwidth-table.jpg)

视频原帧，字幕对应区间：`00:01:05--00:02:59`。

| 指标 | A100 | H100 | B200 |
|---|---:|---:|---:|
| SM 数 | 108 | 132 | 148 |
| 每 SM 寄存器容量 | 256 KB | 256 KB | 256 KB |
| 每 SM L1 + shared memory | 192 KB | 256 KB | 256 KB |
| L2 cache | 40 MB | 50 MB | 96–126 MB |
| HBM 容量 | 80 GB | 80 GB | 192 GB |
| 估算寄存器带宽 | 约 116 TB/s | 约 401 TB/s | 约 447 TB/s |
| 估算 HBM 带宽 | 约 2 TB/s | 约 3.35 TB/s | 约 8 TB/s |

这张表最值得注意的不是绝对数字，而是**比值**。以 H100 为例，寄存器带宽约 401 TB/s，HBM 带宽约 3.35 TB/s，相差约两个数量级。换句话说，同一个数据元素，放在寄存器里被反复使用，与每次都从 HBM 取回，成本相差约百倍。这些数不是让我们机械记忆，而是在提醒：如果一个 elementwise 算子的每一步都把中间结果写回 HBM，那么算术本身再便宜，也可能被数据移动完全淹没。

由此可以直接推出本讲第一条工程准则：**衡量一个算子链的成本时，先数 HBM 往返次数，再谈算术量**。第 4 章的 GeLU 案例将证明，这条准则足以解释五倍以上的实测差距。

### 2.2 Grid、block、thread 与数据作用域

一个 CUDA-style kernel launch 会创建一个 grid，grid 由许多 thread block 组成，block 再包含许多 thread。这个层级不只是命名方式，还规定了协作边界：

- **Grid** 覆盖整个问题；不同 block 的执行先后通常不能依赖。
- **Block / CTA** 被整体调度到某个 SM，在 block 内可用 shared memory 和同步原语协作。
- **Thread** 拥有自己的逻辑索引和寄存器状态。
- **Global memory / HBM** 可被整个 grid 访问，但往返成本最高。
- **Shared memory** 由同一 block 内线程共享，容量较小但更靠近计算。
- **Register** 通常是线程私有、最快的存储资源，却也会限制 occupancy。

作用域与硬件层级的对应关系是理解一切映射问题的基础。寄存器私有于线程，意味着跨线程交换数据必须借助 shared memory 或更外层；shared memory 私有于 block，意味着跨 block 的归约无法在不回写 global memory 的情况下完成；block 之间的执行顺序不受保证，意味着任何"先算完所有 block 的部分和，再求总和"的算法都必须拆成两个 kernel，或者依赖原子操作。我们在第 7 章会看到，正是"block 之间不能廉价协作"这一事实，决定了跨 tile 归约要放在单个 program 的循环内部完成，而不是拆给多个 program 再合并。

一个 block 在执行期间不能被随意拆到多个 SM。某个 SM 可以同时驻留多个 block，但数量要受寄存器、shared memory、线程数和架构上限共同约束。这个约束的定量形式将在 2.4 节展开，它是 occupancy 计算的全部内容。

### 2.3 Warp：GPU 实际发射指令的基本群体

CUDA 编程模型里，我们写 thread、block 和 grid；硬件执行时，同一个 SM 会以 **warp** 为基本调度单位。NVIDIA GPU 上，一个 warp 通常包含 32 个线程。

这个设计有一个深远的后果：**同一个 warp 内的 32 个线程共享一条指令流**。硬件为整个 warp 取指、译码、发射一次，32 个线程各自在自己的寄存器和数据上执行同一条指令，即 SIMT（Single Instruction, Multiple Threads）模型。它的优点是取指与调度开销被 32 路摊薄；代价是灵活性受限——当 warp 内线程想做不同的事情时，硬件无法在同一周期同时满足。

若同一 warp 内的线程走不同控制流分支，例如一半执行 `if`、另一半执行 `else`，硬件往往需要分阶段执行两个分支并屏蔽不参与的线程。这叫 **warp divergence**。

可以用一个简单模型估计 divergence 的代价。设两个分支的指令数分别为 $I_1$ 与 $I_2$，线程按 $p$ 与 $1-p$ 的比例分布。无分歧时（整个 warp 走同一分支），执行代价为 $\max(I_1,I_2)$ 中的一段；发生分歧时，硬件需要串行执行两段，代价为：

$$
C_{\mathrm{div}}=I_1+I_2,
$$

- $C_{\mathrm{div}}$：分歧 warp 的实际执行代价。
- $I_1,I_2$：两个分支各自的指令数。

注意代价与比例 $p$ 无关：哪怕 31 个线程走 `if`、只有 1 个线程走 `else`，两段也必须都执行一遍，那个落单线程所在的分支以 $1/32$ 的利用率独占一段发射槽。因此在写带分支的 kernel 时，关键不是"分支比例"，而是"分支边界是否与 warp 边界对齐"。

> [!IMPORTANT]
> 分支本身不一定昂贵；真正危险的是同一 warp 内线程发生分歧。若整个 warp 都选择同一分支，通常不会产生同样的串行化代价。

由此得到一条实用的写法建议：让控制流条件尽可能按 warp 粒度一致。例如按"行号是否越界"分支通常安全（一整行由若干完整 warp 处理，越界判断在行粒度一致），而按"元素值是否为负"分支则几乎必然在每个 warp 内制造分歧。Triton 的向量化抽象在相当程度上替我们规避了手写分歧控制流的机会，但理解这一点对阅读 PTX 与诊断慢速 kernel 仍然必要。

### 2.4 Occupancy：有多少 warp 能同时驻留

Occupancy 衡量一个 SM 上实际驻留的 active warp 数，相对于硬件允许的最大 active warp 数的比例：

$$
\mathrm{occupancy}
=
\frac{N_{\mathrm{active\ warps}}}
{N_{\mathrm{max\ warps}}}.
$$

- $N_{\mathrm{active\ warps}}$：因寄存器、shared memory、block 数等资源约束后，当前可驻留的 warp 数。
- $N_{\mathrm{max\ warps}}$：该架构单个 SM 支持的最大驻留 warp 数。

为什么要关心这个比例？因为 GPU 隐藏内存延迟的方式与 CPU 完全不同。CPU 依靠大缓存、分支预测和乱序执行把延迟"消化"在单个线程内部；GPU 的单个 warp 是顺序执行的，遇到一次 HBM 访问就要等待数百个周期。GPU 的对策是让 SM 同时驻留许多 warp：当一个 warp 发出内存请求进入等待，调度器立即切换到另一个已就绪的 warp 继续发射指令。只要驻留 warp 足够多，内存延迟就能被其他 warp 的计算填满——这就是 **latency hiding**。Occupancy 正是"有多少 warp 可供切换"的度量。

#### Occupancy 的三重资源约束

一个 SM 能驻留多少 block，由三类资源的**最小值**决定：

$$
B_{\mathrm{active}}
=
\min\!\left(
B_{\mathrm{reg}},\,
B_{\mathrm{smem}},\,
B_{\mathrm{arch}}
\right),
$$

- $B_{\mathrm{active}}$：单个 SM 实际可同时驻留的 block 数。
- $B_{\mathrm{reg}}$：寄存器总量允许的 block 数，$B_{\mathrm{reg}}=\lfloor R_{\mathrm{SM}}/R_{\mathrm{block}}\rfloor$。
- $B_{\mathrm{smem}}$：shared memory 总量允许的 block 数，$B_{\mathrm{smem}}=\lfloor S_{\mathrm{SM}}/S_{\mathrm{block}}\rfloor$。
- $B_{\mathrm{arch}}$：架构硬上限允许的 block 数，即 $B_{\mathrm{arch}}=\min(B_{\mathrm{max\ blocks/SM}},\ \lfloor W_{\max}/W_{\mathrm{block}}\rfloor)$，分别对应每 SM 最大 block 数与最大 warp 数折算。
- $R_{\mathrm{SM}}$：每 SM 的寄存器总数（课堂模型取 65,536）。
- $R_{\mathrm{block}}$：单个 block 占用的寄存器数。
- $S_{\mathrm{SM}}$、$S_{\mathrm{block}}$：每 SM 可用 shared memory 与单 block 的 shared memory 用量。
- $W_{\max}$：每 SM 最大驻留 warp 数（课堂模型取 64）。
- $W_{\mathrm{block}}$：每个 block 的 warp 数。

#### 课堂算例：寄存器成为瓶颈

课堂例子中，每个 block 有 128 个线程，每线程使用 160 个寄存器；若每个 SM 有 65,536 个寄存器、最多驻留 64 个 warp，则：

$$
R_{\mathrm{block}}=128\times160=20{,}480,
$$

$$
B_{\mathrm{active}}
=
\left\lfloor\frac{65{,}536}{20{,}480}\right\rfloor
=3,
$$

$$
N_{\mathrm{active\ warps}}
=3\times\frac{128}{32}=12,
\qquad
\mathrm{occupancy}=\frac{12}{64}=18.75\%.
$$

- $R_{\mathrm{block}}$：单个 block 占用的寄存器数。
- $B_{\mathrm{active}}$：寄存器容量允许同时驻留的 block 数。
- 128：每 block 的线程数。
- 160：每线程使用的寄存器数。
- 32：每个 warp 的线程数。

![寄存器压力限制 occupancy 的课堂计算](assets/occupancy-register-pressure-example.jpg)

视频原帧，字幕对应区间：`00:12:53--00:14:22`。

> [!NOTE]
> 官方源码附近有一句说明写成“64 threads”，但视频画面中的代码和计算都使用 128 threads；本讲义采用与实际计算一致的 128。

我们把每一步验算清楚。$128\times160=20{,}480$ 个寄存器是一个 block 的"租金"；$65{,}536/20{,}480=3.2$，向下取整得 3——第 4 个 block 需要 $4\times20{,}480=81{,}920>65{,}536$，放不下。3 个 block 共 $3\times128=384$ 个线程，即 $384/32=12$ 个 warp。与上限 64 warp 相比，occupancy 为 $12/64=18.75\%$。注意取整带来的浪费：SM 上还剩 $65{,}536-3\times20{,}480=4{,}096$ 个寄存器闲置，却不足以再容纳一个 block。

#### 变体演算：寄存器用量减半会怎样

假设编译器通过优化把每线程寄存器降到 80 个，其余条件不变：

$$
R'_{\mathrm{block}}=128\times80=10{,}240,
\qquad
B'_{\mathrm{reg}}=\left\lfloor\frac{65{,}536}{10{,}240}\right\rfloor=6,
$$

$$
N'_{\mathrm{active\ warps}}=6\times4=24,
\qquad
\mathrm{occupancy}'=\frac{24}{64}=37.5\%.
$$

- $R'_{\mathrm{block}}$：寄存器减半后单 block 的寄存器占用。
- $B'_{\mathrm{reg}}$：此时寄存器允许的驻留 block 数。

寄存器减半使驻留 warp 数翻倍。这解释了为什么编译器 flags（如限制寄存器数的 `maxrregcount`）有时能显著提升带宽受限 kernel 的性能：更多的驻留 warp 意味着更强的延迟隐藏能力。

#### 变体演算：shared memory 成为瓶颈

再假设每个 block 申请 48 KB shared memory，而 SM 可供 kernel 使用的 shared memory 为 100 KB（为演示而设的整数），寄存器侧允许 6 个 block：

$$
B_{\mathrm{smem}}=\left\lfloor\frac{100}{48}\right\rfloor=2,
\qquad
B_{\mathrm{active}}=\min(6,\,2)=2,
$$

- $B_{\mathrm{smem}}$：shared memory 预算允许的驻留 block 数。

此时尽管寄存器绰绰有余，驻留 block 数仍被 shared memory 卡到 2，occupancy 仅 $2\times4/64=12.5\%$。这条路径解释了为什么盲目增大 tile（从而增大 shared memory 需求）有时会适得其反——它可能把 occupancy 压低到延迟无法隐藏的程度。第 8 章讨论 tile 尺寸选择时会回到这个权衡。

高 occupancy 的价值在于 **latency hiding**：当一个 warp 等待内存时，调度器可以切换到另一个已就绪的 warp。不过 occupancy 不是越高越好。为了追求满 occupancy 而减少寄存器，可能导致 spilling，反而把数据溢出到更慢的内存。Spill 的读写走 local memory（物理上在 HBM），其代价往往超过 occupancy 提升带来的收益。因此正确的提法是：在不引起 spill 的前提下，保持足够隐藏延迟的 occupancy；具体"足够"是多少，取决于 kernel 的访存与计算比例，通常需要实测。

### 2.5 Coalescing 与 bank conflict 不是同一件事

![连续合并访问与 shared-memory bank conflict 对比](assets/coalescing-bank-conflict.svg)

讲义重绘，综合课堂 `00:14:24--00:18:15` 对两类地址问题的讲解。

- **Memory coalescing** 主要描述相邻线程访问 global memory 时，地址能否合并成少量连续事务。
- **Bank conflict** 主要描述 shared memory 被划分为多个 bank 后，同一 warp 的访问是否挤到同一个 bank。

两者经常被混为一谈，是因为它们都关心"线程到下标、下标到地址"的映射；但它们发生在不同存储层级，诊断方法与对策也不同。Coalescing 看的是**一个 warp 的一次访存指令产生几个内存事务**；bank conflict 看的是**一个 warp 的一次 shared memory 访问需要几个周期串行服务**。

#### Coalescing 的定量分析

Global memory 的请求以固定粒度的事务（课堂简化模型取 128 byte 一段）为单位服务。设 warp 内线程 $t$ 访问 float32 元素，元素下标为：

$$
i(t)=i_0+s\cdot t,\qquad t=0,1,\dots,31,
$$

- $i(t)$：线程 $t$ 访问的元素下标。
- $i_0$：warp 内首个线程访问的下标。
- $s$：相邻线程之间的下标步长（stride）。

对应的 byte 地址为 $a(t)=4\,i(t)$，整个 warp 触及的地址跨度为：

$$
\mathrm{span}=4\bigl(i(31)-i(0)\bigr)+4=4(31s+1)\ \text{bytes}.
$$

- $\mathrm{span}$：warp 一次访存指令触及的连续地址区间长度。
- 4：float32 的字节数。
- 31：warp 内首尾线程的下标差系数。

所需的 128-byte 事务数近似为：

$$
T_{\mathrm{txn}}\approx\left\lceil\frac{\mathrm{span}}{128}\right\rceil.
$$

- $T_{\mathrm{txn}}$：该次访存实际产生的内存事务数。
- 128：课堂模型中一个事务服务的字节数。

代入三种典型 stride 验算：

| stride $s$ | 跨度（bytes） | 事务数 | 每事务有效字节 | 带宽利用率 |
|---:|---:|---:|---:|---:|
| 1 | $4\times32=128$ | 1 | 128 | 100% |
| 2 | $4\times63=252$ | 2 | 64 | 50% |
| 8 | $4\times249=996$ | 8 | 16 | 12.5% |
| 32 | $4\times993=3972$ | 32 | 4 | 3.1% |

结论一目了然：**stride 为 1 时，32 个线程的请求恰好合并成一次 128 字节事务**；stride 每扩大一倍，有效带宽大致减半；stride 大到 32 时，每个线程独占一个事务，带宽利用率跌到 $1/32$。这就是为什么"按行连续遍历"与"按列跳着遍历"在数学上访问同一批元素，实测却可能相差一个数量级。

连续的 global-memory 地址通常更容易 coalesce；shared-memory 地址即使连续，也仍要结合 bank 映射分析。两者都和“线程到地址的映射”有关，但发生在不同存储层级。

#### Bank conflict 的映射公式推导

Shared memory 的硬件组织方式与 global memory 不同：它被划分为若干并行的 bank，每个 bank 一个周期可以服务一个请求；若同一 warp 的多个线程在同一周期命中**同一个 bank 的不同地址**，这些请求必须串行，冲突几度就慢几倍。

课堂使用一个简化的 32-bank、每 bank 宽 4 byte 模型。对 4-byte 元素，bank 可近似写成：

$$
\operatorname{bank}(a)=\left(\frac{a}{4}\right)\bmod 32.
$$

- $a$：相对 shared-memory 起点的 byte 地址。
- 4：每个 bank 对应的 byte 宽度。
- 32：课堂模型中的 bank 数。
- $\operatorname{bank}(a)$：该地址映射到的 bank 编号。

现在推导 stride 访问下的冲突度数。设线程 $t$ 访问元素下标 $i(t)=s\cdot t$（即 byte 地址 $4st$），其 bank 编号为：

$$
b(t)=(s\,t)\bmod 32.
$$

- $b(t)$：线程 $t$ 命中的 bank 编号。
- $s$：以元素为单位的访问步长。

不同的 bank 编号共有多少个？由模运算的基本性质，序列 $\{s\,t\bmod 32\}_{t=0}^{31}$ 的周期为 $32/\gcd(s,32)$，因此：

$$
N_{\mathrm{banks}}=\frac{32}{\gcd(s,32)},
\qquad
C_{\mathrm{conflict}}=\frac{32}{N_{\mathrm{banks}}}=\gcd(s,32),
$$

- $N_{\mathrm{banks}}$：32 个线程实际触及的不同 bank 数。
- $C_{\mathrm{conflict}}$：冲突度数，即平均每个被触及的 bank 要串行服务多少个请求。
- $\gcd(s,32)$：步长与 bank 数的最大公约数。

代入数值验算：

| stride $s$ | $\gcd(s,32)$ | 触及 bank 数 | 冲突度数 | 直观描述 |
|---:|---:|---:|---:|---|
| 1 | 1 | 32 | 1 | 每线程一个 bank，无冲突 |
| 3 | 1 | 32 | 1 | 奇数 stride 必遍历全部 bank |
| 2 | 2 | 16 | 2 | 只用偶数 bank，2 路冲突 |
| 4 | 4 | 8 | 4 | 4 路冲突 |
| 32 | 32 | 1 | 32 | 全部挤在同一 bank，32 路冲突 |

这个公式解释了两个常见经验法则：其一，**奇数 stride 永远无冲突**，因为 $\gcd(s,32)=1$；其二，**2 的幂次 stride 最危险**，冲突度数恰好等于 stride 本身（当 $s$ 为 2 的幂时 $\gcd(s,32)=s$，对 $s\le32$ 成立）。

若一个 warp 的 32 个线程访问同一行中连续的 32 个 `float32`，总跨度恰为：

$$
32\ \text{threads}\times4\ \text{bytes}=128\ \text{bytes}.
$$

- 32：warp 中的线程数。
- 4 bytes：一个 `float32` 元素的大小。
- 128 bytes：课堂示例中可形成一次完整合并访问的连续跨度。

反之，若线程按大 stride 读取一列，地址会落到许多分散的 global-memory transaction；若 shared-memory 的 stride 又恰好让多个地址映射到同一 bank，则会发生 bank conflict。Swizzling 的目标就是重新排列 shared-memory 地址映射，降低这种冲突。一个典型的 swizzle 手法是把逻辑下标 $(r,c)$ 映射为物理列 $c\oplus (r\bmod P)$（$P$ 取 2 的幂）：原本同列（stride 等于行宽，通常是 2 的幂，必然冲突）的访问被打散到不同 bank。推导与上式完全相同，只是把 $s$ 换成了逐行变化的等效步长，使 $\gcd$ 恒为 1。

> [!NOTE]
> “32 banks、每 bank 4 bytes”是本讲为建立直觉使用的模型，不应外推为所有架构、所有数据宽度和所有广播模式下不变的定律。

### 2.6 Wave quantization：尾部的一小批 block 也要单独跑一轮

前四节讨论的是"单个 SM 内部"的约束；本节把视角拉到整卡：当一次 launch 的 block 数超过整卡一次能容纳的数量时，执行会分轮进行，而最后一轮往往装不满。

假设每个 SM 同时驻留 1 个 block（驻留多个时同理，把"wave 容量"换成 SM 数乘以每 SM 驻留 block 数即可）。定义：

$$
W_{\mathrm{waves}}=\left\lceil\frac{B_{\mathrm{grid}}}{S\times r}\right\rceil,
\qquad
\eta=\frac{B_{\mathrm{grid}}}{W_{\mathrm{waves}}\times S\times r},
$$

- $W_{\mathrm{waves}}$：执行全部 block 所需的调度轮数（wave 数）。
- $B_{\mathrm{grid}}$：本次 launch 的 block 总数。
- $S$：GPU 的 SM 数。
- $r$：每个 SM 同时驻留的 block 数（由 2.4 节的资源约束决定）。
- $\eta$：wave 利用率，即实际 block 数占满 $W_{\mathrm{waves}}$ 整轮容量的比例。

若每个 block 运行时间相近，总时间近似正比于 $W_{\mathrm{waves}}$ 而非 $B_{\mathrm{grid}}$——这就是 wave quantization：**延迟随 block 数呈阶梯状跳变，跳变点出现在 $S\times r$ 的整数倍处**。

假设 B200 有 148 个 SM，而一次 launch 有 160 个 block。第一轮每个 SM 分到一个 block 后，还剩 12 个 block；这 12 个 block 仍需要第二轮调度。若每个 block 运行时间相近，尾部这 12 个 block 会使总时间接近两轮，而不是 $160/148$ 轮。

![148 个 SM 执行 160 个 block 时的第二个尾部 wave](assets/wave-quantization-148-plus-12-blocks.jpg)

视频原帧，字幕对应区间：`00:18:15--00:19:18`。

代入公式验算（取 $r=1$）：

$$
W_{\mathrm{waves}}=\left\lceil\frac{160}{148}\right\rceil=2,
\qquad
\eta=\frac{160}{2\times148}\approx 54.1\%.
$$

- 160：本例的 block 总数。
- 148：B200 的 SM 数。
- 2：所需 wave 数。
- $54.1\%$：第二轮只有 12 个 SM 工作、其余 136 个空转所拉低的整体利用率。

也就是说，多发射的 12 个 block（仅比 148 多 $8.1\%$ 的工作量）让总时间几乎翻倍。围绕这个跳变点取几个邻近值，可以更清楚地看到阶梯：

| block 数 | wave 数 | 利用率 $\eta$ | 相对总时间（设单 wave 为 1） |
|---:|---:|---:|---:|
| 148 | 1 | 100.0% | 1 |
| 149 | 2 | 50.3% | 2 |
| 160 | 2 | 54.1% | 2 |
| 296 | 2 | 100.0% | 2 |
| 297 | 3 | 66.9% | 3 |
| 444 | 3 | 100.0% | 3 |

注意 149 与 296 的对比：前者工作量不到后者的一半，耗时却相同；而 296 与 297 只差一个 block，耗时却差出一个完整的 wave。这叫 wave quantization。它解释了为什么看似很小的 grid-size 变化，也可能产生阶梯状的延迟变化。

工程上有两个直接推论。第一，当 kernel 的每个 block 耗时相近时，应尽量让 $B_{\mathrm{grid}}$ 接近 $S\times r$ 的整数倍，或者干脆远大于它（此时尾部占比被摊薄）；最差的点恰好是"略超过整数倍"。第二，反过来，当实测时间随问题规模出现无法由 FLOPs 解释的阶梯时，应怀疑撞上了 wave 边界——这是 profiler 之前就能做的零成本诊断。对 elementwise 这类 block 极小的 kernel，更常见的做法是让 block 数远多于 wave 容量，使量化误差占比可以忽略。

### 本章小结

- GPU 存储层级的容量、带宽和延迟差异很大，HBM 流量常是关键成本。
- Warp divergence、寄存器压力、occupancy、coalescing、bank conflict 分别描述不同问题，不能混为一谈。
- Occupancy 的目标是隐藏延迟，不是盲目追求 100%。
- Grid 尺寸跨过一个 wave 边界时，延迟可能出现非线性跳变。

## 3. Benchmark 与 Profiler：先测量，再优化

第 2 章建立的是解释能力；本章建立的是证据能力。没有可靠的测量，第 2 章的概念只能用来事后讲故事；有了可靠的测量，它们才能变成可检验的假说。本章分两步：先讨论"怎样才算一次可信的计时"，再讨论"计时结果异常时，profiler 能把总时间拆成什么"。

### 3.1 一个可靠 benchmark 应回答什么

Benchmark 至少要固定四件事：

1. 输入形状、dtype 与布局；
2. 使用的 GPU 与软件栈；
3. warm-up 和重复次数；
4. 计时边界中是否包含编译、同步和数据传输。

这四件事各有具体的陷阱，值得逐一展开。

**第一，形状、dtype 与布局决定一切。** 第 2 章的所有结论都以具体形状为前提：同一个 softmax kernel，在 $N=1024$ 时可能一行恰好放下、occupancy 良好，在 $N=1025$ 时可能因 padding 到 2048 而浪费近一半算力。dtype 决定每个元素的字节数，直接改变带宽需求；布局（行优先/列优先、是否连续）决定 stride，从而决定 coalescing。因此"某 kernel 耗时 X 毫秒"这句话若不带形状与 dtype，几乎没有信息量。

**第二，GPU 与软件栈是不可省略的上下文。** 同一份代码在 A100 与 B200 上的绝对时间可以相差数倍，相对排序甚至可能不同（例如某 kernel 在旧卡上受带宽限制、在新卡上变成受 occupancy 限制）。软件栈方面，PyTorch 版本、CUDA 版本、Triton 版本都会影响生成的代码与库函数选择。

**第三，warm-up 与重复次数抵抗两类噪声。** GPU 计时的第一次调用几乎总是异常的，原因至少有三：

- **编译与 autotune**：Triton kernel 首次遇到某个 shape 与 `constexpr` 组合时要现编译，PyTorch 的 `torch.compile` 首次调用要跑完整的图捕获与代码生成，这些开销可达毫秒到秒级，若计入计时将彻底淹没稳态性能。
- **缓存冷态**：第一次运行时输入数据不在 L2，指令缓存未预热，测得的是"冷启动"时间；稳态服务中数据往往有部分驻留，两者可能相差明显。
- **时钟与功耗状态**：GPU 的 boost clock 需要时间与负载才会升上去，空闲后首测常常处于低频状态。

因此标准做法是：先跑若干次 warm-up（不计时），再重复计时若干次取统计量。重复次数的意义在于压制残余波动——单次测量的方差来自时钟抖动、其他进程、DRAM 刷新等不可控因素；取中位数（而非均值）可以进一步抵抗偶发尖峰。课堂使用 `triton.testing.do_bench`，它内部封装了 warm-up、重复测量、同步与统计汇总，还在重复之间主动清空 L2 缓存，使每次重复面对一致的缓存状态，从而让结果更接近"冷缓存"下的可复现值。

**第四，计时边界必须与问题匹配。** GPU 调用通常是异步的：CPU 把 kernel 推入队列后立即返回，真正的执行在后台进行。如果 CPU 只测到“发射 kernel”的时间，而没有在正确位置同步，就可能严重低估真实执行时间。正确的计时结构是：

$$
t_{\mathrm{kernel}}=t_{\mathrm{after\ sync}}-t_{\mathrm{before\ launch}},
$$

- $t_{\mathrm{kernel}}$：kernel 的真实执行时间。
- $t_{\mathrm{before\ launch}}$：发射前在 CPU 侧记录的时间戳。
- $t_{\mathrm{after\ sync}}$：发射后调用设备同步、确认 GPU 完成全部排队工作之后记录的时间戳。

少了中间的同步，括号里只剩下 CPU 排队的时间，对微秒级 kernel 而言这可能低估一个数量级以上。同样要明确计时是否包含 host 与 device 之间的数据拷贝：若研究对象是 kernel 本身，应把数据预先放GPU 上并排除拷贝；若研究对象是端到端服务，则拷贝是真实成本的一部分。两种口径都合法，但不可混用。

### 3.2 矩阵乘法为什么小尺寸时看不出三次增长

方阵乘法的算术工作量随边长 $n$ 近似按 $O(n^3)$ 增长：

$$
\mathrm{FLOPs}\approx 2n^3.
$$

- $n$：方阵边长。
- $2n^3$：每个输出元素约包含 $n$ 次乘法和 $n$ 次加法后的主导运算量。

推导只需一步：输出有 $n^2$ 个元素，每个元素是长度为 $n$ 的内积，含 $n$ 次乘法与 $n-1\approx n$ 次加法，合计约 $2n$ 次浮点运算，相乘即得 $2n^3$。

但课堂在 B200 上测得，小矩阵从 256 增至 1024 时，运行时间几乎不变：

![不同方阵边长的矩阵乘法课堂 benchmark](assets/matmul-scaling-benchmark-results.jpg)

视频原帧，字幕对应区间：`00:25:35--00:26:25`。

| $n$ | 课堂测得时间（ms） |
|---:|---:|
| 256 | 0.6149 |
| 512 | 0.5928 |
| 1024 | 0.5909 |
| 2048 | 0.7010 |
| 4096 | 2.5596 |
| 8192 | 17.6036 |

我们用一个简单的加性模型解释这张表。设实测时间由固定开销与规模相关项叠加而成：

$$
T(n)\approx T_{\mathrm{fixed}}+\frac{2n^3}{P_{\mathrm{eff}}},
$$

- $T(n)$：边长为 $n$ 时的实测总时间。
- $T_{\mathrm{fixed}}$：与规模近似无关的固定开销，包括 kernel launch、参数传递、框架调度、波次未填满的空转等。
- $P_{\mathrm{eff}}$：该规模下实际达到的有效算力（FLOP/s）。

逐项对照表格验算。从 $n=256$ 到 $n=1024$，算术量增长 $(1024/256)^3=64$ 倍，时间却从 0.6149 ms 微降到 0.5909 ms——说明此区间内 $2n^3/P_{\mathrm{eff}}\ll T_{\mathrm{fixed}}$，曲线被固定项压平。粗略估计固定项量级：$n=256$ 时 $2n^3\approx3.4\times10^7$ FLOP，即使有效算力只有 10 TFLOP/s，算术项也只有约 $3\ \mu s$，相比 0.6 ms 的总时间可忽略；故 $T_{\mathrm{fixed}}\approx0.6$ ms。再看另一端：从 $n=4096$ 到 $n=8192$，算术量增长 8 倍，时间从 2.5596 ms 增至 17.6036 ms，比值约 6.88——已接近 8，说明此区间算术项主导，$T_{\mathrm{fixed}}$ 占比降到可忽略。$n=2048$ 恰好处在过渡带：$2n^3\approx1.7\times10^{10}$ FLOP，算术项与固定项同量级，时间开始缓慢抬头。

除了固定开销，还有一个更微妙的效应在压平小尺寸曲线：**缓存**。三个 $n\times n$ 的 float32 矩阵总字节数为 $3\times4n^2=12n^2$ byte。$n=1024$ 时仅约 12.6 MB，远小于 B200 的 96–126 MB L2，全部输入输出可以驻留 L2，HBM 几乎不参与；$n=8192$ 时约 805 MB，必须反复穿越 HBM。因此小尺寸阶段不仅算术少，而且数据移动也被 L2 吸收，双重理由让时间对 $n$ 不敏感。这再次印证 2.1 节的准则：谈性能必须先定位数据住在哪一层。

小尺寸阶段，launch、调度、框架路径和 GPU 未充分占用等固定成本占主导；到 4096、8192 后，实际算术量才显著主导时间。因此，“复杂度是 $O(n^3)$”与“小尺寸实测近似水平线”并不矛盾——渐近复杂度描述的是 $n\to\infty$ 时主导项的行为，而小尺寸时主导项根本不是算术。

> [!WARNING]
> 这张表是课堂当时特定 B200 和软件环境的观测，不是跨 GPU、跨版本可直接复用的性能承诺。

### 3.3 Profiler 负责把总时间拆开

Benchmark 告诉我们“慢”，profiler 则帮助回答“慢在哪里”。对 GPU 工作负载，重点观察：

- 发射了多少个 kernel；
- 每个 kernel 的持续时间；
- kernel 之间是否有同步或空洞；
- 内存拷贝与计算是否重叠；
- 编译后的 kernel 名称是否暴露了 fusion；
- 输入形状变化后，热点是否改变。

这六条对应第 2 章的六类病因，可以建立一张对照表：kernel 数量异常多，提示框架把本可融合的算子拆开了（对应第 4 章的 GeLU 案例）；单个 kernel 时间远超其 FLOPs 推算值，提示带宽受限或 occupancy 不足；kernel 之间存在空洞，提示 CPU 侧发射跟不上或有不必要的同步；拷贝与计算串行，提示数据流水线没有重叠；kernel 名称中出现 `fused`，是编译器做过图级融合的直接证据；形状变化后热点迁移，则提示瓶颈随规模在带宽、算力、固定开销之间切换。

Profiler 的名字有时很长，但能透露关键信息。例如 `fused_add_mul_tanh` 表示编译器可能已经把多个 elementwise 运算融合进同一个 kernel。读 profiler 输出的正确姿势，是把 kernel 清单与自己对计算图的理解对照：数学上有几个算子，时间线上就该有几个（或更少，若已融合）kernel；数量对不上，就是第一个该追问的地方。

### 本章小结

- Benchmark 应固定形状、dtype、硬件、warm-up、重复次数和同步边界。
- 渐近复杂度描述规模足够大时的主导趋势；小尺寸可能被固定成本控制。
- Benchmark 给出结果，profiler 给出结构；二者需要一起使用。
- 性能数字必须带上下文，不能脱离 GPU 和软件版本传播。

## 4. GeLU 性能侦探：Fusion 到底省了什么

本章是全讲的第一个完整案例，我们将严格走一遍第 1 章的闭环：先测三个数学等价实现的性能（基准），再用 profiler 看它们的 kernel 结构（证据），用第 2 章的存储层级语言解释差距（假说），最后讨论结论的边界（检验）。这个案例的结论是本讲最重要的一条定量直觉：**对带宽受限的逐元素算子，决定速度的是 HBM 往返次数，而不是算术。**

### 4.1 三个数学等价的实现

课堂使用常见的 tanh 近似 GeLU：

$$
\operatorname{GeLU}(x)
=
\frac{x}{2}
\left[
1+
\tanh\!\left(
\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)
\right)
\right].
$$

- $x$：输入张量中的一个元素。
- $\operatorname{GeLU}(x)$：该元素的近似 GeLU 输出。
- $0.044715$：tanh 近似式中的经验常数。
- $\sqrt{2/\pi}$：对内部多项式进行缩放的常数。

这个近似式的结构值得先看清：内层是一个三次多项式 $x+0.044715x^3$，经 $\sqrt{2/\pi}$ 缩放后送入 $\tanh$，外层再与 $x$ 组合成门控形式。整个计算对每个元素独立，没有任何跨元素依赖——这意味着它天然可以并行，也意味着它的性能上限完全由数据移动决定：每个元素读一次、写一次就是理论最优，任何额外的中间读写都是纯浪费。

课堂比较三条路径：

1. 用多个 PyTorch 运算逐项写出公式；
2. 调用 PyTorch 内置 `gelu(..., approximate="tanh")`；
3. 用 `torch.compile` 编译逐项版本。

路径 1 与路径 3 的**源代码完全相同**，唯一区别是是否经过编译器；因此它们之间的差距可以干净地归因于"框架逐算子执行"与"编译器融合执行"的差别，而不掺杂算法或实现质量的混杂因素。路径 2 则代表手工调优库函数的上限参照。

![三个 GeLU 实现的数值一致性与运行时间](assets/gelu-three-implementations-results.jpg)

视频原帧，字幕对应区间：`00:31:47--00:32:19`。

在一个 $16384\times16384$ 张量上，课堂运行得到：

| 实现 | 时间（ms） |
|---|---:|
| 逐项 PyTorch | 3.7583 |
| PyTorch 内置 GeLU | 0.6670 |
| `torch.compile` 后的逐项版本 | 0.9388 |

三者结果一致，但性能明显不同。逐项版本比内置版本慢约 5.6 倍，比编译版本慢约 4 倍。这个例子最重要的观察不是“某语言快”，而是 **实现产生了多少次全局内存往返和多少次 kernel launch**。

我们可以用带宽模型做一个粗糙但有启发性的核算。$16384^2\approx2.68\times10^8$ 个 float32 元素，读一遍加写一遍共约 $2\times2.68\times10^8\times4\approx2.15$ GB。按 B200 约 8 TB/s 的 HBM 带宽，理想单遍 kernel 的纯传输时间约 $0.27$ ms；内置 GeLU 实测 0.667 ms，与这一下界同量级（差距来自 launch 开销、带宽利用率不足 100% 等），说明它接近带宽极限。逐项版本若产生 5 次中间往返（见下节），传输量约为 $6\sim7$ 倍，对应 $1.6\sim1.9$ ms 的传输下界，再叠加十余次 launch 的固定开销，与实测 3.7583 ms 在数量级上吻合。数字不需要精确，关键是**量级对得上**：带宽模型足以解释差距，不需要引入任何神秘因素。

### 4.2 未融合版本为何昂贵

逐项表达式包含乘法、加法、tanh 等多个运算。若框架为每个运算分别发射 kernel，则中间张量要反复经历：

$$
\text{HBM 读入}
\rightarrow
\text{片上计算}
\rightarrow
\text{HBM 写回}.
$$

下一步又把刚写回的中间量读回来。数学操作不变，数据移动却成倍增加。

把逐项 GeLU 按 eager 语义展开，大致需要以下 kernel（每行一个算子，读写的"单位"都是整个 $16384\times16384$ 张量，记为 $E$ 个元素）：

| 步骤 | 运算 | 读 | 写 |
|---|---|---|---|
| 1 | $t_1=x^3$（或 $x\cdot x\cdot x$ 两步） | $E$ | $E$ |
| 2 | $t_2=0.044715\,t_1$ | $E$ | $E$ |
| 3 | $t_3=x+t_2$ | $2E$ | $E$ |
| 4 | $t_4=\sqrt{2/\pi}\,t_3$ | $E$ | $E$ |
| 5 | $t_5=\tanh(t_4)$ | $E$ | $E$ |
| 6 | $t_6=1+t_5$ | $E$ | $E$ |
| 7 | $y=0.5\,x\,t_6$ | $2E$ | $E$ |

合计读约 $9E$、写约 $7E$，共 $16E$ 次元素级 HBM 访问，外加 7 次 kernel launch 的固定开销。而理论最优是读 $E$、写 $E$，共 $2E$。两者的流量比为 8，与实测 5.6 倍的时间比在同一量级（时间比小于流量比，是因为小 kernel 的 launch 开销摊薄了差距，且部分中间张量可能命中 L2）。这张表就是"未融合为何昂贵"的全部答案：**每一行中间张量的写与读都是纯粹的浪费，因为下一步马上要用，而它本可以留在寄存器里。**

![未融合与融合 GeLU 的数据移动对比](assets/gelu-fusion-memory.svg)

讲义重绘，依据 profiler 与 fusion 总结区间 `00:32:33--00:35:31`。

Profiler 也给出直接证据。未融合版本出现许多独立 kernel：

![未融合 GeLU 的多 kernel profiler 结果](assets/gelu-naive-multi-kernel-profile.jpg)

视频原帧，字幕对应区间：`00:32:33--00:33:54`。

`torch.compile` 后，profiler 中主要出现一个融合 Triton kernel：

![编译后 GeLU 的单个融合 Triton kernel](assets/gelu-compiled-triton-profile.jpg)

视频原帧，字幕对应区间：`00:34:21--00:35:03`。

融合后，每个元素可以只从 HBM 读取一次，在寄存器中完成多步计算，再写回一次。它同时减少了中间张量流量和 launch 开销。注意编译版本（0.9388 ms）仍比单遍传输下界（约 0.27 ms）慢，这部分差距提醒我们：融合消除的是**中间**流量，输入输出本身的两遍流量与 launch 开销依然存在，它们构成任何逐元素算子不可逾越的下限。

### 4.3 “Triton 为什么更快？”这个问题要先校正

> [!QUOTE]
> 学生把结果概括为“Triton 更快”；讲者随即指出，在这个例子里编译出的 Triton kernel 实际慢于 PyTorch 内置 GeLU，性能还会依赖硬件。对应讨论：`00:36:09--00:36:51`。

这段问答给出了一条重要的方法论：

- 可以说融合版本比未融合逐项版本快；
- 不能据此说 Triton 总比高质量库函数快；
- 内置 kernel 可能经过更深入的手工调优，也可能使用不同近似或特殊指令；
- 只有在同一输入、同一硬件、同一精度要求下的实测才有意义。

把这条方法论说得更形式化一点：性能比较是一个关于元组（实现，形状，dtype，硬件，软件栈）的函数，任何只固定其中一部分变量就下的全称结论都是不可靠的。本例中，`torch.compile` 生成的 Triton kernel 慢于内置 GeLU，可能的原因包括内置版本使用了更快的 tanh 近似路径、更合适的向量化宽度，或更贴合该硬件的 launch 配置；这些都不是 Triton 语言本身的缺陷，而是"自动生成的通用代码"与"人工针对打磨的库函数"之间的正常差距。第 5 章之后我们手写 Triton 时，应当把目标定为**接近并在特定场景超过库函数**，而不是默认必然更快。

### 本章小结

- 数学等价的实现可以产生完全不同的 kernel 图和内存流量。
- Fusion 的主要收益是保留中间值、减少 HBM 往返和 launch 次数。
- Profiler 中的多个小 kernel 与单个 fused kernel 是可验证证据。
- 本次实测中内置 GeLU 最快，因此不能把“Triton”当作自动性能保证。

## 5. Triton 的抽象：一个 program 处理一个 tile

前两章建立了"为什么"：为什么要融合、为什么要关注数据移动。从本章开始解决"怎么做"：用 Triton 亲手写出融合 kernel。本章先讲清 Triton 的编程抽象与它和 CUDA 层级之间的映射关系，再完成第一个完整 kernel（GeLU），最后下到 PTX 层验证编译器到底生成了什么。

### 5.1 从 CUDA thread 转向 Triton program

Triton 仍然生成 GPU kernel，但编程抽象与手写 CUDA 不同。程序员通常不逐线程描述每条操作，而是描述：**一个 Triton program instance 处理哪一块数据，以及这一块上的向量化操作是什么。**

可以把层级粗略理解为：

$$
\text{grid of Triton programs}
\rightarrow
\text{one program per tile}
\rightarrow
\text{compiler maps work to warps/threads}.
$$

这个抽象的价值在于把第 2 章中最容易出错的两件事交给了编译器：tile 内部的数据如何分配到 warp 与线程、如何生成合并访存与向量指令。程序员保留的控制权是同样重要的另外两件事：tile 怎么切（决定数据复用与 grid 尺寸）、tile 内算什么（决定片上完成的融合范围）。

为了让映射关系不虚浮，我们给出一个对应表，并在 5.4 节用 PTX 证据验证它：

| Triton 层概念 | CUDA/硬件层对应物 | 说明 |
|---|---|---|
| grid（launch 时指定） | grid of thread blocks | 一一对应 |
| 一个 program instance | 一个 CUDA block（CTA） | 编译器把一个 program 映射到一个 block |
| `BLOCK_SIZE` 个 tile 元素 | 分布到该 block 的若干 warp、每 warp 32 线程 | 元素数通常大于线程数，差额由 thread coarsening 吸收 |
| `tl.load` / `tl.store` | 合并的 `ld.global` / `st.global` | 编译器负责生成连续、对齐的访问 |
| `tl.max`、`tl.sum` 等归约 | warp shuffle + shared memory 组合 | 程序员不必手写归约树 |

`tl.program_id(axis=0)` 取得当前 program 的一维编号，`tl.arange(0, BLOCK_SIZE)` 构造 tile 内的一组逻辑位置。这里的 `BLOCK_SIZE` 是 **tile 中的元素数**，不是 CUDA block 的线程数。这个区别是初学者最容易混淆的一点，我们反复强调：`BLOCK_SIZE=1024` 的含义是"本 program 负责 1024 个元素"，至于这 1024 个元素由多少线程、每个线程处理几个，是编译器的自由。

### 5.2 Host wrapper 决定发射多少个 program

若输入有 $N$ 个元素，每个 program 处理 $B$ 个元素，需要的 program 数为：

$$
P=\left\lceil\frac{N}{B}\right\rceil.
$$

- $N$：张量元素总数。
- $B$：每个 Triton program 处理的 tile 大小。
- $P$：grid 中 program instance 的数量。

以课堂源码中的 $N=8192$、$B=1024$ 为例，$P=8$。字幕口头转写有一次写成 8000，但官方源码和 $8\times1024$ 的计算都表明实际是 8192。

向上取整的除法引入了一个必须处理的问题：当 $N$ 不是 $B$ 的整数倍时，最后一个 tile 不满。例如 $N=9000$、$B=1024$ 时 $P=9$，最后一个 program 只有 $9000-8\times1024=808$ 个有效元素，其余 $1024-808=216$ 个位置越界。Triton 的标准对策是**始终按满 tile 生成逻辑下标，再用 mask 屏蔽无效位置**——这让所有 program 执行形状一致的代码（对编译器友好），同时保证边界正确。这条"固定形状加 mask"的模式将贯穿本讲全部四个 kernel。

Host 侧伪代码如下：

```python
def triton_gelu(x: torch.Tensor):
    y = torch.empty_like(x)
    n = x.numel()
    block_size = 1024
    grid = (triton.cdiv(n, block_size),)
    gelu_kernel[grid](x, y, n, BLOCK_SIZE=block_size)
    return y
```

Host wrapper 负责分配输出、计算 grid、传递指针和编译期参数；真正逐 tile 执行的逻辑位于 `@triton.jit` kernel 中。几个细节值得点出：`triton.cdiv` 即 $\lceil N/B\rceil$ 的整除实现；`BLOCK_SIZE` 作为 `tl.constexpr` 传入，意味着它在编译期固化，不同的 `BLOCK_SIZE` 会触发不同的编译产物——这与第 3 章"首次调用包含编译开销"的提醒直接相关。

### 5.3 Kernel 的 load—compute—store

```python
@triton.jit
def gelu_kernel(x_ptr, y_ptr, num_elements,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp_2a = tl.exp(2 * a)
    tanh_a = (exp_2a - 1) / (exp_2a + 1)
    y = 0.5 * x * (1 + tanh_a)

    tl.store(y_ptr + offsets, y, mask=mask)
```

![Triton GeLU 中的 program id、mask、load、compute 与 store](assets/triton-gelu-load-compute-store.jpg)

视频原帧，字幕对应区间：`00:45:13--00:46:09`。

这段 kernel 包含五个关键步骤：

1. `pid` 决定当前 program 负责第几个 tile；
2. `offsets` 生成这个 tile 的全局元素下标；
3. `mask` 防止最后一个不满 tile 越界；
4. `tl.load` 把有效元素读入片上表示，并完成向量化计算；
5. `tl.store` 只把有效位置写回。

我们逐行推演其地址逻辑。设 `pid=3`、`BLOCK_SIZE=1024`，则 `start=3072`，`offsets` 为向量 $[3072,3073,\dots,4095]$；`x_ptr + offsets` 是对指针做逐元素偏移，得到 1024 个目标地址的向量；`tl.load` 以这 1024 个地址发出访存，并只保留 `mask` 为真的位置。由于 offsets 连续，这次 load 正是 2.5 节分析的 stride-1 理想模式，编译器可以把它组织成完全合并的事务——程序员没有写任何"对齐"或"向量化"的代码，却得到了 coalescing 的收益，这正是 tile 抽象的兑现。

计算部分的每一行都是形状为 `(BLOCK_SIZE,)` 的向量运算，全部发生在寄存器中：从 `tl.load` 返回到 `tl.store` 之前，数据不离开片上。对照第 4 章的逐项实现，这里把 7 个 kernel 的工作压进了一段 load 与一段 store 之间——**fusion 在手写 Triton 中不是一个编译器特性，而是默认写法**。

最后一个 tile 即使只有少量有效元素，也常被补齐到固定 `BLOCK_SIZE`。mask 是正确性的一部分，不是可有可无的性能装饰。

课堂源码通过指数函数重写 tanh：

$$
\tanh(a)=\frac{e^{2a}-1}{e^{2a}+1}.
$$

- $a$：GeLU tanh 近似内部的缩放多项式值。
- $e^{2a}$：Triton `tl.exp` 计算的指数项。

验证一下这个恒等式：$\tanh(a)=\dfrac{e^{a}-e^{-a}}{e^{a}+e^{-a}}$，分子分母同乘 $e^{a}$ 得 $\dfrac{e^{2a}-1}{e^{2a}+1}$，与代码一致。这不是说所有实现都必须如此，而是因为示例所用 Triton API 中没有直接调用对应 tanh 的路径。代码中的常数 `0.79788456` 即 $\sqrt{2/\pi}\approx0.7978845608$，与 4.1 节公式中的缩放常数对应。

### 5.4 从 Triton 向下看 PTX：thread coarsening

Triton 编译后会落到更低层的中间表示和目标代码。PTX 中常见：

- `%ctaid.x`：block id；
- `%tid.x`：thread id；
- `%r*`、`%f*`：整数和浮点寄存器；
- `ld.global`、`st.global`：global-memory 读写。

![PTX 中重复的寄存器乘加，展示 thread coarsening](assets/ptx-thread-coarsening.jpg)

视频原帧，字幕对应区间：`00:52:42--00:53:18`。

课堂观察到，一个底层线程连续处理了 8 个元素。这是 **thread coarsening**：让单个线程做更多工作，减少调度开销并提高指令级并行。不过这个“8”是当前编译结果，不是 Triton 的固定语义；换形状、编译器或 GPU 都可能改变。

这个 8 可以用一个简单关系复现。设 tile 有 $B$ 个元素，编译器为该 program 配置 $W$ 个 warp，则每个 CUDA 线程分到的元素数为：

$$
c=\frac{B}{32\,W}.
$$

- $c$：thread coarsening 因子，即单线程处理的元素数。
- $B$：tile 元素数（本例 $B=1024$）。
- $W$：编译器为该 kernel 选择的 warp 数。
- 32：每 warp 线程数。

代入 $B=1024$、$c=8$，反解得 $W=1024/(32\times8)=4$，即编译器为这个 program 配置了 4 个 warp、共 128 个 CUDA 线程。于是映射链条完全闭合：**一个 Triton program 对应一个 128 线程的 CUDA block，每个线程以 stride-128 的方式处理 8 个元素**，8 组重复的乘加指令正是 PTX 截图中看到的模式。

为什么要让线程做 8 份而不是 1 份工作？收益至少有四项，且都可以定量理解：

1. **摊薄地址与循环开销**。计算一次基地址后，8 个元素的地址由常量偏移推出，地址算术成本降为 $1/8$。
2. **摊薄调度与发射成本**。128 个线程而非 1024 个线程完成同样的工作，warp 数与调度切换次数都降为 $1/8$。
3. **暴露指令级并行（ILP）**。同一线程的 8 个元素彼此独立，编译器可以交错发射它们的乘加，填满流水线气泡；若每线程只有 1 个元素，隐藏延迟就完全依赖 warp 间切换。
4. **增大每次访存的批量**。单线程连续读 8 个连续（或等距）元素，便于编译器组合成更宽的向量访存指令。

代价同样明确：每线程寄存器用量大致随 $c$ 增长，而 2.4 节已经证明寄存器是 occupancy 的首要约束。$c$ 选得过大，驻留 warp 数下降，warp 级延迟隐藏能力受损。因此 coarsening 因子的选择是 ILP 与 TLP（线程级并行）之间的权衡，交给编译器按目标架构自动决定，正是 Triton 抽象的另一层价值。

> [!WARNING]
> Triton program、CUDA block、warp、thread、tile 是相关但不同的概念。尤其不能把 `BLOCK_SIZE=1024` 直接解释成“发射 1024 个 CUDA 线程”。

### 本章小结

- Triton 的核心抽象是“一个 program 处理一个 tile”。
- Host wrapper 决定 grid，kernel 用 `program_id`、offset 和 mask 描述 tile 内计算。
- 固定 tile 加 mask 既方便编译器生成规则代码，也保证尾部正确性。
- PTX 可用于验证编译器如何映射工作，但观察到的具体 coarsening 因子不是语言保证。

## 6. 单个 program 内归约：Fused Softmax

GeLU 展示了逐元素融合；本章升级一个维度，处理**带归约**的算子。Softmax 是大模型中出现频率最高的归约算子之一，也是展示"融合如何改变 HBM 流量结构"的最佳样本：它的逐算子实现会产生多个全尺寸中间张量，而融合实现可以做到一遍扫描。

### 6.1 稳定 softmax 的数学形式

对矩阵 $X\in\mathbb{R}^{M\times N}$，逐行稳定 softmax 为：

$$
m_i=\max_{0\le j<N}X_{ij},
$$

$$
Y_{ij}
=
\frac{\exp(X_{ij}-m_i)}
{\sum_{k=0}^{N-1}\exp(X_{ik}-m_i)}.
$$

- $X$：输入矩阵。
- $Y$：softmax 输出矩阵。
- $M$：行数。
- $N$：每行元素数。
- $m_i$：第 $i$ 行最大值，用于数值稳定。
- $i$：行下标。
- $j,k$：列下标。

先证明减去行最大值不改变结果。对任意常数 $c$：

$$
\frac{\exp(X_{ij}-c)}{\sum_k\exp(X_{ik}-c)}
=
\frac{e^{-c}\exp(X_{ij})}{e^{-c}\sum_k\exp(X_{ik})}
=
\frac{\exp(X_{ij})}{\sum_k\exp(X_{ik})}.
$$

- $c$：对整行统一减去的任意常数。

分子分母的 $e^{-c}$ 恰好约去，故按行减任意常数在数学上是恒等变换。数值上却不是：取 $c=m_i$ 后，指数的输入最大为 0，$\exp$ 的输出最大为 1，彻底避免了上溢；同时分母至少包含 $\exp(m_i-m_i)=1$，也避免了下溢导致的全零分母。这就是"稳定"二字的全部含义——**数学等价、数值更安全**。

### 6.2 为什么逐算子 softmax 会反复访问 HBM

若 `max`、减法、`exp`、`sum`、除法都各自成为独立 kernel，中间矩阵需要多次读写 HBM。我们把逐算子路径完全展开，逐个 kernel 数清读写（单位均为元素次数，$E=MN$ 为全矩阵元素数，行标量 $O(M)$ 项单独标注）：

| 步骤 | kernel | 读 | 写 | 中间产物 |
|---|---|---|---|---|
| 1 | 逐行 max | $MN$ | $M$ | 行最大值向量 |
| 2 | 减 max（广播） | $MN+M$ | $MN$ | 平移后的全矩阵 |
| 3 | exp | $MN$ | $MN$ | 指数后的全矩阵 |
| 4 | 逐行 sum | $MN$ | $M$ | 行和向量 |
| 5 | 除（广播） | $MN+M$ | $MN$ | 输出矩阵 |

忽略低阶的逐行标量，课堂用主导项估算：

$$
\text{naive reads}\approx 5MN,
\qquad
\text{naive writes}\approx 3MN.
$$

- $MN$：矩阵元素总数。
- $5MN$：未融合路径的主导读取元素数（上表五个 kernel 各读一遍全矩阵）。
- $3MN$：未融合路径的主导写入元素数（第 2、3、5 步各写出一个全尺寸中间张量或输出）。

注意这里出现了一个比 GeLU 更恶劣的结构：**全尺寸中间张量有三个**（平移后、指数后、输出），其中前两个纯属临时产物，却各自完整地在 HBM 上写一遍、读一遍。这正是"归约类算子未融合"的典型代价——归约把数据压成标量，而逐算子实现却先把中间结果膨胀回全尺寸再压。

若一整行能放进一个 Triton program 的片上工作集，就可以读一次、在片上完成两次归约和逐元素变换、再写一次：

$$
\text{fused traffic}\approx MN\ \text{reads}+MN\ \text{writes}.
$$

- $MN$：所有输入或输出元素的数量。
- $MN\ \text{reads}$：融合 kernel 对输入的主导读取量。
- $MN\ \text{writes}$：融合 kernel 对最终输出的主导写入量。

因此按主导流量得到理想比值：

$$
\frac{5MN+3MN}{MN+MN}=4.
$$

- $5MN+3MN$：未融合路径的主导总读写量。
- $MN+MN$：融合路径的主导总读写量。
- 4：只由流量模型得到的理想比值。

这个 4 是流量模型的理想比值，不是“必然四倍加速”。真实时间还受归约实现、occupancy、寄存器压力、带宽利用率和 launch 固定成本影响。例如行宽很小时，每个 program 的工作量过小，launch 与调度开销占比上升，实测加速比会明显低于 4；行宽很大时，单 program 的寄存器与 shared memory 压力上升，可能反过来压低 occupancy。流量模型给出的是"融合值得做"的方向性结论，具体收益仍须按第 3 章的方法实测。

#### 数值验算：代入具体形状

取注意力中常见的形状 $M=4096$、$N=4096$、fp32，把上表换算成字节并估算时间下界。全矩阵 $MN\approx1.68\times10^7$ 个元素，约 67 MB。逐算子路径的主导流量为 $(5+3)MN\approx5.4\times10^8$ 字节（约 537 MB，含两个全尺寸中间张量的完整读写）；融合路径为 $2MN\approx1.3\times10^8$ 字节（约 134 MB）。按 3 TB/s 有效带宽估算：

$$
T_{\mathrm{naive}}\gtrsim\frac{5.4\times10^8}{3\times10^{12}}\approx180\ \mu s,
\qquad
T_{\mathrm{fused}}\gtrsim\frac{1.3\times10^8}{3\times10^{12}}\approx45\ \mu s.
$$

- $T_{\mathrm{naive}}$：逐算子路径的带宽时间下界。
- $T_{\mathrm{fused}}$：融合路径的带宽时间下界。

比值约 4，与流量模型的预言一致。再看缓存的修正作用：67 MB 的中间张量在现代 GPU 的 L2（96–126 MB 量级）边缘，逐算子路径的部分中间读写可能命中 L2 而不走到 HBM，这会使实测加速比**低于** 4；反之若 batch 使矩阵再大几倍，中间张量彻底溢出 L2，实测加速比会逼近甚至个别情形下超过朴素估计（因为未融合路径还多付了 5 次 launch 与同步的开销）。这再次说明流量模型是一阶工具：它锁定主项，缓存效应与固定成本是二阶修正，方向可判、大小需测。

> [!NOTE]
> 若逐项精确追踪低阶的行最大值/行和标量读入，读取数会带额外 $O(M)$ 项。官方源码注释与严格逐项计数在一个低阶项上并不完全一致，因此这里保留课堂真正想表达的主导项 $5MN$。

### 6.3 一行一个 program

![一个 Triton program 在片上完成整行 softmax（官方图）](assets/triton-softmax.png)

课堂对应区间：`01:00:42--01:03:45`。

实现思路如下：

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row * x_row_stride + offsets,
                mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(y_ptr + row * y_row_stride + offsets, y, mask=mask)
```

#### 归约在片上是怎样实现的：从 log 深度到实现成本

`tl.max` 与 `tl.sum` 看起来只是两行代码，理解其实现成本有助于解释 6.2 节末尾"流量比值 4 不是必然四倍加速"的告诫。归约把 $B$ 个值压成 1 个，任何实现都至少需要 $B-1$ 次二元运算；区别在于这些运算如何组织。

并行归约采用树形结构：第 0 层 $B$ 个值两两配对归约得 $B/2$ 个，第 1 层再配对得 $B/4$ 个，依此类推，深度为：

$$
D=\log_2 B,
$$

- $D$：归约树的层数。
- $B$：待归约的元素数（本 kernel 中即 `BLOCK_SIZE`）。

在 GPU 上，树的上半部分（相邻 32 个 lane 以内）用 warp shuffle 指令完成——同一 warp 内直接交换寄存器，不经过任何内存；剩余跨 warp 的层必须经由 shared memory 中转并配合 block 级同步。以 $B=2048$、每 program 配 4 个 warp（128 线程）为例：每线程先串行归约自己分到的 $2048/128=16$ 个元素（thread coarsening，与 5.4 节同一机制），得到 128 个部分值；warp 内 shuffle 归约 5 层（$2^5=32$）得到 4 个 warp 级结果；最后经 shared memory 做 2 层跨 warp 归约得标量。总深度 $\log_2 2048=11$ 层，但其中只有最后 2 层需要 shared memory 与同步。

这条实现链解释了两个现象。其一，归约不是免费的：$B$ 越大，shuffle 与同步的层数越多，这部分开销与 HBM 流量无关，会稀释流量模型预言的加速比。其二，`BLOCK_SIZE` 取 2 的幂不只是"方便"：非 2 的幂会让树形归约出现不齐整的尾部层级，编译器生成的代码明显更差。这也回答了 6.3 节"取不小于 $N$ 的方便尺寸"一句中"方便"二字的精确含义。

与 GeLU kernel 相比，新出现的结构有三处，逐一说明。

**第一，并行轴从元素变成了行。** `tl.program_id(0)` 取到的是行号，grid 为 $(M,)$，共 $M$ 个 program。行首地址由 `row * x_row_stride` 给出——stride 作为运行时参数传入，使 kernel 不假设输入一定行优先连续（转置、切片后的视图也能正确处理）。

**第二，归约发生在 tile 内部。** `tl.max(x, axis=0)` 与 `tl.sum(numerator, axis=0)` 把形状为 `(BLOCK_SIZE,)` 的向量压成标量，编译器负责生成 warp shuffle 与 shared memory 组合的归约树。两次归约的结果都留在片上，直接参与后续向量运算——这正是 6.2 节流量模型的兑现：整行读一次（`tl.load`）、写一次（`tl.store`），中间没有任何全尺寸张量落地。

**第三，padding 的哨兵值服务于归约语义。** `BLOCK_SIZE` 通常取不小于 $N$ 的方便尺寸（例如大于等于 $N$ 的最小 2 的幂，便于编译器生成规整的归约代码），无效位置用 `other=-float("inf")` 填充。padding 位置填 $-\infty$，原因是：

$$
\max(x,-\infty)=x,
\qquad
e^{-\infty}=0.
$$

- $x$：任意有效输入元素或有效行最大值。
- $-\infty$：无效 padding 位置的哨兵值。

逐条验证两条恒等式在整个计算链上的作用：max 阶段，padding 不会篡夺最大值；减 max 后，padding 位置仍为 $-\infty$；exp 后变为 $0$，于是 sum 阶段分母不受影响；最后 `mask` 保证这些位置不写出。于是 padding 既不会改变行最大值，也不会增加分母。这套"哨兵值取归约单位元"的论证模式具有一般性，第 7 章将再次使用（sum 的单位元是 0），读者应把它当作设计任何 masked 归约时的标准检查清单。

用一个小数值例子把整条链走完。设 $N=5$、`BLOCK_SIZE=8`，某行有效值为 $[2,\ -1,\ 3,\ 0,\ 1]$，padding 三位置为 $-\infty$：

$$
m=\max([2,-1,3,0,1,-\infty,-\infty,-\infty])=3,
$$

$$
x-m=[-1,-4,0,-3,-2,-\infty,-\infty,-\infty],
$$

$$
e^{x-m}=[e^{-1},e^{-4},1,e^{-3},e^{-2},0,0,0],
$$

$$
d=e^{-1}+e^{-4}+1+e^{-3}+e^{-2}\approx0.368+0.018+1+0.050+0.135=1.571,
$$

$$
y\approx[0.234,\ 0.012,\ 0.636,\ 0.032,\ 0.086].
$$

- $m$：行最大值，本例为 3。
- $d$：指数和，即分母。
- $y$：该行 softmax 输出。

验算：$y$ 的五个有效分量之和为 $0.234+0.012+0.636+0.032+0.086=1.000$，归一化成立；最大值元素 $3$ 对应最大输出 $0.636$，单调性成立；三个 padding 位置全程未改变 $m$ 与 $d$，也不会被 `mask` 写出。所有高维实现——无论 tile 多大、是否跨 tile——都应在这样的小例子上先验证一遍再上规模。

### 本章小结

- 稳定 softmax 需要先减去每行最大值。
- 当一整行能由一个 program 处理时，最大值、指数和求和都可在片上完成。
- 未融合和融合实现的核心差异是 HBM 流量，而不是数学公式。
- padding softmax 时使用 $-\infty$，可同时保持 max 与 sum 的正确性。

## 7. 一行装不下：跨 tile Row Sum

第 6 章的 softmax 依赖一个前提：整行能放进一个 tile。这个前提在行宽几百到几千时成立，但注意力分数、词表 logits 等场景的行宽可达数万甚至更大，单 tile 放不下。本章回答由此产生的问题：**当一行必须切成多个 tile 时，归约如何在单个 program 内跨 tile 完成。** 这是结构上的第三次升级：从"一个 program 处理一个 tile"变成"一个 program 顺序处理多个 tile"。

### 7.1 为什么需要在 program 内循环

上一章假设一整行能放进一个 tile。若行宽超过合适的 tile 大小，就需要把一行切成多个 tile，由同一个 program 依次读取并累积。

为什么不让多个 program 各算一部分、再合并？回忆 2.2 节的结论：不同 block 之间无法廉价协作——block 的执行顺序不受保证，跨 block 交换中间结果必须经由 global memory，并需要额外的同步机制（第二个 kernel，或原子操作）。这意味着"多个 program 分算一行"会把部分和写回 HBM，恰好破坏第 6 章辛苦省下的流量。相比之下，"一个 program 循环多个 tile"让部分和始终留在寄存器里，整行仍然只读一次、只写一次（行和的"写"只有一个标量）。两者的工作量相同，数据移动结构完全不同，这是本章选择的根本理由。

对输入 $X\in\mathbb{R}^{M\times N}$，行和为：

$$
y_i=\sum_{j=0}^{N-1}X_{ij}.
$$

- $X$：输入矩阵。
- $y_i$：第 $i$ 行的标量输出。
- $M$：行数。
- $N$：行宽。
- $i,j$：行、列下标。

![跨多个 tile 计算一行 row sum（官方图）](assets/triton-row-sum.png)

课堂对应区间：`01:06:53--01:08:21`。

图中一行有 10 个元素，tile size 为 4，因此依次处理：

$$
[3,1,4,1],\quad[5,9,2,6],\quad[5,3,0,0].
$$

- 每个方括号：当前循环读取的一个 4 元素 tile。
- 最后两个 0：越界位置使用的 sum 单位元，而不是原始输入。

第三个 tile 的两个无效位置用 0 填充，总和为：

$$
3+1+4+1+5+9+2+6+5+3=39.
$$

- 左侧十个数：这一行的十个有效输入元素。
- 39：跨三个 tile 累积后的行和。

验算：$3+1+4+1=9$，$5+9+2+6=22$，$5+3=8$，$9+22+8=39$，与图一致。

### 7.2 完整形状推演：三行、每行十个元素

把例子扩展为 $M=3$、$N=10$、`BLOCK_SIZE=4`。Host 端用 `row_sum_kernel[(M,)](...)` 发射 kernel，因此 grid 是 `(3,)`：共有 3 个 Triton program，`program_id` 分别为 0、1、2，每个 program 独立处理一整行。

每一行需要遍历的 tile 数为：

$$
T=\left\lceil\frac{N}{B}\right\rceil
=\left\lceil\frac{10}{4}\right\rceil=3,
\qquad
\mathrm{grid}=(M,)=(3,).
$$

- $M$：输入矩阵的行数，本例为 3。
- $N$：每行的有效元素数，本例为 10。
- $B$：`BLOCK_SIZE`，即每个 tile 的逻辑宽度，本例为 4。
- $T$：每个 program 需要循环处理的 tile 数，本例为 3。
- $\mathrm{grid}$：Triton program 的发射网格，本例包含 3 个一维 program。

三个循环迭代对所有行使用相同的列 offset 和 mask：

| tile | `start` | `offsets` | `mask = offsets < 10` | 有效列 |
|---:|---:|---|---|---|
| 0 | 0 | `[0, 1, 2, 3]` | `[T, T, T, T]` | 0–3 |
| 1 | 4 | `[4, 5, 6, 7]` | `[T, T, T, T]` | 4–7 |
| 2 | 8 | `[8, 9, 10, 11]` | `[T, T, F, F]` | 8–9 |

总共发生 $M\times T=3\times3=9$ 次 tile load。第三个 tile 的后两 lane 因 mask 为 false，由 `other=0.0` 提供 sum 的单位元，不会越界访问。

现在把指针与累加器的状态逐步推演完整。设行优先连续布局，第 $i$ 行行首指针为 `row_start = x_ptr + i*10`（以元素为单位）。以第 0 行为例，三次迭代的状态如下：

| 迭代 | 指针表达式 | 实际访问的元素地址 | 载入向量 | 迭代后 `accumulator` |
|---:|---|---|---|---|
| 初始 | — | — | — | `[0, 0, 0, 0]` |
| 0 | `row_start + [0,1,2,3]` | `x_ptr+0 .. x_ptr+3` | `[3, 1, 4, 1]` | `[3, 1, 4, 1]` |
| 1 | `row_start + [4,5,6,7]` | `x_ptr+4 .. x_ptr+7` | `[5, 9, 2, 6]` | `[8, 10, 6, 7]` |
| 2 | `row_start + [8,9,10,11]`，mask 后两位为 F | `x_ptr+8, x_ptr+9`（后两位不访问，由 `other=0.0` 补齐） | `[5, 3, 0, 0]` | `[13, 13, 6, 7]` |

最后 `tl.sum([13,13,6,7])=13+13+6+7=39`，`tl.store(y_ptr+0, 39)`。注意累加器是**逐 lane** 相加的向量：第 $j$ 个 lane 累积的是各 tile 中第 $j$ 列位置上的值，跨 tile 的归约被推迟到最后一次 `tl.sum`，整个循环期间没有任何中间结果离开片上。

例如输入和逐 lane 累积过程可以完整写成：

```text
X[0] = [ 3,  1,  4,  1 | 5, 9, 2, 6 | 5,   3,   -, -]
acc0 = [ 3,  1,  4,  1] + [5, 9, 2, 6] + [5,   3, 0, 0]
     = [13, 13,  6,  7]  -> tl.sum = 39

X[1] = [ 1,  2,  3,  4 | 5, 6, 7, 8 | 9,  10,   -, -]
acc1 = [ 1,  2,  3,  4] + [5, 6, 7, 8] + [9,  10, 0, 0]
     = [15, 18, 10, 12]  -> tl.sum = 55

X[2] = [-1, -2, -3, -4 | 4, 3, 2, 1 | 0.5, 0.5, -, -]
acc2 = [-1, -2, -3, -4] + [4, 3, 2, 1] + [0.5, 0.5, 0, 0]
     = [3.5, 1.5, -1, -3] -> tl.sum = 1
```

逐行验算：第 1 行 $1+2+\cdots+10=55$；第 2 行 $-(1+2+3+4)+(4+3+2+1)+0.5+0.5=-10+10+1=1$，均与逐 lane 累加结果一致。

最终每个 program 只写一个标量：program 0 写 `y[0]=39`，program 1 写 `y[1]=55`，program 2 写 `y[2]=1`。因此输出为 `y = [39, 55, 1]`，shape 是 `(M,) = (3,)`，而不是 `(3, 1)` 或 `(3, 4)`。

> [!IMPORTANT]
> 这里的“一行三个 tile”不意味着 grid 里有 9 个 program。Grid 仍然只有 3 个 program；每个 program 在自己的循环里执行 3 次 tile load。Program 数和 program 内循环次数是两个独立维度。

这个区分值得再强调一次，因为它对应两种截然不同的性能结构：若真的发射 9 个 program 各算一个 tile，就需要把 9 个部分和写回 HBM 再做一次跨 program 归约（多一个 kernel、多一轮读写）；而 3 个 program 各自循环，部分和零落地。**并行的粒度由归约的依赖结构决定，而不是由数据切分的粒度决定**——行与行之间无依赖，按行并行；行内各 tile 有依赖（同属一个和），放在循环里串行。

### 7.3 累加器应跨 tile 保留

```python
@triton.jit
def row_sum_kernel(x_ptr, y_ptr, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    row_start = x_ptr + row * n_cols
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        values = tl.load(row_start + offsets, mask=mask, other=0.0)
        accumulator += values

    result = tl.sum(accumulator, axis=0)
    tl.store(y_ptr + row, result)
```

这里的关键不是“每个 tile 求一个标量后写回”，而是让向量累加器跨循环迭代保留，最后只做一次归约和一次写回。

两个实现细节值得点出。其一，`for start in range(0, n_cols, BLOCK_SIZE)` 是普通的 Python 风格循环，但循环上界 `n_cols` 是运行时值，Triton 会把它编译为设备端的动态循环；循环次数不需要是编译期常量。其二，本例假设行优先连续布局，故行首直接用 `row * n_cols` 计算；若要像第 6 章那样支持任意 stride，应传入 `x_row_stride` 并以 `row * x_row_stride` 替代，两者结构完全相同。

为什么用向量累加器而不是每轮 `tl.sum` 成标量再加？两种写法数学等价，但向量累加器把归约树（warp shuffle + shared memory 的昂贵部分）从 $T$ 次压到 1 次，循环体内只剩逐 lane 加法，便宜得多。这与"融合"的精神一致：**凡是能推迟到片上最后一步做的事，就不要提前做。**

#### 延伸：当行数太少时，按行并行够不够

Row sum 的并行结构是"一行一个 program"，grid 为 $(M,)$。这引出一个必须会回答的问题：若 $M$ 很小（例如 $M=8$），而 GPU 有 148 个 SM，会发生什么？此时至多 8 个 SM 有工作，其余 140 个空转——这是比 2.6 节尾部 wave 更极端的利用不足，因为连第一个 wave 都填不满。

对策是**按列再切一刀**：把一行的 $T$ 个 tile 分给多个 program（例如每 program 负责 $T/S'$ 个 tile，共 $S'$ 个 program 服务一行），每个 program 算出部分和，再由第二个 kernel（或原子加）合并。这正是 7.1 节否定的"多 program 分算一行"方案——注意它并没有被绝对否定，而是被"单行就能填满硬件"这个前提否定。当前提不成立时，跨 program 合并的额外 HBM 流量（每行 $S'$ 个部分和的写与读）就是值得付出的代价。设行宽 $N$、每 program 处理 $N/S'$ 列，则部分和流量为 $2MS'$ 个元素，与全矩阵 $MN$ 相比可忽略的条件是 $S'\ll N$，通常容易满足。

工程含义：并行切分维度的选择，应使 grid 中的 program 数达到硬件并行规模的数倍（填满 wave 并留出调度余量），又不至于细到让合并开销与固定成本反超。这个判据在下一章的 matmul 中将直接决定 grid 是 $(\lceil M/B_M\rceil,\lceil N/B_N\rceil)$ 的二维形状。

> [!IMPORTANT]
> Padding 值取决于归约的单位元：sum 用 0，max 用 $-\infty$。不能把 softmax 的 padding 规则机械复制到所有归约。

### 7.4 Block 与 tile 的语义边界

#### 跨 tile 归约的流量复核

在对比 block/tile/program 三个概念之前，先用流量模型复核一遍本章实现的效率。行和的输入输出为：读入全矩阵 $X$ 共 $MN$ 个元素，写出 $M$ 个标量。理论最优流量为：

$$
\mathrm{traffic}_{\mathrm{opt}}=MN+M,
$$

- $MN$：输入矩阵的全部元素，每个必须且只需读一次。
- $M$：输出的行和标量数。

本章"program 内跨 tile 累加"实现的实际流量恰为 $MN$ 读 $+M$ 写，达到最优。作为对照，"每 tile 写部分和"的实现（设每行 $T$ 个 tile）需要额外写出 $MT$ 个部分和、再由第二遍读回 $MT$ 个求和，总流量为：

$$
\mathrm{traffic}_{\mathrm{partial}}=MN+MT+MT+M=MN+2MT+M,
$$

- $MT$：$M$ 行、每行 $T$ 个部分和。
- 两个 $MT$：部分和的一次写与一次读。

多出的 $2MT$ 正是"把本可留在寄存器的累加器踢回 HBM"的代价。以 $M=4096$、$N=16384$、`BLOCK_SIZE=1024`（$T=16$）为例，$2MT\approx1.3\times10^5$ 个元素，约为 $MN\approx6.7\times10^7$ 的 0.2%——对本例影响虽小，但当 $N$ 变小、$T$ 相对占比上升，或部分和路径迫使第二次 kernel launch 的固定成本计入时，差距会被放大。练习 3 的目的正是让读者在自己的硬件上量出这条曲线的实际形状。

- **Tile** 是我们在算法上切出的数据块。
- **Triton program** 是处理一个或多个 tile 的逻辑实例。
- **CUDA block** 是后端映射到硬件调度的执行单位。

一个 program 可以循环处理多个 tile；一个 tile 也不应被直接等同为固定数量的 CUDA thread。把这些概念分开，才能理解编译器还有优化和映射空间。本章提供了一个最清楚的例证：算法上的一行被切成 3 个 tile（数据切分），逻辑上由 1 个 program 处理（并行结构），物理上映射为 1 个 CUDA block（执行单位）——三个数字各是各的，把它们混为一谈就无法讨论"grid 该多大、循环该多长"这类设计问题。

### 本章小结

- 行宽超过单 tile 容量时，一个 program 可以循环遍历多个 tile。
- 跨 tile 的片上累加器减少中间结果写回。
- Sum 的 padding 单位元是 0；不同归约要选择不同哨兵值。
- Tile 是数据分块，program 是逻辑实例，CUDA block 是后端执行映射。

#### 延伸：跨 tile 的 softmax——两遍扫描与在线归约

本章解决了"一行装不下时的 sum"；自然要问：第 6 章的 softmax 在行宽超限时怎么办？逐行分析三种方案的结构。

**方案一：两遍扫描。** 第一遍循环各 tile，用向量累加器求行最大值 $m_i$ 与指数和（注意指数和依赖 $m_i$，故朴素做法是先求 max、再循环一遍求 $\sum\exp(x-m_i)$）；第二遍重新读取各 tile，计算 $\exp(x-m_i)/\mathrm{denominator}$ 并写出。输入被读两次、输出写一次，流量约 $2MN+MN=3MN$，仍显著优于逐算子的 $8MN$，但不如单 tile 情形的 $2MN$。

**方案二：在线 softmax（online softmax）。** 数学上，max 与指数和可以在**单遍**内联合维护。设已处理前缀的最大值为 $m$、指数和为 $d$，读入新 tile 后先求其最大值 $m'$，则更新规则为：

$$
m_{\mathrm{new}}=\max(m,m'),
\qquad
d_{\mathrm{new}}=d\,e^{m-m_{\mathrm{new}}}+\sum_{j\in\mathrm{tile}}e^{x_j-m_{\mathrm{new}}}.
$$

- $m$：已处理前缀的滚动最大值。
- $d$：以 $m$ 为基准的滚动指数和，即 $\sum e^{x_j-m}$。
- $m'$：新读入 tile 的最大值。
- $m_{\mathrm{new}},d_{\mathrm{new}}$：合并后的滚动最大值与指数和。

验证正确性：合并前 $d=\sum_{\mathrm{prefix}}e^{x_j-m}$，换基准到 $m_{\mathrm{new}}$ 需整体乘以 $e^{m-m_{\mathrm{new}}}$；新 tile 的贡献以新基准直接求和。两项相加即得以 $m_{\mathrm{new}}$ 为基准的全前缀指数和，归纳成立。初始值取 $m=-\infty$、$d=0$（又是单位元）。这样最大值与分母在一遍循环内同时得到，但第二遍仍需重读输入计算输出，故总流量同为 $3MN$；在线归约省下的不是流量，而是第一遍里的一次完整 tile 循环时间（max 与 sum 合并为一趟），以及避免存储中间最大值向量。

**方案三：拆分归约轴。** 按 7.3 节末尾的思路把列分给多个 program 做分段在线归约，再跨 program 合并。FlashAttention 沿这条路线走得更远：它把 softmax 的在线归约与注意力矩阵乘的 K 循环交织在一起，使注意力分数矩阵本身永不落地——这是本讲"融合即流量"原则在大模型中最重要的一次应用，后续讲义会专门展开。

对读者的要求不是背下三种方案，而是能说出它们的**流量结构**（各读几遍、写几遍、中间量落不落地）——流量结构一旦清楚，方案优劣在多数情形下不辩自明。

## 8. 二维数据复用：Tiled Matmul + Fused ReLU

前三个 kernel 的共同点是对每个输入元素只做常数次访问；矩阵乘法打破了这个模式：每个输入元素天然要被许多输出元素使用。本章是本讲结构的顶点——二维 tiling、双操作数复用、沿归约维的循环、边界 mask 的组合，以及收尾时的逐元素融合，全部汇集在一个 kernel 里。

### 8.1 矩阵乘法为什么必须做 tiling

设：

$$
A\in\mathbb{R}^{M\times K},
\qquad
B\in\mathbb{R}^{K\times N},
\qquad
C=AB\in\mathbb{R}^{M\times N},
$$

$$
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj}.
$$

- $A,B$：输入矩阵。
- $C$：输出矩阵。
- $M$：输出行数。
- $N$：输出列数。
- $K$：归约维度。
- $i,j,k$：行、列和归约下标。

若每计算一个 $C_{ij}$ 都从 HBM 独立读取整行 $A_i$ 与整列 $B_j$，输入会被重复搬运。量化一下这种浪费：每个 $A$ 元素 $A_{ik}$ 被第 $i$ 行的全部 $N$ 个输出使用，每个 $B$ 元素 $B_{kj}$ 被第 $j$ 列的全部 $M$ 个输出使用；朴素实现把这份"天然复用"全部丢弃，每个输出重新从 HBM 取数，读取量达到 $2MNK$ 次元素访问。

Tiling 让一个 program 计算 $C$ 的一个二维块，并让载入的 $A$、$B$ 子块被多个输出元素复用。课堂把朴素方案的主导 HBM 访问量写成 $O(MKN)$，而浮点运算量也是 $O(MKN)$，所以算术强度只保持常数量级：

$$
\mathrm{AI}_{\mathrm{naive}}
\approx
\frac{2MKN}{2MKN+MN}
=O(1).
$$

- $2MKN$（分子）：矩阵乘法的主导浮点运算数。
- $2MKN+MN$（分母）：朴素教学模型中输入重复读取与输出写入的主导元素流量。
- $\mathrm{AI}_{\mathrm{naive}}$：朴素实现每搬运一个元素完成的近似运算量。

如果理想化地让 $A$、$B$ 各从 HBM 读取一次，再写出 $C$，流量会变为 $MK+KN+MN$：

$$
\mathrm{AI}_{\mathrm{ideal}}
\approx
\frac{2MKN}{MK+KN+MN}.
$$

- $MK$：矩阵 $A$ 的元素数。
- $KN$：矩阵 $B$ 的元素数。
- $MN$：输出矩阵 $C$ 的元素数。
- $\mathrm{AI}_{\mathrm{ideal}}$：所有输入被充分复用时的理想算术强度。

代入方阵 $M=N=K=n$ 看趋势：朴素模型 $\mathrm{AI}\approx 2n^3/(2n^3+n^2)\approx 1$，与 $n$ 无关；理想模型 $\mathrm{AI}\approx 2n^3/(3n^2)=2n/3$，随 $n$ 线性增长。这正是 3.2 节实测"大尺寸 matmul 时间终于反映 $n^3$"的另一面：只有大尺寸才给了算术强度增长到摆脱带宽束缚的空间。

现实中整个矩阵装不进 shared memory，因此 tiling 是介于“完全不复用”和“全矩阵片上驻留”之间的可实现方案。

![64×64 输出 tile 沿 K 维累加 64×32 和 32×64 子块（官方图）](assets/gemm-tiled.png)

课堂对应区间：`01:16:19--01:18:20`。

图中输出 tile 为 $64\times64$，每次沿 $K$ 维载入 $64\times32$ 的 $A$ tile 与 $32\times64$ 的 $B$ tile。一次 tile 乘法完成约：

$$
2\times64\times64\times32
$$

- 第一个 2：一次乘加按一次乘法与一次加法计两次浮点运算。
- 两个 64：输出 tile 的行数与列数。
- 32：当前 $K$-tile 的宽度。

次浮点运算，但输入元素只需从更慢层级载入后在 tile 内复用。tile 边长增大时，理想的算术强度通常也随之提高，直到被寄存器、shared memory、occupancy 或布局约束限制。

可以把复用账算得更细：载入的 $64\times32$ 个 $A$ 元素与 $32\times64$ 个 $B$ 元素共 $64\times32+32\times64=4096$ 个元素，支撑了 $64\times64\times32=131{,}072$ 次乘加。平均每个载入的元素承担 $131{,}072/4096=32$ 次乘加——复用倍数恰好等于另一个 tile 的对应边长。一般地，$B_M\times B_N$ 输出 tile、$B_K$ 宽的 K-tile 方案下，每载入一个 $A$ 元素被复用 $B_N$ 次、每载入一个 $B$ 元素被复用 $B_M$ 次。tile 越大，复用越多；而 2.4 节与 2.6 节的约束（寄存器、shared memory、wave）决定了 tile 不能无限大。Tiled matmul 的设计空间，本质就是这组复用收益与资源约束的交点。

#### Tiled matmul 算术强度的完整推导

把复用账换算成算术强度。设 grid 覆盖全部输出，共 $\lceil M/B_M\rceil\times\lceil N/B_N\rceil$ 个 program；忽略边界 padding，每个 program 沿 $K$ 循环 $K/B_K$ 次，每次从 HBM 载入 $A$ 的 $B_M\times B_K$ 块与 $B$ 的 $B_K\times B_N$ 块。全部 program 的 $A$ 块合计把 $A$ 读了 $\lceil N/B_N\rceil$ 遍（每一列 tile 的 program 都要读同一段 $A$），$B$ 被读了 $\lceil M/B_M\rceil$ 遍。于是总流量与算术量为：

$$
\mathrm{Bytes}_{\mathrm{HBM}}
\approx
4\left(
MK\cdot\frac{N}{B_N}
+
KN\cdot\frac{M}{B_M}
+
MN
\right),
\qquad
\mathrm{FLOPs}=2MNK,
$$

- 4：float32 的字节数。
- $MK\cdot N/B_N$：$A$ 被每个 $N$ 方向 tile 列重复读取的总元素数。
- $KN\cdot M/B_M$：$B$ 被每个 $M$ 方向 tile 行重复读取的总元素数。
- $MN$：输出 $C$ 写出一次的元素数。

代入方阵 $M=N=K=n$ 并取 $B_M=B_N=b$：

$$
\mathrm{AI}_{\mathrm{tiled}}
\approx
\frac{2n^3}{4\left(2n^3/b+n^2\right)}
=
\frac{b}{8+4b/n}
\approx
\frac{b}{8}\quad(n\gg b),
$$

- $b$：输出 tile 的边长（$B_M=B_N=b$）。
- $\mathrm{AI}_{\mathrm{tiled}}$：tiling 实现的算术强度。

结论干净有力：**tile 方案的算术强度近似与 tile 边长成正比，与问题规模无关**。$b=64$ 时 $\mathrm{AI}\approx8$ FLOP/byte；对照 B200 的算力带宽比（约 $2250\ \mathrm{TFLOP/s}\,/\,8\ \mathrm{TB/s}\approx280$ FLOP/byte 的量级，视精度而定），$b=64$ 仍不足以纯靠 fp32 通用算力吃满峰值——这正是现代 matmul 依赖 tensor core（其 $P$ 更高，但同时 $B_K$、数据布局与流水线也完全不同）以及更大 tile、多级流水的原因。推导的价值不在具体数字，而在它指出唯一的杠杆：**想提高算术强度，就增大 tile 或降低外层重复读取**（例如 L2 友好的 tile 遍历顺序、split-K 的取舍），而这一切又立刻撞上 2.4 节的寄存器与 shared memory 预算。Matmul 优化之所以是一个专门的研究领域，正是因为这几个约束的联立没有解析解，只能在实测中搜索。

#### Tile 尺寸的资源账：以 $64\times64\times32$ 为例

把 2.4 节的 occupancy 计算框架套到本 kernel 上，看看 $64\times64\times32$ 这个 tile 要交多少"租金"。

**累加器的寄存器开销。** `acc` 形状为 $64\times64$，共 4096 个 float32。设编译器为该 program 配置 $W$ 个 warp，则每线程分摊的累加器寄存器数为：

$$
R_{\mathrm{acc}}=\frac{B_M B_N}{32W}.
$$

- $R_{\mathrm{acc}}$：每线程仅用于输出累加器的寄存器数。
- $B_M B_N$：输出 tile 的元素数，本例 4096。
- $W$：编译器配置的 warp 数。
- 32：每 warp 线程数。

取 $W=4$（128 线程），$R_{\mathrm{acc}}=4096/128=32$——仅累加器就占 32 个寄存器，已相当可观；再加上地址、载入缓冲与临时量，每线程寄存器很容易逼近 2.4 节算例中的高位区间。若把 tile 加大到 $128\times128$ 而 $W$ 不变，$R_{\mathrm{acc}}=128$，几乎必然 spill 或被迫增配 warp。这就是"tile 越大寄存器越紧"的定量来源。

**K-tile 缓冲的 shared memory 开销。** 每次循环要把 $64\times32$ 的 $A$ 块与 $32\times64$ 的 $B$ 块 staged 到片上（实际由编译器/流水线管理），fp32 下共 $(2048+2048)\times4=16$ KB；若使用双缓冲（prefetch 下一块与计算当前块重叠），直接翻倍到 32 KB。对照 2.4 节"shared memory 成为瓶颈"的算例，一个 SM 驻留的此类 program 数立刻被压缩到个位数。

**结论。** $64\times64\times32$ 不是随手写下的数字，而是在算术强度（希望 tile 大）与 occupancy（希望 tile 小）之间的一个经典折中点。读者做 8.3 节末尾练习 4 时，应当能把扫描结果中的每一个拐点对应到这三笔账的某一笔上。

### 8.2 用 stride 把二维坐标变成地址

对二维张量，元素地址偏移可写成：

$$
\operatorname{offset}(r,c)=r\,s_r+c\,s_c.
$$

- $r,c$：行、列坐标。
- $s_r$：沿行移动一步的 stride。
- $s_c$：沿列移动一步的 stride。

例如一个形状 $2\times4$ 的行优先连续矩阵，stride 为 $(4,1)$。坐标 $(1,2)$ 的线性偏移为：

$$
1\times4+2\times1=6.
$$

- 1、2：目标元素的行、列坐标。
- 4、1：行 stride 与列 stride。
- 6：相对张量起始地址的线性元素偏移。

显式 stride 让 kernel 不必假设输入总是某一种连续布局。这一点在真实代码中极其重要：转置视图（stride 交换）、切片视图（stride 不变、起点偏移）、广播展开的张量（某维 stride 为 0）都改变 stride，而不改变逻辑形状。Kernel 若把布局写死，遇到这些视图就会产生静默错误或强制框架先做拷贝；把 stride 作为运行时参数传入，则同一份 kernel 对任意布局都正确。

#### Stride 的三个推论：转置、切片与广播

Stride 公式 $\operatorname{offset}(r,c)=r\,s_r+c\,s_c$ 的三个直接推论，覆盖实践中绝大多数"形状对了但结果错了"的地址 bug。

**推论一：转置不搬数据。** 对行优先矩阵 $X$（stride $(s,1)$），`X.T` 只是把 stride 换成 $(1,s)$ 的视图，底层数据一字节不动。若 kernel 把 stride 当参数，`X.T` 可以直接传入且结果正确；若 kernel 假设列连续（$s_c=1$），传入 `X.T` 会把行当成列读，得到静默错误的结果。这解释了 PyTorch 中 `.contiguous()` 存在的理由：某些手写 kernel 只支持连续布局，框架不得不在调用前显式拷贝。

**推论二：切片保持 stride、只动起点。** 取子块 $X[r_0:r_1,\ c_0:c_1]$，stride 不变，数据起点偏移 $r_0 s_r+c_0 s_c$。因此 tiled kernel 处理"子矩阵"与处理"带 stride 参数的原矩阵"是同一件事——8.3 节的指针矩阵公式天然支持从大矩阵中裁出的任意视图，这正是大模型中 KV cache 按段读写可行的原因。

**推论三：stride 为 0 实现广播。** 形状 $(M,N)$ 而 stride 为 $(0,1)$ 的张量，每一行都映射到同一段内存——逻辑上有 $M$ 行，物理上只有一行数据。在 Triton 中把这样的 stride 传入，`tl.load` 会重复读到同一地址；对只读操作这通常正是广播语义（如 bias 按行广播）。但要警惕两点：其一，重复读同一地址虽能命中缓存，仍占用访存发射槽；其二，**绝不能对 stride-0 的维度执行写**——多个 program 会写同一地址，结果未定义。

**推论四（综合）：stride 决定 coalescing 的上限。** 把 2.5 节的地址模式分析接到 stride 公式上：warp 内相邻线程若沿列连续访问（$s_c=1$ 方向），地址连续，可以完全合并；若沿行方向访问（步长 $s_r$），等效 stride 为 $s_r$ 个元素，事务数按 $\lceil 4(31s_r+1)/128\rceil$ 膨胀。因此同一个 kernel，输入从行优先换成列优先（转置视图），访存模式可能从完美合并退化到 32 路分散——形状没变、数学没变、stride 变了，性能天地之别。判断一个 tiled kernel 的访存是否友好，标准动作是把指针矩阵中**变化最快的那个下标**找出来，确认它乘的 stride 是否为 1：8.3 节的 `a_ptrs` 中 `indices_k[None, :]` 沿列方向变化最快，乘的是 `stride_ak`，行优先时恰为 1，合并成立；`b_ptrs` 中变化最快的是 `indices_n[None, :]`，乘 `stride_bn`，行优先 $B$ 同样为 1。两个 load 都通过检查——这不是巧合，而是行优先布局与这个 tile 结构的天然契合。

把三个推论合起来看，stride 机制实际上是"逻辑张量"与"物理存储"之间的自由映射：只要会算 $r s_r+c s_c$，转置、切片、广播、padding 后的大数组视图都统一到同一个公式下。这也是为什么本讲的 kernel 全部把 stride 作为显式参数，而不是在 kernel 内部用 `n_cols` 之类的形状量现算——后者只覆盖行优先连续这一种特例。

### 8.3 Triton 中的二维指针矩阵与 K 循环

下面给出与官方课堂实现同构的完整版本。它包含 host wrapper、grid、全部边界 mask、输出指针和最终 store；只要环境已安装 PyTorch 与 Triton，并传入同一 CUDA 设备上 shape 可相乘、dtype 相同的二维张量，就可以直接调用 `triton_matmul_relu(a, b)`。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 当前 program 负责 C 的第 (pid_m, pid_n) 个输出 tile。
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    indices_k = tl.arange(0, BLOCK_K)

    # 广播后分别得到 [BLOCK_M, BLOCK_K] 和
    # [BLOCK_K, BLOCK_N] 的指针矩阵。
    a_ptrs = (
        a_ptr
        + indices_m[:, None] * stride_am
        + indices_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + indices_k[:, None] * stride_bk
        + indices_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 沿归约维 K 每次推进 BLOCK_K；末尾 K-tile 用 mask 补 0。
    for k in range(0, K, BLOCK_K):
        a_mask = (
            (indices_m[:, None] < M)
            & (indices_k[None, :] + k < K)
        )
        b_mask = (
            (indices_k[:, None] + k < K)
            & (indices_n[None, :] < N)
        )
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Accumulator 仍在片上；写回前顺手融合 ReLU。
    acc = tl.maximum(acc, 0.0)

    c_ptrs = (
        c_ptr
        + indices_m[:, None] * stride_cm
        + indices_n[None, :] * stride_cn
    )
    c_mask = (
        (indices_m[:, None] < M)
        & (indices_n[None, :] < N)
    )
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.ndim == 2 and b.ndim == 2
    assert a.device == b.device and a.dtype == b.dtype

    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c
```

#### Host 侧的 grid 设计：与 wave 量化对账

`grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))` 决定发射多少 program。把 2.6 节的 wave 量化公式套上来：设 tile 为 $64\times64$，每个 program 映射为一个 CUDA block，若每 SM 可驻留 $r$ 个 block，则 wave 容量为 $148r$（以 B200 为例）。对 $M=N=4096$，grid 为 $64\times64=4096$ 个 program——远大于 wave 容量，尾部量化误差占比可忽略，这是大 matmul 的天然优点。但对 $M=N=1024$，grid 仅 $16\times16=256$ 个 program，若 $r=2$ 则 wave 容量为 296：一轮装不满（256/296），大量 SM 空转。此时除了减小 tile 换更多 program（牺牲算术强度），还可以考虑 split-K——把 $K$ 维也切开以扩充 grid，代价是引入跨 program 的部分和合并，其权衡结构与 7.3 节末"行数太少时"的讨论完全同构。**grid 设计的三条轴（$M$、$N$、$K$ 切分）分别服务于复用、占用率与并行度，没有免费的维度。**

![Triton matmul 的二维索引、指针矩阵与 K-tile 循环](assets/triton-matmul-k-tile-loop.jpg)

视频原帧，字幕对应区间：`01:20:53--01:21:59`。

`[:, None]` 与 `[None, :]` 通过广播构造二维地址网格。`a_mask`、`b_mask` 同时处理矩阵外边界和最后一个不满的 $K$-tile；`c_mask` 则保证边缘输出 tile 不越界写入。每次循环载入一对 $K$-tile，`tl.dot` 累加到同一个输出 accumulator，循环结束后才写回。

#### 指针矩阵的完整推导

我们把 `a_ptrs` 的构造逐步展开。`indices_m` 形状为 `(BLOCK_M,)`，`indices_m[:, None]` 形状为 `(BLOCK_M, 1)`；`indices_k[None, :]` 形状为 `(1, BLOCK_K)`。按广播规则相加，得到形状 `(BLOCK_M, BLOCK_K)` 的偏移矩阵，其 $(p, q)$ 元素为：

$$
\operatorname{off}^{A}_{pq}
=
\bigl(\operatorname{pid}_m\,B_M+p\bigr)s_{am}
+
q\,s_{ak},
$$

- $\operatorname{off}^{A}_{pq}$：指针矩阵第 $(p,q)$ 个元素相对 `a_ptr` 的元素偏移。
- $\operatorname{pid}_m$：当前 program 在 $M$ 方向的 tile 编号。
- $B_M$：即 `BLOCK_M`。
- $p,q$：tile 内的行、列局部坐标。
- $s_{am},s_{ak}$：$A$ 的行、列 stride。

对照 8.2 节的公式 $\operatorname{offset}(r,c)=rs_r+cs_c$，这里全局行坐标 $r=\operatorname{pid}_m B_M+p$，全局列坐标（即归约下标）$c=q$，完全一致——指针矩阵不过是把 stride 公式同时应用到 tile 内全部 $B_M\times B_K$ 个坐标上。`b_ptrs` 的推导对称：全局行坐标（归约下标）$k=q$，全局列坐标 $n=\operatorname{pid}_n B_N+p$，偏移为 $q\,s_{bk}+(\operatorname{pid}_n B_N+p)s_{bn}$，形状 `(BLOCK_K, BLOCK_N)`。

循环末尾的两行指针推进值得单独论证。第 $\tau$ 次迭代（$\tau=0,1,\dots$）处理的归约下标区间是 $[\tau B_K,\ \tau B_K+B_K)$；指针在每次迭代后加上 $B_K s_{ak}$（对 $B$ 是 $B_K s_{bk}$），相当于把指针矩阵整体平移到下一个 K-tile。与"每轮重新从 `indices_k + k` 现算地址"相比，指针推进把每轮的地址更新压成一次向量加法——这又是 5.4 节"摊薄地址开销"思路的再现。需要注意 mask 中的 `indices_k[None, :] + k < K` 使用的是循环变量 $k$（逻辑坐标），而指针用累加方式推进（物理地址），两者一个管正确性、一个管位置，不能互相替代。

#### 边界 mask 的正确性论证

三类 mask 各自防守一条边界，我们逐条给出不变式。

**`a_mask`** 的两个条件分别防守 $M$ 边界与 $K$ 边界：$(\operatorname{pid}_m B_M+p)<M$ 保证行不越界（最后一个 $M$-tile 不满时），$q+k<K$ 保证归约下标不越界（最后一个 K-tile 不满时）。两个条件必须同时成立，因为越界发生在任一维都意味着该元素不属于 $A$。

**`b_mask`** 对称：$(q+k)<K$ 防守 $K$ 边界，$(\operatorname{pid}_n B_N+p)<N$ 防守 $N$ 边界。

**`other=0.0` 为何不污染结果**：被 mask 屏蔽的位置以 0 载入，参与 `tl.dot` 时对累加器的贡献为 $0\times(\cdot)=0$。这要求 mask 在 $A$ 与 $B$ 上是**对齐的**：对任意越界的归约下标 $k'$，$A$ 的对应列与 $B$ 的对应行同时被置 0，乘积项恒为 0，$C_{ij}$ 不变。这正是 6.3 节"哨兵值取归约单位元"模式的第三次出现——乘加归约的单位元是加法单位元 0。

**`c_mask`** 防守输出边界：只有 $(\operatorname{pid}_m B_M+p)<M$ 且 $(\operatorname{pid}_n B_N+p')<N$ 的位置才写回。这里即使不加 mask，越界位置算出的值也是"垃圾但确定"的；真正的问题是**越界写入会破坏相邻张量的内存**，属于正确性灾难而非数值误差，因此 `c_mask` 不可省略。

把三条放在一起看，可以提炼出一条通用模式：**load 的 mask 保护读侧（防越界访问）并用单位元保持数值中立，store 的 mask 保护写侧（防破坏他人内存）；归约维的边界在 load 侧处理，输出维的边界在 store 侧处理。** 这套论证以后写任何 tiled kernel 都可直接复用。

#### 一个具体的数值小例

设 $M=N=96$、$K=64$，$B_M=B_N=64$、$B_K=32$，行优先连续（stride 分别为 $(K,1)=(64,1)$、$(N,1)=(96,1)$、$(N,1)=(96,1)$）。grid 为 $(\lceil96/64\rceil,\lceil96/64\rceil)=(2,2)$，共 4 个 program。取 $(\operatorname{pid}_m,\operatorname{pid}_n)=(1,1)$ 的 program：它负责输出行 $64\ldots127$、列 $64\ldots127$，其中行 96 起越界、列 96 起越界。第一次 K 循环（$k=0$）：`a_ptrs` 的 $(0,0)$ 元素偏移为 $64\times64+0=4096$，对应 $A_{64,0}$；`a_mask` 在第 32 行（全局行 96）起为假。第二次迭代指针加 $32\times1=32$，覆盖 $A_{64..95,\ 32..63}$，$K$ 恰好整除，无 K 边界 mask 触发。最终 `c_mask` 使只有左上角 $32\times32$ 的有效区域写回。读者可以把这些数字代回上面的公式逐一验证。

> [!NOTE]
> 这段代码为了完整展示地址与边界逻辑，固定使用 $64\times64\times32$ tile。生产实现通常还会针对 shape 与 GPU autotune `BLOCK_M`、`BLOCK_N`、`BLOCK_K`、warp 数和 pipeline stage 数。

#### 为什么是 `tl.dot`：从向量指令到矩阵指令

循环体中只有一条实质计算语句 `acc += tl.dot(a, b)`，它与前面三个 kernel 的逐元素运算在硬件层面有本质区别。逐元素乘加映射为标量/向量 FMA 指令：每条指令处理若干对元素，完成一次乘加。而 `tl.dot` 声明的是一整块小矩阵乘，编译器可以把它映射到 GPU 的**矩阵乘指令**（NVIDIA 的 tensor core MMA 类指令）：一条指令直接完成一个小形状（例如 $16\times8\times16$ 量级）的矩阵乘累加，把几十个 FMA 的工作压进一次发射。

这对本讲的主线意味着什么？前两章建立的"算术 vs 带宽"框架中，峰值算力 $P$ 并非单一数字：通用 FMA 路径与 tensor core 路径的峰值相差可达一个数量级（同一块 B200，fp32 通用路径与低精度 tensor core 路径的标称峰值完全不同）。于是 matmul 优化的目标函数变得更微妙：不仅要让算术强度超过 $P/W$，还要确保走的是高 $P$ 的那条路径——这要求数据类型、tile 形状、布局都满足矩阵指令的约束（例如维度是 8 或 16 的倍数、特定的 fragment 布局）。`tl.dot` 的价值正在于把这些约束的满足交给编译器：程序员声明"这里是一块矩阵乘"，编译器负责选择 MMA 指令、安排 fragment 与 swizzle（2.5 节）以避免 shared memory 冲突。

这也解释了本讲义示例代码的一个刻意保守之处：它用 fp32 累加器、固定 tile、未 autotune，目的是让地址与边界逻辑完全可读。生产级 matmul（cuBLAS、Triton 模板库）在这副骨架上叠加的是：低精度输入 + fp32 累加、多级流水线（prefetch 与 `tl.dot` 重叠）、swizzled 布局、软件流水化的 K 循环，以及覆盖全部 shape 区间的 autotune 表。骨架不变，工程量在细节。



把 ReLU 放在最终写回之前：

$$
Y_{ij}=\max(C_{ij},0),
$$

- $C_{ij}$：矩阵乘法累加后的输出元素。
- $Y_{ij}$：融合 ReLU 后写回的元素。

这样无需先写出 $C$，再由另一个 kernel 读回并计算 ReLU。它与 GeLU 案例体现的是同一条原则：**中间结果还在片上时，尽可能完成相邻的便宜逐元素操作。** 量化收益：独立 ReLU kernel 需要额外读 $MN$、写 $MN$ 共 $2MN$ 次元素访问；融合后这 $2MN$ 完全消失，对带宽受限阶段（例如小矩阵、大 batch 的逐元素后处理）是实实在在的节省。在 Transformer 中，matmul 之后紧跟的操作（bias、激活、残差加、dropout）几乎都可以按同样方式融合进 epilogue，这正是 cuBLASLt、Triton matmul 模板普遍提供 epilogue fusion 接口的原因。

> [!WARNING]
> 更大的 tile 并不自动更快。它可能提高复用，也可能消耗更多寄存器、降低 occupancy，或导致大量 padding。实际选择通常需要结合架构约束与 autotuning。

把这条警告与本讲前面的定量工具串起来：tile 边长翻倍，复用倍数翻倍（收益），但累加器占用的寄存器按 $B_M\times B_N$ 增长、K-tile 缓冲占 shared memory 按 $B_K(B_M+B_N)$ 增长（成本）；2.4 节告诉我们这会压低驻留 block 数，2.6 节告诉我们 grid 变小还可能撞上 wave 边界，padding 比例 $\approx(B_M/B_N$ 相对 $M/N$ 的余数）则直接浪费算力。autotune 就是在这些此消彼长的项之间做网格搜索——理解了每一项的来源，autotune 的结果才不会显得神秘。

### 本章小结

- Tiled matmul 的收益来自让 $A$、$B$ 子块服务多个输出元素，提高数据复用。
- Stride 把逻辑坐标与实际内存布局连接起来。
- K 维循环把多个局部乘积累加到片上输出 tile，最后统一写回。
- ReLU 可在 accumulator 写回前融合，避免新增一次 HBM 往返。

## 9. 标题中的 XLA：本讲没展开，但它在编译栈中的位置

### 9.1 事实边界

本章不是视频内容复述。完整 01:26:41 视频、人工字幕与官方 `lecture_06.py` 均没有 XLA/JAX 教学段落。之所以补充这一节，是为了回答标题中的 “XLA” 与前述 Triton kernel 处于什么关系。

把这条边界再划得具体一些：第 1 至 8 章的每一个论点、数字与截图都可在视频或官方源码中找到出处；本章的所有内容则来自 OpenXLA 与 JAX 的公开文档，课堂并未讲授。我们仍然把它保留在讲义中，理由有二：其一，标题中的 XLA 是读者检索本讲时最自然的疑问之一，不回答会让"标题与内容不符"显得像疏漏；其二，第 4 章已经出现过 XLA 的影子——`torch.compile` 生成的融合 Triton kernel，其背后正是"框架图 → 编译器 → kernel"这条与 XLA 同构的路径，把编译栈讲清楚有助于理解 Triton 在整个生态中的位置。

### 9.2 从框架图到 GPU kernel

![XLA 编译栈位置示意：StableHLO、HLO、GPU 后端与 PJRT](assets/xla-pipeline.svg)

补充图，非视频内容；对照全片 `00:00:00--01:26:41` 后确认不存在 XLA 讲解。

这张图是讲义补充绘制，不是视频截图。可以把 XLA 的位置理解为：

1. 前端框架把高层计算表示为可移植的 **StableHLO** 操作；
2. XLA 转为内部 HLO，进行代数化简、布局、fusion 等优化；
3. GPU 后端继续 lowering，生成可在设备上执行的 kernel；
4. **PJRT** 负责连接框架、编译器与具体设备运行时。

对四个名词各给一句精确定义，以免它们停留在缩写层面。**StableHLO** 是一套有版本化规范的可移植操作集，地位类似"高级汇编"：前端框架（JAX，以及通过 torch_xla 的 PyTorch）把用户程序翻译成 StableHLO，即可与具体后端解耦。**HLO**（High Level Operations）是 XLA 内部的优化表示，StableHLO 进入 XLA 后转为 HLO，在其上执行代数化简（如 $x+0\to x$）、常量折叠、布局分配与 fusion 决策。**Lowering** 指从高抽象表示逐级降到低抽象表示的过程，本讲其实已经见过一次：第 5 章从 Triton 源码到 PTX 就是一次 lowering，XLA 管线只是把同样的思想放大到整张计算图。**PJRT** 是设备运行时接口层，负责把编译产物加载到具体设备（CPU/GPU/TPU）、管理 buffer 与执行调度，使同一份编译管线可以服务异构硬件。

对照本讲前面的内容，这条流水线的每一站都有对应物。第 1 站对应 PyTorch/JAX 层用户写下的数学表达式；第 2 站的 fusion 正是第 4 章 GeLU 案例干的事，只不过由编译器在图级自动完成，而不是我们手写；第 3 站的 lowering 对应第 5 章从 Triton 到 PTX 的下沉，只是 XLA 不一定经过 Triton；第 4 站则超出本讲范围，属于设备运行时管理。

OpenXLA 官方的 GPU 架构说明指出，XLA:GPU 既有基于 LLVM/PTX 的原生 emitter，也可以通过 TritonIR emitter 生成特定 GPU 计算。因此 Triton 与 XLA 并不位于同一抽象层：

| 维度 | Triton | XLA |
|---|---|---|
| 主要抽象 | 一个 program 如何处理一个 tile | 一张计算图/高层 IR 如何整体优化并 lower |
| 典型使用者 | kernel 作者、编译器生成器 | 框架与编译器系统 |
| 主要控制点 | load/store、tile、mask、layout、归约 | 图级 fusion、布局、后端选择、代码生成 |
| 二者关系 | 可直接写 kernel，也可成为后端生成目标 | 某些 GPU 路径可选择 TritonIR emitter |

最容易犯的错误是把两者看成二选一的“GPU 编程语言”。更准确地说，Triton 是 kernel DSL，而 XLA 是能够在更大范围内分析并生成 kernel 的编译系统；XLA 的某条后端路径可以利用 Triton。

用本讲的语言重新表述这张表：Triton 程序员决定第 5 至 8 章里的每一个 tile、mask 与循环；XLA 的图级优化器决定的是更高一层的问题——哪些算子应该融合成一个 kernel（第 4 章的教训）、每个融合 kernel 该用什么实现（可能调用 cuBLAS、原生 emitter，或生成 Triton）。当我们手写 Triton 时，我们是在做 XLA 后端某一个"emit"环节的工作，并且拥有比自动 emitter 更细的控制权；当我们使用 `torch.compile` 或 JAX 时，我们把这份控制权交给了编译器，换取零手写成本。第 4 章的实测（编译版 GeLU 仍略慢于内置库函数）提醒我们：自动与手写之间没有永恒的高下，只有具体形状与硬件下的实测。

### 9.3 最小 JAX → HLO 实验（课外可选）

> [!NOTE]
> 本小节不是课堂内容，而是理解标题中 XLA 的可选学习目标。它使用 JAX 官方当前的 AOT/lowering API，只演示“Python 函数怎样变成 StableHLO、怎样查看 XLA HLO、怎样继续编译”；不把输出冒充本视频中的演示。

下面选择一个与上一章呼应的 `matmul + ReLU`。`jax.jit(...).lower(...)` 会先按输入 shape 与 dtype 特化计算，再 lower 到 XLA 的编译器输入表示。`lowered.as_text()` 打印 StableHLO；`compiler_ir(dialect="hlo")` 则取得 XLA HLO 对象。

```python
import jax
import jax.numpy as jnp


def linear_relu(x, w):
    return jax.nn.relu(x @ w)


# 实际数组只用于给 lowering 提供具体 shape 与 dtype；CPU 环境也可运行。
x = jnp.ones((4, 8), dtype=jnp.float32)
w = jnp.ones((8, 4), dtype=jnp.float32)

lowered = jax.jit(linear_relu).lower(x, w)

# 1. XLA 的可移植输入表示：通常能看到 stablehlo.dot_general
#    与 stablehlo.maximum 等操作。
print(lowered.as_text())

# 2. 查看 XLA HLO 文本。as_hlo_text() 属于返回的 XlaComputation。
hlo = lowered.compiler_ir(dialect="hlo")
print(hlo.as_hlo_text())

# 3. 继续生成当前设备可执行文件并运行。
compiled = lowered.compile()
y = compiled(x, w)
print(y.shape)  # (4, 4)
```

这个最小实验建议观察三件事：

1. 输入 shape 为 $(4,8)$ 和 $(8,4)$，所以 lowering 已特化为输出 shape $(4,4)$；换 shape 通常会得到另一份特化程序。
2. StableHLO 中通常仍能辨认矩阵乘与 maximum；HLO 文本是更接近 XLA 优化管线的表示。
3. `compile()` 才继续生成当前 CPU/GPU/TPU 的可执行文件；仅看到 HLO 中相邻的 `dot` 与 `maximum`，不能据此断言它们最终一定成为一个 GPU kernel。

第 3 条值得展开，因为它直接呼应第 4 章的方法论。HLO 中相邻的两个操作是否融合，取决于后端的 fusion 决策（算子类型、形状、目标硬件、成本模型），IR 文本本身不是承诺。这与"Triton 不等于更快"是同一条戒律的另一面：**中间表示里看到结构，不等于最终二进制里存在该结构；一切仍以 profiler 中的实际 kernel 清单为准。**

> [!WARNING]
> 旧教程可能使用已经删除的 `jax.xla_computation`。现行 JAX 文档推荐 `jax.jit(f).lower(...).compiler_ir("hlo")`。如果目的是跨进程保存和部署，官方还建议使用 `jax.export`，而不是把这里打印出的内部 IR 当成稳定序列化格式。

完成这个实验后，读者应能把两条路径区分开：手写 Triton 是直接规定 tile 级 kernel；JAX/XLA 则从高层数组程序出发，经 StableHLO/HLO 和后端 lowering 生成设备程序。

#### 对照：PyTorch 侧的编译路径

第 4 章用过的 `torch.compile` 与 JAX/XLA 同构但不同源，把两者并列有助于避免概念混淆：

| 环节 | PyTorch 路径 | JAX/XLA 路径 |
|---|---|---|
| 图捕获 | Dynamo 跟踪 Python 字节码得到 FX 图 | `jax.jit`/`lower` 按 shape 特化函数 |
| 中间表示 | FX Graph / AOTAutograd 产物 | StableHLO → HLO |
| 优化与代码生成 | Inductor（GPU 上主要生成 Triton kernel） | XLA 优化管线 + GPU 后端 emitter |
| 运行时 | CUDA 运行时 + Triton JIT | PJRT |

对应 9.2 节的四站流水线，两者在前端与 IR 名称上不同，在第 2 站（图级 fusion）与第 3 站（kernel 生成）上解决的是**同一组问题**。这也解释了第 4 章 profiler 截图中为什么会出现 Triton kernel：Inductor 的默认 GPU 代码生成器就是 Triton。于是整条知识链闭合——本讲手写的 Triton kernel，正是 `torch.compile` 在背后替我们自动生成的那类产物；学会手写，意味着在自动生成的结果不够好时，我们有能力诊断（用第 2、3 章的工具）并接管（用第 5 至 8 章的工具）。

### 9.4 拓展阅读

#### 一页纸回顾：从 Python 表达式到设备执行的三条路径

至此，本讲出现过的三种执行路径可以并排放在一张对照表里，作为本章的收束：

| 路径 | 谁决定融合 | 谁决定 tile 与映射 | 谁生成设备代码 | 本讲出处 |
|---|---|---|---|---|
| 手写 PyTorch eager | 无人（逐算子） | 框架预置 kernel | 框架预置 kernel | 4.1 路径 1 |
| 手写 Triton | 程序员（kernel 内） | 程序员 | Triton 编译器 → PTX | 第 5–8 章 |
| `torch.compile` / JAX+XLA | 编译器（图级） | 编译器/模板 | Inductor（生成 Triton）或 XLA 后端 | 4.1 路径 3、9.2–9.3 |

读表方式：从左到右，自动化程度递增、可控粒度递减。三行之间不存在全局最优：eager 胜在零心智负担与调试透明；手写 Triton 胜在对流量结构的完全控制；编译路径胜在开发与维护成本。第 4 章的实测数字（3.7583 / 0.6670 / 0.9388 ms）恰好各占一行——这张表不是理论分类，而是被同一组数字钉在现实中的。

- [OpenXLA：XLA:GPU Architecture](https://openxla.org/xla/gpu_architecture)
- [OpenXLA：Terminology](https://openxla.org/xla/terminology)
- [StableHLO Specification](https://openxla.org/stablehlo/spec)
- [JAX：Ahead-of-time lowering and compilation](https://docs.jax.dev/en/latest/aot.html)

> [!NOTE]
> 上述链接用于补足标题背景。它们没有改变本讲视频的事实范围：课堂实际主线在 GPU 性能模型与 Triton kernel。

### 本章小结

- 本次视频实际没有讲 XLA；本章是透明标注的外部补充。
- StableHLO/HLO 承载较高层的计算表示，XLA 负责优化与后端 lowering。
- Triton 面向 tile 级 kernel 表达；XLA 面向更高层编译流程。
- XLA 的 GPU 后端可以使用 TritonIR emitter，因此二者可以上下游协作。

## 总结与延伸

本讲从 GPU 的存储与调度约束出发，形成了一条完整的 kernel 推理链：

1. **先看硬件约束**：HBM 与片上存储的带宽差异、warp 调度、寄存器压力和 occupancy 决定了可行空间。
2. **再看真实证据**：benchmark 判断总延迟，profiler 揭示 kernel 数量与热点。
3. **识别数据移动**：GeLU 与 softmax 表明，很多“计算慢”其实是中间张量反复进出 HBM。
4. **用 tile 表达复用**：Triton program 让我们按 tile 组织 load、compute、store，并用 mask 保持边界正确。
5. **逐步增加结构**：从一维 elementwise，到单 tile 归约、跨 tile 归约，再到二维 tiled matmul。
6. **在写回前融合**：GeLU 的多步逐元素计算、softmax 的多阶段归约、matmul 后的 ReLU，都展示了减少中间写回的收益。

把这条链再压缩一次，可以得到一个更便于随身携带的版本：**先数字节，再数 FLOP；先数 kernel，再数指令；一切结论以实测为准。** 字节数与 kernel 数都是可以口算或从 profiler 直接读出的量，它们解释了本讲遇到的每一个性能谜团——GeLU 的 5.6 倍、softmax 的 4 倍流量比、小矩阵的时间平台、wave 边界的阶梯。相比之下，"这个实现写得底层""那个框架编译得好"这类形容词层面的判断，在本讲的案例中没有一次经受住实测。

#### 复盘：一条完整的诊断链长什么样

为了帮助读者把这些工具串成肌肉记忆，我们把一个虚构但完全典型的诊断过程走一遍。假设某逐元素算子链（归一化 + 激活 + 缩放）实测偏慢：

1. **Benchmark 定边界**：固定形状 $(4096, 4096)$、fp32、同一 GPU，warm-up 后重复 100 次取中位数，得 1.8 ms。
2. **算流量下界**：输入输出各一遍共 $2\times4096^2\times4\approx134$ MB，按 3 TB/s 有效带宽估算下界约 $45\ \mu s$。实测是下界的 40 倍——存在巨大解释空间。
3. **Profiler 看结构**：时间线上出现 6 个独立 elementwise kernel 与 2 次中间张量分配，总 kernel 时间与 1.8 ms 吻合，无异常空洞——瓶颈不在 launch，而在流量。
4. **形成假说**：6 个 kernel 意味着约 $7\sim8$ 倍于最优的 HBM 流量（对照 4.2 节的逐项表格法），与 40 倍的差距同量级，假说成立。
5. **改写**：合并为单个 Triton kernel（或先尝试 `torch.compile`），中间量全部留在寄存器。
6. **复测**：0.06 ms，约为下界的 1.3 倍，剩余差距来自带宽利用率与 launch 固定成本，进入合理区间。诊断闭环完成。

注意这个过程中没有一步依赖"经验直觉"：下界由流量模型给出，瓶颈位置由 profiler 给出，改写方向由假说给出，成败由复测给出。这正是第 1 章闭环在实战中的形态。

四个 Triton kernel 还留下了一份可复用的结构清单，日后面对新算子时可以逐项自查：逐元素部分是否需要独立的 HBM 往返，还是可以并入相邻 kernel；归约能否在单 tile 内完成，若不能，累加器是否跨 tile 保留在片上；二维以上问题中，哪些输入被哪些输出复用，tile 边长是否把复用倍数推到了资源约束允许的极限；边界处 load mask 的哨兵值是否是该归约的单位元，store mask 是否完备；grid 尺寸与硬件 SM 数之间是否存在 wave 量化浪费。这份清单不要求记住任何具体数字，它本身就是本讲的方法论沉淀。

### 课堂结尾问答留下的三条边界

**第一，不要从 PTX 开始写所有东西。** PTX 能帮助检查编译结果，但手写低层代码的开发成本、可移植性与维护负担都很高。课堂提到 ThunderKittens、CuTe 等替代或相邻工具；不同 DSL 各自带有适合某类计算结构的 inductive bias，不存在一个抽象对所有 kernel 都最优。

**第二，高维张量没有单一最优切法。** 该沿哪个维度分 tile，要看依赖关系、数据复用、片上容量、连续访问模式和输出归约方向，而不是只看张量有几维。

**第三，性能结论必须回到实测。** 本讲的 GeLU 结果已经证明：编译融合大幅优于未融合逐项实现，但仍慢于当时的 PyTorch 内置 kernel。正确结论应包含形状、硬件与实现上下文。

三条边界可以统一理解为对"抽象层级"的提醒：过低的层级（PTX）付出不成比例的开发与维护成本；不加分析的层级选择（默认切法、默认 tile）错过问题结构带来的收益；而无论选择哪个层级，层级本身从不保证性能——只有测量能裁决。这与第 1 章的闭环首尾呼应。

### 建议的后续练习

1. 修改 GeLU 的 `BLOCK_SIZE`，记录运行时间与编译后 PTX 的变化。
2. 让 softmax 的列数跨过 2 的幂边界，观察 padding、寄存器压力与性能。
3. 为 row sum 比较“每 tile 写部分和”与“program 内跨 tile 累加”两种实现。
4. 为 matmul 扫描 `BLOCK_M`、`BLOCK_N`、`BLOCK_K`，同时记录吞吐量与 occupancy。
5. 用 profiler 验证 fused ReLU 是否真的没有成为单独 kernel。
6. 阅读 OpenXLA GPU 后端资料，区分图级 fusion 与 kernel 内 fusion 的职责。

这六个练习对应本讲的六条主线，各自检验一种定量工具：练习 1 检验 thread coarsening 与 tile 语义的理解（`BLOCK_SIZE` 变了，PTX 中的 coarsening 因子应随之改变）；练习 2 检验 padding 与寄存器预算的关系；练习 3 把 7.1 节的"program 内循环 vs 多 program 合并"之争变成可测量的对比；练习 4 复现 8.3 节末尾的 tile 权衡，并把 occupancy 计算（2.4 节）用于解释结果；练习 5 训练"以 profiler 为准"的习惯；练习 6 则把视野从单 kernel 抬回编译栈。

每个练习都应当按 3.1 节的标准记录结果：固定形状、dtype、GPU 与软件栈，报告 warm-up 后的重复测量统计量，并对每一个观察到的拐点给出本讲概念层面的解释。若某个现象找不到解释——例如改 `BLOCK_SIZE` 后性能纹丝不动，或 occupancy 上升反而变慢——那正是最值得深究的地方：它意味着模型缺了一项（缓存效应、wave 边界、编译器自动向量化……），而补上这一项的过程，就是本讲方法论真正内化的过程。反之，如果所有结果都"符合预期"，应怀疑实验设计是否覆盖了足够宽的形状区间，而不是庆幸模型完美。

课程在此后会进入更大规模的系统主题，包括跨设备通信与多 GPU 训练。理解本讲的 kernel 性能模型，是理解分布式训练的前提：当单卡计算和数据移动都没有被说清楚时，多卡系统只会把瓶颈放大。届时读者会发现，分布式分析中"通信量 vs 计算量"的权衡，与本讲"HBM 流量 vs 算术量"的权衡在数学结构上完全同构——roofline 的思维方式会原样复现，只是存储层级换成了网络。

#### 本讲定量结论一览

把全讲的定量结果集中一处，便于日后速查。每个结论都标注其适用条件，提醒读者不要脱离上下文引用：

| 结论 | 数值/形式 | 适用条件 |
|---|---|---|
| 带宽与算力的分界 | $\mathrm{AI}=P/W$ | roofline 模型，单层存储近似 |
| occupancy | $\lfloor R_{\mathrm{SM}}/R_{\mathrm{block}}\rfloor$ 等资源最小值 | 寄存器、smem、架构上限三者取小 |
| 课堂 occupancy 算例 | $128\times160\to18.75\%$ | 128 线程/block、160 寄存器/线程 |
| coalescing 事务数 | $\lceil 4(31s+1)/128\rceil$ | fp32、stride $s$、128-byte 事务模型 |
| bank 冲突度数 | $\gcd(s,32)$ | 32 bank、4-byte 宽的简化模型 |
| wave 数与利用率 | $\lceil B/(Sr)\rceil$，$B/(Sr\cdot\lceil\cdot\rceil)$ | block 时长均匀近似 |
| matmul 实测平台 | 256–1024 约 0.6 ms 不变 | 课堂特定 B200 环境 |
| GeLU 融合加速 | 3.7583 → 0.6670/0.9388 ms | $16384^2$ fp32，课堂实测 |
| softmax 流量比 | $(5+3)/(1+1)=4$ | 主导项，忽略 $O(M)$ |
| tiled matmul 算术强度 | $\approx b/8$（方阵、fp32） | $B_M=B_N=b$，$n\gg b$ |

#### 本讲常见误读清单

最后列出课堂与自学中最常出现的六种误读，每一条在前文都已有对应论证，此处集中备查：

1. **"复杂度高所以大矩阵一定慢得成比例"**——小尺寸由固定开销主导，3.2 节的平台期是常态而非异常。
2. **"occupancy 越高越好"**——超过隐藏延迟所需后继续推高 occupancy 会以 spill 为代价，见 2.4 节。
3. **"连续地址就不会慢"**——连续只保证 coalescing 一侧；shared memory 连续访问仍可能 bank conflict，两者层级不同，见 2.5 节。
4. **"Triton 写的就快"**——4.3 节的实测反例：编译生成的 Triton kernel 仍慢于手工调优库函数。
5. **"BLOCK_SIZE 就是线程数"**——它是 tile 元素数，线程数与每线程工作量由编译器决定，见 5.1 与 5.4 节。
6. **"IR 里看到融合就等于生成了融合 kernel"**——中间表示不是承诺，一切以 profiler 中的 kernel 清单为准，见 9.3 节。

### 本章小结

- 高性能 kernel 的核心不是“写得更底层”，而是让数据移动、并行映射和计算结构彼此匹配。
- Fusion、tiling、mask、片上累加与 stride 是本讲可迁移到其他算子的关键工具。
- 任何优化都需要 benchmark 和 profiler 共同验证。
- Triton 提供了可控的 kernel 抽象；XLA 则属于更高层的编译与生成体系。
