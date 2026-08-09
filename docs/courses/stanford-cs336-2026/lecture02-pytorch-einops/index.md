# Stanford CS336 2026 第 2 讲：PyTorch、einops 与训练资源核算

![课程视频封面](assets/cover.jpg)

- 课程：Stanford CS336 — Language Modeling from Scratch
- 讲次：Lecture 2 — PyTorch (einops)
- 频道：Stanford Online
- 时长：01:17:25
- 原视频：[YouTube](https://www.youtube.com/watch?v=kuYAsz7zspQ)
- 讲义依据：英文人工字幕、课程视频画面与官方 <code>lecture_02.py</code>

> [!IMPORTANT]
> 这讲并不是一份 PyTorch API 速查表。它建立的是训练语言模型所需的“资源会计学”：任何对象先还原成张量，再追踪它的 shape、dtype、device、FLOPs、内存流量和生命周期。后续实现 Transformer、优化训练吞吐或排查 OOM，都依赖这套语言。

## 导读：从一行 PyTorch 代码看到整台训练机器

### 这讲要解决什么问题

看到 <code>y = x @ w</code> 时，初学者通常只问“结果是多少”；这讲要求继续追问：

1. <code>x</code>、<code>w</code> 和 <code>y</code> 的每个维度分别代表什么？
2. 每个元素用多少字节，张量存放在 CPU 还是 GPU？
3. 这次运算做了多少 FLOPs，用了多少秒，利用了硬件峰值的几成？
4. 时间到底花在算术单元，还是花在显存搬运？
5. 反向传播、优化器状态与激活又会把计算和显存放大多少？

这五个问题串成了本讲的教学主线：

**张量表示 → dtype 与显存 → einops 与维度语义 → FLOPs/MFU → 算术强度/Roofline → 反向传播 → 优化器与训练循环 → 显存换计算。**

### 初学者需要的最少前置知识

- **标量、向量、矩阵**：分别可看成 rank 0、rank 1、rank 2 的张量。
- **矩阵乘法**：内侧维度相等，并沿该维度做“乘后求和”。
- **导数与链式法则**：反向传播只是把局部导数按计算图从后往前组合。
- **二进制位**：1 byte = 8 bits；dtype 的位数直接决定单元素存储成本。
- **数量级估算**：先忽略小项得出可用上界，再逐项补回通信、激活与系统开销。

> [!NOTE]
> 讲者在开头提到 Marin 的 \(10^{23}\) FLOPs 训练实验按期完成，用它强调：训练规模虽然巨大，却可以被资源模型提前预测。napkin math 不是追求小数点后的“精确”，而是帮助我们在启动昂贵实验前发现数量级错误。

### 本章小结

- 一行张量代码同时隐含 shape、dtype、device、计算量与数据移动。
- 本讲的目标是形成统一的资源核算框架，而不是孤立记忆 API。
- 后文每个公式都服务于一个实际判断：能否放下、要跑多久、瓶颈在哪里。

## 一、先做两道数量级估算

### 1. 训练 70B 模型需要多久

对以矩阵乘法为主、上下文不太长的 Transformer，训练总计算量常用 \(6NP\) 粗估。把参数量 \(P=70\times10^9\)、训练 token 数 \(N=15\times10^{12}\) 代入：

$$
C_{\text{train}} \approx 6NP
$$

- \(C_{\text{train}}\)：完整训练所需浮点运算数，单位 FLOPs。
- \(N\)：训练中处理的数据点数；语言模型里通常近似为 token 数。
- \(P\)：模型参数量。
- 系数 \(6\)：前向约 \(2NP\)，反向约 \(4NP\)，后文会推导。

一张 H100 的稠密 16 位矩阵乘法峰值按课件取 \(1979/2\) TFLOP/s，再乘 50% MFU、1024 张卡和每天的秒数：

$$
C_{\text{day}}
=
\frac{1979\times10^{12}}{2}
\times 0.5
\times 1024
\times 86400
$$

- \(C_{\text{day}}\)：1024 张 H100 一天实际可完成的 FLOPs。
- \(1979\times10^{12}\)：带结构化稀疏时的标称 FLOP/s。
- 除以 \(2\)：换成不利用稀疏的稠密峰值。
- \(0.5\)：假定 MFU 为 50%。
- \(1024\)：GPU 数量。
- \(86400\)：一天的秒数。

于是训练时间约为：

$$
T_{\text{days}}
=
\frac{6\times70\times10^9\times15\times10^{12}}
{C_{\text{day}}}
\approx 143.93\ \text{days}
$$

- \(T_{\text{days}}\)：训练天数。
- 分子：训练总 FLOPs。
- \(C_{\text{day}}\)：整套集群每天的有效计算量。

口头讲解把结果约成 143 天；按官方代码的数值实际约为 143.93 天，因此更稳妥的表述是“约 144 天”。这仍是理想化估算：未计入故障、评估、保存 checkpoint、数据等待与通信波动。

### 2. 8 张 80 GB H100 最多放多大 AdamW 模型

采用常见混合精度账本，每个参数至少对应：

- bf16 参数：2 bytes；
- bf16 梯度：2 bytes；
- fp32 Adam 一阶矩：4 bytes；
- fp32 Adam 二阶矩：4 bytes。

因此仅模型状态就需要：

$$
M_{\text{state/param}} = 2+2+4+4=12\ \text{bytes/parameter}
$$

- \(M_{\text{state/param}}\)：每个参数对应的模型状态显存。
- 四项依次是参数、梯度、一阶矩、二阶矩。

8 张 80 GB 卡的理论参数上界为：

$$
P_{\max}
=
\frac{8\times80\times10^9}{12}
\approx 5.33\times10^{10}
=53.3\text{B}
$$

- \(P_{\max}\)：理论最大参数量。
- \(8\times80\times10^9\)：按十进制 GB 计算的总显存字节数。
- \(12\)：每参数模型状态字节数。

![视频中完成的 12 bytes/parameter 显存上界估算](assets/video-capacity-12-bytes-per-parameter.jpg)

*图：讲者在完整揭示公式后得到约 53.33B，并立刻强调激活尚未计入。（字幕定位：00:02:51--00:03:35）*

> [!WARNING]
> 53.3B 是明显偏乐观的上界，不是“8 张 H100 真能直接训练 53B”的承诺。激活、临时 buffer、CUDA context、通信 bucket 和显存碎片都要占空间；若不做 ZeRO/FSDP 等切分，每张卡还可能持有完整模型副本。

### 一张可复用的 napkin-math 清单

估算训练资源时，按下面顺序最不容易漏项：

1. 明确总工作量：token 数、参数量、序列长度、训练步数。
2. 明确硬件理论值对应的 dtype，以及是否含稀疏加速。
3. 用 MFU 把理论峰值折成可实现吞吐。
4. 分清总集群资源与单卡资源。
5. 把结果标成上界、下界或经验估计，并写出忽略项。

### 本章小结

- \(6NP\) 把模型与数据规模转成训练 FLOPs，70B/15T/1024 H100 的理想估算约 144 天。
- AdamW 混合精度模型状态常按 12 bytes/parameter 起算，8×80 GB 的纯状态上界约 53.3B。
- 数量级估算的价值是暴露假设；任何未计入的激活、通信与系统开销都必须明确写出。

## 二、张量、dtype 与 device：先把显存账算清楚

### 张量不仅有数值，还有三份元数据

语言模型中的数据、参数、梯度、优化器状态和激活，最终都是张量。理解一个张量至少要同时看：

- **shape**：每个轴多长、语义是什么；
- **dtype**：每个元素如何编码、占多少字节；
- **device**：位于 CPU 内存还是某张 GPU 的显存。

例如 Transformer 常见 rank-4 激活可写成 \((B,S,H,D)\)：

- \(B\)：batch size；
- \(S\)：sequence length；
- \(H\)：attention head 数；
- \(D\)：每个 head 的隐藏维度。

张量显存由元素个数与单元素字节数相乘：

$$
M_{\text{tensor}}
=
\left(\prod_{i=1}^{r} d_i\right)s
$$

- \(M_{\text{tensor}}\)：张量占用的字节数。
- \(r\)：张量 rank，即轴数。
- \(d_i\)：第 \(i\) 个轴的长度。
- \(s\)：dtype 的单元素字节数。

例如默认 fp32 的 \(4\times8\) 张量有 32 个元素，每个 4 bytes，共 128 bytes。GPT-3 前馈层中一个 \(49152\times12288\) 的 fp32 权重矩阵约占 2.25 GiB，这说明 dtype 不是实现细节，而是模型能否放入设备的首要条件。

### fp32、fp16 与 bf16 的取舍

![fp32 位布局](assets/fp32.png)

*图：fp32 由 1 位符号、8 位指数和 23 位尾数组成。（对应视频字幕区间：00:05:56--00:07:16）*

![fp16 位布局](assets/fp16.png)

*图：fp16 减少到 5 位指数、10 位尾数，显存减半但动态范围明显变窄。（对应视频字幕区间：00:07:16--00:09:00）*

![bf16 位布局](assets/bf16.png)

*图：bf16 保留 fp32 的 8 位指数，只缩短尾数，因此兼顾 2-byte 存储与较大动态范围。（对应视频字幕区间：00:09:00--00:10:50）*

| dtype | 每元素 | 指数位 | 尾数位 | 训练中的主要特点 |
|---|---:|---:|---:|---|
| fp32 | 4 B | 8 | 23 | 范围和精度都较好，但显存与带宽成本高 |
| fp16 | 2 B | 5 | 10 | 精度尚可，动态范围窄，\(10^{-8}\) 可能下溢为 0 |
| bf16 | 2 B | 8 | 7 | 动态范围接近 fp32，分辨率较粗，通常更适合大模型训练 |

这里要区分两个概念：

- **动态范围**决定“多大或多小的数还能表示”；
- **分辨率/精度**决定“相邻可表示数有多密”。

bf16 不是“全面优于 fp16”：它用更多指数位换取范围，也因此只有更少尾数位。但深度学习往往更怕梯度直接上溢/下溢，而能容忍一定舍入噪声。

### 混合精度与 AMP

一个常见策略是让大体量对象使用 bf16，让跨很多步累积的小状态使用 fp32：

- 参数、激活、梯度：bf16；
- 优化器的一阶/二阶矩：fp32；
- 对数、指数、归一化等数值敏感操作：按框架策略保留更高精度。

PyTorch AMP 用上下文管理器自动选择合适精度。下面代码的角色，是声明“在这一区域内，安全的 CUDA 运算优先使用 bf16”：

~~~python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    output = model(inputs)
    loss = loss_fn(output, targets)
~~~

离开上下文后，代码恢复原来的默认精度策略。AMP 是按算子调度 dtype，不等于把上下文里新建的每个张量无条件变成 bf16，也不等于数值稳定性从此无需检查。

> [!NOTE]
> 视频还讨论了 FP8 与 NVFP4。它们依靠多个格式、分块 scale 和 NVIDIA 库内的专用 kernel 扩大可用范围。关键思想不是“位数越少越好”，而是用局部缩放和算子选择，把量化误差控制在训练可承受范围内。

### device：CPU 与 GPU 之间不是透明的

![CPU 与 GPU 间显式搬运张量](assets/cpu-gpu.png)

*图：张量必须显式位于执行算子的设备上，CPU↔GPU 复制本身也有成本。（对应视频字幕区间：00:17:17--00:17:58）*

PyTorch 默认在 CPU 上创建张量。把模型放到 CUDA 后，输入也必须迁移到相同 device：

~~~python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
y = model(x)
~~~

这段代码先统一选择设备，再把模型和输入放到同一处。若遗漏输入迁移，通常会得到 device mismatch；若频繁在循环中往返复制，即使代码能运行，PCIe/NVLink 传输也可能成为瓶颈。

### 本章小结

- 张量必须联合检查 shape、dtype 与 device。
- 显存等于元素数乘单元素字节数；降低 dtype 同时降低存储与带宽需求。
- bf16 以尾数精度换动态范围，混合精度则让不同对象采用不同 dtype。
- CPU/GPU 搬运是显式且有成本的，模型和输入必须位于兼容设备。

## 三、einops：用名字而不是位置理解维度

### 为什么负索引很危险

传统写法 <code>x @ y.transpose(-2, -1)</code> 很短，但 <code>-2</code>、<code>-1</code> 的语义只存在于程序员脑中。模型从三维扩成四维，或者 batch 维发生广播时，代码可能不报错却算错轴。

einops 的核心价值是：把“轴的位置”改写成“轴的名字”，让 shape 推理成为代码的一部分。

### einsum：命名收缩轴

下面代码的角色，是对 batch 内两个序列的 hidden 维做内积，得到两两相似度：

~~~python
from einops import einsum

# x: [batch, seq1, hidden]
# y: [batch, seq2, hidden]
scores = einsum(
    x,
    y,
    "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2",
)
~~~

输出里保留 <code>batch seq1 seq2</code>，而输入中出现、输出中消失的 <code>hidden</code> 会被乘后求和。也可用省略号保留任意数量的前导维：

~~~python
scores = einsum(
    x,
    y,
    "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2",
)
~~~

这里的省略号表达“这些维度参与广播并原样保留”，不是“随便忽略”。

### reduce：明确消去哪一轴

下面代码的角色，是沿 hidden 维求和，同时保留此前所有维度：

~~~python
from einops import reduce

# x: [..., hidden]
y = reduce(x, "... hidden -> ...", "sum")
~~~

如果改成 <code>"mean"</code>、<code>"max"</code> 等，就改变归约规则。读者无需反查 <code>dim=-1</code> 当前究竟对应哪个语义轴。

### rearrange：拆开与合并复合维度

多头计算经常把 <code>heads × hidden</code> 压平为一个 total hidden 轴。下面代码依次完成“拆头 → 每头线性变换 → 合并”：

~~~python
from einops import einsum, rearrange

# x: [seq, total_hidden] = [3, 8]
# w: [hidden1, hidden2] = [4, 4]
x = rearrange(
    x,
    "... (heads hidden1) -> ... heads hidden1",
    heads=2,
)
# x: [3, 2, 4]

x = einsum(
    x,
    w,
    "... hidden1, hidden1 hidden2 -> ... hidden2",
)
# x: [3, 2, 4]

x = rearrange(
    x,
    "... heads hidden2 -> ... (heads hidden2)",
)
# x: [3, 8]
~~~

括号表示多个轴的乘积。已知 total hidden 为 8、heads 为 2，einops 可推断 hidden1 为 4；若不能整除，会立即报错。这种“让 shape 假设可执行”比在注释里写一句更可靠。

> [!WARNING]
> 轴名只保证账目清楚，不替你保证数学意图正确。两个同长度轴即使语义不同，也可能通过 shape 检查。仍应在每个关键变换旁标出输入、输出 shape，并用小张量测试数值。

### 本章小结

- <code>einsum</code> 命名保留轴和收缩轴，适合矩阵乘法、注意力与批量内积。
- <code>reduce</code> 把归约掉的轴写在表达式中。
- <code>rearrange</code> 用括号拆分/合并复合维度，使多头结构更可读。
- einops 的真正收益是把 shape 假设变成可检查的代码。

## 四、从 \(2BDK\) 到 FLOP/s 与 MFU

### FLOPs 和 FLOP/s 不是同一件事

- **FLOPs** 是完成某项任务做了多少浮点运算，描述工作量。
- **FLOP/s** 是每秒完成多少浮点运算，描述执行速度。

对 \(X\in\mathbb{R}^{B\times D}\) 和 \(W\in\mathbb{R}^{D\times K}\)，输出有 \(BK\) 个元素。每个元素做 \(D\) 次乘法和 \(D-1\) 次加法，因此精确工作量是：

$$
C_{\text{matmul}}=BK(2D-1)\approx2BDK
$$

- \(C_{\text{matmul}}\)：这次矩阵乘法的 FLOPs。
- \(B\)：输入行数，训练示例里通常对应 batch 中的数据点数。
- \(D\)：收缩维长度。
- \(K\)：输出维度。
- 近似号：当 \(D\) 很大时忽略相对较小的 \(-BK\)。

测出运行时间后，实际吞吐必须用工作量除以时间：

$$
R_{\text{actual}}
=
\frac{C_{\text{matmul}}}{T_{\text{measured}}}
$$

- \(R_{\text{actual}}\)：实际 FLOP/s。
- \(C_{\text{matmul}}\)：已知的运算 FLOPs。
- \(T_{\text{measured}}\)：同步后测得的秒数。

> [!WARNING]
> 字幕在这一处把口头关系转写成了“FLOPs times time”，但量纲与官方代码都明确是 FLOPs 除以时间。CUDA 默认异步，计时前后还必须同步，否则测到的可能只是 kernel 提交时间。

一个最小、可复用的 GPU benchmark 结构如下。代码的角色是预热、同步并记录多次耗时：

~~~python
import time
import torch

def benchmark(fn, trials=5):
    for _ in range(2):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return min(times)
~~~

取最小值接近“系统干扰较少时”的 kernel 性能；若目标是用户体验或稳定吞吐，也应报告中位数和尾延迟。

### MFU：离对应精度的硬件峰值还有多远

MFU 用实际吞吐除以该 dtype、该模式下的硬件标称峰值：

$$
\mathrm{MFU}
=
\frac{R_{\text{actual}}}{R_{\text{promised}}}
$$

- \(\mathrm{MFU}\)：Model FLOPs Utilization，取值通常在 0 到 1。
- \(R_{\text{actual}}\)：按实际工作量与同步时间算得的 FLOP/s。
- \(R_{\text{promised}}\)：硬件规格表在相同 dtype、稀疏模式下的峰值 FLOP/s。

![视频中的 MFU 定义与实测路径](assets/video-mfu-definition.jpg)

*图：完整画面同时给出实际 FLOP/s、promised FLOP/s 与二者比值；讲者把 MFU ≥ 0.5 作为相当不错的经验水平。（字幕定位：00:37:12--00:38:42）*

> [!NOTE]
> 官方代码里的线性层 benchmark 默认创建 fp32 张量，所以该例应比较 H100 的 fp32 峰值（代码映射为 67.5 TFLOP/s），不能拿前面用于 bf16/fp16 稠密估算的 \(1979/2\) TFLOP/s 直接作分母。

视频现场还用 8 张 H100 的累计能力做数量级感知：先说“两周”，随后自我修正为“一周、约 \(5\times10^{21}\) FLOPs”；当前官方源码仍写“两周”，计算约 \(9.58\times10^{21}\) FLOPs。两者只是时长假设不同，最终讲义保留这项版本差异，避免混成同一个数字。

### MFU 低不一定说明“代码写坏了”

MFU 把通信、调度、数据等待和低算术强度都算进损失，却不告诉你是哪一项造成的。小矩阵、形状不适合 Tensor Core、dtype 不匹配、频繁 kernel launch 或显存带宽饱和，都可能让 MFU 远低于 1。下一章用 arithmetic intensity 把“算得慢”拆成计算瓶颈与内存瓶颈。

### 本章小结

- \(BK(2D-1)\) 是矩阵乘法精确 FLOPs，\(2BDK\) 是大维度下的常用近似。
- 实际 FLOP/s 等于 FLOPs 除以同步后的耗时。
- MFU 的分母必须与实测 dtype、稀疏模式一致。
- MFU 是结果指标；定位根因还需要算术强度与 Roofline。

## 五、算术强度与 Roofline：为什么算力常在等数据

### 一次算子的时间来自两条路径

![数据在内存与计算单元之间流动](assets/compute-memory.png)

*图：计算前要从内存取输入，计算后还要写回输出；两类资源可以成为不同瓶颈。（对应视频字幕区间：00:40:50--00:41:26）*

理想化地假设数据搬运和计算完全重叠，则算子时间由较慢一项决定：

$$
T_{\text{op}}
=
\max\left(
\frac{Q_{\text{bytes}}}{BW},
\frac{C_{\text{flops}}}{R_{\text{peak}}}
\right)
$$

- \(T_{\text{op}}\)：算子理想执行时间。
- \(Q_{\text{bytes}}\)：从显存读写的总字节数。
- \(BW\)：显存带宽，单位 bytes/s。
- \(C_{\text{flops}}\)：算子浮点运算数。
- \(R_{\text{peak}}\)：加速器对应 dtype 的峰值 FLOP/s。
- \(\max\)：两条路径即使重叠，也必须等待较慢者完成。

工作负载的算术强度定义为每搬运一字节完成多少 FLOPs：

$$
I_{\text{workload}}
=
\frac{C_{\text{flops}}}{Q_{\text{bytes}}}
$$

- \(I_{\text{workload}}\)：工作负载算术强度，单位 FLOP/byte。
- \(C_{\text{flops}}\)：算术工作量。
- \(Q_{\text{bytes}}\)：必要数据移动量。

硬件的“加速器强度”则是峰值算力与带宽之比：

$$
I_{\text{accelerator}}
=
\frac{R_{\text{peak}}}{BW}
$$

- \(I_{\text{accelerator}}\)：从 memory-bound 转向 compute-bound 的阈值。
- \(R_{\text{peak}}\)：硬件峰值 FLOP/s。
- \(BW\)：显存带宽。

课件用 H100 稠密 16 位峰值 \(989.5\) TFLOP/s 和 \(3.35\) TB/s，得到约 \(295.37\) FLOP/byte。低于这个阈值偏 memory-bound，高于它才有机会偏 compute-bound。

![视频中 H100 的算术强度判据](assets/video-arithmetic-intensity-threshold.jpg)

*图：完整画面同时展示通信时间、计算时间、硬件阈值约 295.37 FLOP/byte，以及 ReLU 仅 0.25 FLOP/byte。（字幕定位：00:45:56--00:47:42）*

### 用五个算子建立直觉

以下估算沿用课件的 bf16：读写一个元素各 2 bytes，并假设输入只从 HBM 读取一次、输出只写回一次。

| 算子 | 近似 FLOPs | 近似字节数 | 算术强度 | 典型判断 |
|---|---:|---:|---:|---|
| ReLU，长度 \(n\) | \(n\) | \(4n\) | \(1/4\) | memory-bound |
| GELU，长度 \(n\) | \(20n\) | \(4n\) | \(5\) | 仍 memory-bound |
| 点积，长度 \(n\) | \(2n-1\) | \(4n+2\) | \(\approx1/2\) | memory-bound |
| 矩阵×向量，\(n\times n\) | \(n(2n-1)\) | \(\approx2n^2\) | \(\approx1\) | memory-bound |
| 矩阵×矩阵，\(n\times n\) | \(n^2(2n-1)\) | \(6n^2\) | \(\approx n/3\) | \(n\) 足够大时 compute-bound |

GELU 比 ReLU 做更多计算，却移动相近的数据。课件采用常见 tanh 近似：

$$
\mathrm{GELU}(x)
\approx
\frac{x}{2}
\left[
1+\tanh\left(
\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)
\right)
\right]
$$

- \(x\)：输入标量。
- \(\mathrm{GELU}(x)\)：平滑门控后的输出。
- \(\tanh\)：双曲正切。
- \(0.044715\)：该近似的拟合系数。

孤立 kernel 中，ReLU 未必会按 FLOPs 比例显著快于 GELU，因为两者都可能主要等待相同规模的数据搬运。

对 \(n=1024\) 的 bf16 方阵乘法，课件得到约 \(341.17\) FLOP/byte，已经越过 \(295.37\) 的 H100 阈值：

![视频中 1024 方阵乘法跨过计算瓶颈阈值](assets/video-matmul-compute-bound.jpg)

*图：完整画面展示矩阵乘法强度约 341.17，大于硬件强度约 295.37，因此该形状进入 compute-bound 一侧。（字幕定位：00:51:21--00:52:59）*

> [!WARNING]
> “矩阵乘法是 compute-bound”必须带尺寸与复用条件。小 GEMM、瘦长矩阵、矩阵向量乘法以及自回归逐 token 解码，可能复用不足而 memory-bound。dtype 改变时，峰值算力和每元素字节数也会同时改变阈值。

### Roofline 把两个上限画在一张图里

Roofline 的横轴是算术强度，纵轴是可实现 FLOP/s。低强度区域受带宽限制，性能随强度线性上升；越过拐点后受峰值算力限制，形成水平屋顶。

![视频中的 Roofline 图](assets/video-roofline.jpg)

*图：斜线是带宽上限，水平线是计算峰值，拐点就是 accelerator intensity。（字幕定位：00:55:49--00:57:01）*

在只考虑这两个理想上限时，MFU 上界可写为：

$$
\mathrm{MFU}_{\text{roofline}}
=
\min\left(
1,
\frac{I_{\text{workload}}}{I_{\text{accelerator}}}
\right)
$$

- \(\mathrm{MFU}_{\text{roofline}}\)：Roofline 模型给出的理想利用率上界。
- \(I_{\text{workload}}\)：算子算术强度。
- \(I_{\text{accelerator}}\)：硬件算力/带宽阈值。
- \(1\)：达到计算峰值后的上限。

这是官方课件给出的模型化关系，不表示真实系统一定达到该值。kernel 启动、通信、缓存行为、数据布局和并行同步会继续压低实际 MFU。

### 本章小结

- 算子时间由数据移动与计算两条路径中更慢的一条控制。
- 算术强度是 FLOPs/byte；与硬件的 FLOP/s÷bytes/s 阈值比较即可初判瓶颈。
- 元素级算子、点积和矩阵向量乘法常偏 memory-bound；足够大的 GEMM 才容易 compute-bound。
- Roofline 给出理想性能上界，不替代真实 profiling。

## 六、反向传播为何把 \(2\) 变成 \(6\)

### 计算图与链式法则

![深层网络的前向激活链](assets/deep-network.png)

*图：每层把前一层激活变成下一层激活；反向必须沿同一路径逆序传播梯度。（对应视频字幕区间：00:57:45--00:58:14）*

考虑标量损失：

$$
\ell=\frac{1}{2}(x^\top w-5)^2
$$

- \(\ell\)：标量损失。
- \(x\)：输入向量。
- \(w\)：需要求梯度的参数向量。
- \(x^\top w\)：模型预测。
- \(5\)：示例目标值。

PyTorch 只会为 <code>requires_grad=True</code> 的叶子张量建立所需计算图；调用 <code>loss.backward()</code> 后，梯度累加进 <code>w.grad</code>。注意是“累加”而非覆盖，这正是后文梯度累积可行的基础。

### 一层线性变换的反向包含两个矩阵乘法

设一层前向为 \(H_2=H_1W_2\)。前向只做一次矩阵乘法；反向既要把梯度传给输入，也要计算权重梯度：

$$
\frac{\partial\ell}{\partial H_1}
=
\frac{\partial\ell}{\partial H_2}W_2^\top,
\qquad
\frac{\partial\ell}{\partial W_2}
=
H_1^\top\frac{\partial\ell}{\partial H_2}
$$

- \(\ell\)：最终标量损失。
- \(H_1\)：该层输入激活。
- \(H_2\)：该层输出激活。
- \(W_2\)：该层权重。
- \(\partial\ell/\partial H_2\)：从后续层传入的上游梯度。
- \(\partial\ell/\partial H_1\)：继续向前一层传播的梯度。
- \(\partial\ell/\partial W_2\)：用于更新权重的梯度。

若矩阵规模相近，每个矩阵乘法成本近似相同：

- 前向：1 次，约 \(2BD^2\) FLOPs；
- 反向：2 次，约 \(4BD^2\) FLOPs；
- 合计：3 次，约 \(6BD^2\) FLOPs。

推广到参数量 \(P\)、一个 batch 含 \(B\) 个数据点时，单个训练 step 是：

$$
C_{\text{step}}\approx6BP
$$

- \(C_{\text{step}}\)：单个 batch 的训练 FLOPs。
- \(B\)：本 step 的数据点数；语言模型里应结合 token 数理解。
- \(P\)：参与稠密计算的参数量。
- \(6\)：前向 \(2\) 加反向 \(4\)。

完整数据集共有 \(N\) 个数据点时才写：

$$
C_{\text{train}}\approx6NP
$$

- \(C_{\text{train}}\)：完整训练过程 FLOPs。
- \(N\)：训练总数据点/token 数。
- \(P\)：模型参数量。

![视频中前向 2、反向 4、合计 6 的 FLOPs 账本](assets/video-training-flops-6np.jpg)

*图：讲者先对单层反向的两个矩阵乘法计数，再总结为完整训练的 \(6NP\) 近似。（字幕定位：01:05:40--01:06:43）*

> [!NOTE]
> 视频总结页的“per training step 为 \(6NP\)”容易混淆符号：若 \(N\) 指完整训练集/token 总数，它描述完整训练；若讨论单 step，应写 \(6BP\)。此外，长上下文下 attention 的序列长度二次项不可忽略，\(6NP\) 不再覆盖所有主要计算。

### 用模块代码看激活为何要保留

下面代码的角色，是构造 \(L\) 个“线性层 + ReLU”顺序连接的深层网络：

~~~python
class Block(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(dim, dim) / math.sqrt(dim)
        )

    def forward(self, x):
        return torch.relu(x @ self.weight)

class DeepNetwork(torch.nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Block(dim) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
~~~

每层反向计算权重梯度时需要该层前向输入，因此 autograd 默认保存中间激活。网络越深、batch 越大，这部分显存越显著。

### 本章小结

- 反向传播沿计算图逆序应用链式法则，既算输入梯度也算参数梯度。
- 对稠密线性层，反向约为前向 FLOPs 的 2 倍，训练合计约为前向的 3 倍。
- 单 step 常写 \(6BP\)，完整训练常写 \(6NP\)；不要混用 batch 与全数据集符号。
- 中间激活是反向所需的缓存，因此成为训练显存的核心组成。

## 七、优化器状态与完整训练循环

### 从 SGD 到 AdaGrad、RMSProp 与 Adam

本讲用 AdaGrad 展示“优化器为何需要参数以外的状态”。它累积历史平方梯度，再按每个坐标的累计尺度调整步长：

$$
G_t=G_{t-1}+g_t^2
$$

- \(G_t\)：第 \(t\) 步后每个参数坐标的平方梯度累积量。
- \(G_{t-1}\)：上一步优化器状态。
- \(g_t\)：当前梯度。
- 平方：逐元素平方。

参数更新为：

$$
\theta_t
=
\theta_{t-1}
-\eta\frac{g_t}{\sqrt{G_t+\epsilon}}
$$

- \(\theta_t\)：更新后的参数。
- \(\theta_{t-1}\)：更新前参数。
- \(\eta\)：学习率。
- \(g_t\)：当前梯度。
- \(G_t\)：累计平方梯度。
- \(\epsilon\)：避免除零的小常数。

关系可这样记：

- momentum：SGD 加梯度的指数滑动平均；
- AdaGrad：SGD 除以历史平方梯度的累计尺度；
- RMSProp：把 AdaGrad 的平方梯度累计改成指数滑动平均；
- Adam：结合一阶动量与二阶矩缩放。

### 训练显存的四本账

对本讲的简化深网，bf16 参数/梯度、fp32 AdaGrad 状态下：

$$
M_{\text{total}}
=
2P+2P+4P+2BDL
$$

- \(M_{\text{total}}\)：这份简化账本的总字节数。
- \(P\)：参数量。
- 第一个 \(2P\)：bf16 参数。
- 第二个 \(2P\)：bf16 梯度。
- \(4P\)：fp32 AdaGrad 二阶状态；Adam 会是 \(8P\)。
- \(B\)：batch size。
- \(D\)：每层激活宽度。
- \(L\)：层数。
- \(2BDL\)：按每层保存一个 bf16 激活的简化估算。

这仍未含临时张量、allocator 保留区和通信 buffer。激活项依赖 batch、序列长度和模型结构，而参数/梯度/优化器状态主要随 \(P\) 线性增长。

> [!WARNING]
> 视频中某个显存 inspector 画面把已经是“总参数字节”的变量再次乘 2 或 4，变量命名与注释不一致。上式按官方源码的最终单位口径重写：每一项都直接是 bytes，避免对 <code>parameter_memory</code> 重复乘 dtype 系数。

### 完整训练循环：视频略过，但源码保留

讲者在 01:12:04 左右为了时间跳过了训练循环细讲。下面内容来自同讲官方源码，作用是把此前分散的对象串起来：

~~~python
model = DeepNetwork(dim=D, num_layers=L).to(device)
optimizer = AdaGrad(model.parameters(), lr=0.01)

for step in range(num_train_steps):
    x, y = get_batch()

    # 1. forward
    pred_y = model(x).mean()
    loss = torch.nn.functional.mse_loss(pred_y, y)

    # 2. backward
    loss.backward()

    # 3. update
    optimizer.step()

    # 4. clear accumulated gradients
    optimizer.zero_grad(set_to_none=True)
~~~

四步的资源生命周期是：

1. **forward** 创建激活并构造 autograd 图；
2. **backward** 读取激活，生成/累加梯度；
3. **step** 读取梯度和优化器状态，更新参数与状态；
4. **zero_grad** 释放或清空梯度，准备下一步。

<code>set_to_none=True</code> 通常比填零更节省写带宽，并能区分“没有梯度”与“梯度恰为零”。若把清零放错位置，可能意外累积梯度或在更新前把梯度删除。

> [!NOTE]
> 源码示例让标量 <code>pred_y</code> 与向量 <code>y</code> 做 MSE，会发生广播；它适合演示训练循环，却不是严谨的监督建模结构。真实任务应先断言预测与目标 shape 符合设计。

### 本章小结

- 优化器通过保存历史统计量改变每个参数的更新尺度，因此会增加持久显存。
- AdaGrad 状态为 4 bytes/parameter；常见混合精度 Adam 两个 fp32 矩合计 8 bytes/parameter。
- 训练循环的顺序是 forward、backward、step、zero_grad。
- 应对每项显存统一使用“字节”口径，并检查广播是否掩盖 shape 错误。

## 八、显存不足时：梯度累积与激活检查点

### 梯度累积：把大 batch 拆成多个 micro-batch

大 batch 往往有更稳定的梯度估计，但一次放入全部样本会让激活随 \(B\) 增长。梯度累积利用 <code>.backward()</code> 默认累加梯度的行为：

~~~python
optimizer.zero_grad(set_to_none=True)

for micro_x, micro_y in micro_batches:
    pred = model(micro_x)
    loss = loss_fn(pred, micro_y) / num_micro_batches
    loss.backward()

optimizer.step()
~~~

这段代码每次只保留一个 micro-batch 的激活，所有 micro-batch 的梯度累加完才更新参数。若损失默认取 micro-batch 均值，除以 <code>num_micro_batches</code> 可使最终梯度等价于完整大 batch 的平均梯度。

![视频中的梯度累积步骤](assets/video-gradient-accumulation.jpg)

*图：完整画面列出计算 micro-batch、暂不清梯度、累计到目标 batch 后再更新并清零。（字幕定位：01:12:30--01:13:19）*

把 batch 从 \(B\) 拆成 \(K\) 个 micro-batch 后，简化激活峰值近似变为：

$$
M_{\text{act,peak}}
\approx
2\frac{B}{K}DL
$$

- \(M_{\text{act,peak}}\)：单次 micro-batch 的激活峰值字节数。
- \(2\)：bf16 每元素 2 bytes。
- \(B\)：目标有效 batch size。
- \(K\)：micro-batch 数量。
- \(D\)：激活宽度。
- \(L\)：层数。

> [!WARNING]
> 视频口头有一句“save compute”，但梯度累积主要节省的是**激活峰值显存**，并不减少处理相同样本的总算术量；更多小 kernel 还可能降低吞吐。含 dropout、BatchNorm 或按 step 调度器时，micro-batch 与单次大 batch 也未必完全等价。

### 激活检查点：前向少存，反向重算

Activation checkpointing、gradient checkpointing 与 rematerialization 在本讲中指同一思想：

- 前向只保存部分层的激活；
- 反向需要缺失激活时，从最近 checkpoint 重新前向；
- 用额外计算换更低显存。

PyTorch 的最小形式如下。代码的角色，是让某层的中间激活不常驻，而在反向时重新执行该层：

~~~python
from torch.utils.checkpoint import checkpoint

for layer in self.layers:
    x = checkpoint(layer, x, use_reentrant=False)
~~~

随机算子必须正确保存/恢复 RNG 状态；有副作用、依赖全局可变状态或前后不确定的函数，不适合直接重算。

### checkpoint 频率的复杂度

对 \(L\) 层链式网络，有三种理想策略：

| 策略 | 保存的激活规模 | 额外重算 |
|---|---:|---:|
| 保存每层 | \(O(L)\) | 无 |
| 一个都不保存，每次从头重算 | \(O(1)\) | \(O(L^2)\) |
| 每隔 \(\sqrt L\) 层保存 | \(O(\sqrt L)\) | 总重算 \(O(L)\) |

![视频中的三种 checkpoint 频率](assets/video-checkpoint-frequency.jpg)

*图：画面把“全存、全不存、每隔 \(\sqrt L\) 层存”并列，直观展示显存与重算的折中。（字幕定位：01:15:33--01:16:15）*

> [!WARNING]
> 视频画面把第三种策略的 recomputation 标作 \(O(\sqrt L)\)，而官方源码写 \(O(L)\)。可兼容的精确定义是：checkpoint 间最长重算段为 \(O(\sqrt L)\)，但遍历整网反向时，总额外重算仍是 \(O(L)\)。另外，源码演示的“每层都调用 checkpoint”与“只在每隔 \(\sqrt L\) 层保存边界”不是同一调度策略。

### 两种技术解决不同的轴

- 梯度累积主要把激活峰值对 batch 的依赖从 \(B\) 降到 micro-batch 大小。
- 激活检查点主要降低对网络深度与中间激活数量的依赖。
- 两者都不会减少参数、梯度和优化器状态本身。
- 两者可组合，但会增加执行次数、调度开销或重算 FLOPs。

### 本章小结

- 梯度累积用多个 micro-batch 模拟有效大 batch，核心收益是降低激活峰值显存。
- 损失缩放、清梯度时机和有状态算子决定其是否与大 batch 等价。
- 激活检查点通过反向重算换显存，checkpoint 频率决定时空折中。
- 每隔 \(\sqrt L\) 层保存时，激活规模为 \(O(\sqrt L)\)，全网总额外重算应按 \(O(L)\) 理解。

## 总结与延伸

### 一张统一资源账

![视频结尾的全讲总结](assets/video-lecture-summary.jpg)

*图：结尾把张量、混合精度、算术强度、\(6NP\)、显存组成与两种显存优化串成同一条主线。（字幕定位：01:16:16--01:17:13）*

遇到任何训练算子，可以沿下面的顺序审计：

1. **语义账**：每个张量的轴分别是什么？einops 表达式是否保留/消去了正确维度？
2. **存储账**：元素数是多少？dtype 每元素几字节？参数、梯度、优化器状态、激活各自活多久？
3. **计算账**：精确 FLOPs 是多少？可否近似为 \(2BDK\)、\(6BP\) 或 \(6NP\)？
4. **时间账**：实测是否正确同步？实际 FLOP/s 与同 dtype 的峰值相比，MFU 多高？
5. **瓶颈账**：FLOPs/byte 位于 Roofline 拐点哪侧？应该减少搬运还是提高计算占用？
6. **交换账**：若 OOM，是拆 micro-batch，还是少存激活并接受重算？

### 本讲最容易带走的五个误区

1. **“低精度只是少占显存”**：它也改变带宽、Tensor Core 峰值、动态范围和 Roofline 阈值。
2. **“FLOPs 多就一定慢”**：memory-bound 算子增加少量计算可能几乎不增加耗时，算子融合甚至会因减少搬运而更快。
3. **“矩阵乘法总是 compute-bound”**：只有形状足够大、数据复用充分时成立。
4. **“梯度累积省计算”**：它主要省峰值激活显存，总 FLOPs 不会凭空消失。
5. **“checkpoint 的复杂度只有一个答案”**：要明确讨论最长重算段、全网总重算，还是具体 PyTorch 调度。

### 建议动手实验

- 把 einops 三个例子的 shape 改错一次，观察哪类错误能被立即捕获。
- 对 fp32、fp16、bf16 的同形状 GEMM 分别测吞吐，确认 MFU 分母也随 dtype 改变。
- 从小到大扫描方阵维度，画出实际 FLOP/s 与算术强度，寻找从 memory-bound 到 compute-bound 的转折。
- 在固定有效 batch 下改变 micro-batch 数，记录峰值显存、step 时间和最终梯度差异。
- 对同一深网比较无 checkpoint、逐层 checkpoint、分段 checkpoint 的峰值显存与重算时间。

### 向后续课程延伸

本讲的简化模型以稠密线性层为主。进入 Transformer 后，还要把 embedding、attention 的 \(S^2\) 项、MLP expansion、KV cache、通信与并行切分逐项加入账本。方法不变：所有复杂系统最终仍应还原成“哪些张量在何时以何种精度流向哪里，并为此做了多少算术”。

### 本章小结

- shape、dtype、device、FLOPs、bytes 和生命周期共同决定训练可行性。
- MFU 描述离峰值多远，算术强度解释为什么远。
- \(6NP\) 是有假设的近似，显存账也必须显式列出忽略项。
- 梯度累积和激活检查点是两种不同方向的显存—计算交换。
- 能把每一项写成带单位的账，才真正具备从零实现与优化语言模型的基础。
