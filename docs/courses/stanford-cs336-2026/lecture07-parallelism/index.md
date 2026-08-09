# CS336 2026 Lecture 7：并行训练——从 Collective 到数据、张量与流水线并行

> **课程**：Stanford CS336 — Language Modeling from Scratch（Spring 2026）  
> **讲次**：Lecture 7: Parallelism  
> **讲者**：Percy Liang  
> **视频**：[Stanford Online / YouTube](https://www.youtube.com/watch?v=SzpOcwdIL0Y)  
> **时长**：01:21:02  
> **材料说明**：讲义基于公开视频、人工英文字幕与官方课件源码交叉整理；公开仓库不再分发这些原始文件。

![Lecture 7 视频封面：Percy Liang 讲解 Parallelism](assets/cover.jpg)

这节课不是并行技术名词的罗列，而是在回答同一个系统问题：**当模型、数据和计算分布在多张 GPU 上时，应该复制什么、切分什么，又必须为此移动什么？**

前一讲的优化对象主要在单张 GPU 内部：算子融合、分块和数据复用都在减少 HBM 与计算单元之间的数据移动。本讲把边界扩展到多张 GPU、多个节点。计算资源增加后，显存容量和 FLOPs 都增加了，但 GPU 之间的通信会成为新的瓶颈。因此，一种并行策略是否有效，不能只看“用了多少张卡”，还要同时看：

- 每张 GPU 保存哪些参数、梯度、优化器状态和激活；
- 每张 GPU 负责哪一部分样本或哪一部分模型；
- 哪个张量需要跨卡移动，移动多少次、多少字节；
- 通信能否与计算重叠；
- 通信路径是在同一 NVLink 域，还是跨越较慢的节点间网络。

> [!IMPORTANT]
> 阅读后续每种策略时，始终用五个问题检查：**存什么、算什么、传什么、何时传、为什么值得传。**

---

## 从单卡内存墙到多卡通信墙

### 两个不同的扩展目标

多 GPU 训练常常混合了两个目标，但二者必须分开：

1. **容量扩展**：模型状态或激活放不进单张 GPU，需要把它们切到多张卡；
2. **吞吐扩展**：模型能放进单卡，但希望使用更多 FLOPs，在相同时间内处理更多 token。

数据并行通常首先解决吞吐问题，因为每张卡仍保留完整模型。张量并行和流水线并行会切开模型，因而能直接降低每卡的部分模型状态。真实的大模型训练通常把这些方法组合起来，而不是三选一。

### 层级越远，数据移动通常越贵

可以把训练机器理解为一个通信层级：

1. GPU 芯片内部的寄存器、共享内存与缓存；
2. 单张 GPU 的 HBM；
3. 单节点或同一高速域内的 NVLink / NVSwitch；
4. 跨节点的 InfiniBand 或 Ethernet / RoCE。

越往下走，通常延迟越高、有效带宽越低。于是“多一张 GPU”只增加了潜在算力；如果每一步都要在慢链路上搬运大量张量，训练未必按卡数线性加速。

课程首先给出一张全局地图：并行策略都建立在少数通信原语之上，最后再沿 batch、width、depth 三个轴切分训练。

![课程路线：rank、collective 与三种主要并行方式](assets/collective-overview.jpg)

*对应视频 `00:06:44--00:08:16`。画面给出四个 rank 及本课要讨论的 collective 列表。*

### 一个统一抽象：复制与切分

并行训练的许多设计都可以看成 replication 与 sharding 的交换：

- **复制**让本地计算更独立，减少即时通信，但会重复占用显存；
- **切分**降低每卡存储压力，但需要在某些计算边界重组数据；
- **重计算**用额外 FLOPs 换显存；
- **通信**把暂时缺失的数据从其他设备取来。

这四个动作会贯穿后续所有方法。真正困难的地方不是让某个示例“跑起来”，而是把代价安排在最快的链路上，并让通信尽可能藏在计算后面。

### 本章小结

- 多 GPU 训练同时面对容量扩展与吞吐扩展，两者不能混为一谈。
- GPU 数量增加并不保证线性加速，数据移动的层级决定了通信成本。
- 后续每种策略都可以用“复制、切分、重算、通信”四个动作来理解。

---

## Collective：多 GPU 协作的基本词汇

### rank 与 world size

分布式进程组中的每个参与者都有一个 **rank**，参与者总数叫 **world size**。本课大量使用四个进程，对应 rank 0、1、2、3，world size 为 4。

这里的 rank 是进程组内的全局编号。多节点训练时，还需要区分本机编号 `local_rank`：全局 rank 5 并不意味着它一定使用本机的 GPU 5。课程代码在单机上演示，因此二者差异被有意隐藏。

### 八种常见通信模式

下面用四个 rank 的小张量建立直觉。判断一种 collective 时，先问三件事：输入最初在哪些 rank、输出最终在哪些 rank、是否发生逐元素归约。

| 操作 | 直观变换 | 输出位置 | 常见用途 |
|---|---|---|---|
| broadcast | 一个源的完整数据复制给大家 | 所有 rank | 分发配置或权重 |
| scatter | 一个源的完整数据切片后分发 | 每个 rank 一片 | 分发分片 |
| gather | 各 rank 的分片收回 | 一个目标 rank | 汇总结果 |
| reduce | 各 rank 数据逐元素求和/平均等 | 一个目标 rank | 聚合统计量 |
| all-gather | 所有人交换分片并拼成完整张量 | 所有 rank | 临时重建参数或激活 |
| reduce-scatter | 先归约，再把结果切片分发 | 每个 rank 一片 | 聚合并保留分片梯度 |
| all-reduce | 归约后的完整结果复制给大家 | 所有 rank | DDP 同步梯度 |
| all-to-all | 每个 rank 都给每个目标发一片 | 每个 rank 收到按来源重排的数据 | MoE token 路由 |

几个容易混淆的关系：

- scatter 的逆操作是 gather；
- broadcast 与 reduce 的方向相反，但 reduce 还带有求和、平均、最大值等运算；
- all-gather 不是 reduce，它只交换并拼接分片；
- reduce-scatter 同时做归约和分片；
- all-reduce 可以分解为 reduce-scatter 后接 all-gather。

### 为什么 all-reduce 是 DDP 的主力

假设每个 rank 都计算出一份完整梯度。DDP 希望所有 rank 在更新前拿到同一份全局平均梯度，于是需要 all-reduce。它可以写成：

$$
\operatorname{AllReduce}(x_0,\ldots,x_{P-1})
=\operatorname{AllGather}\!\left(
\operatorname{ReduceScatter}(x_0,\ldots,x_{P-1})
\right).
$$

第一阶段把归约后的不同分片留在不同 rank；第二阶段交换这些分片，让每个 rank 得到完整结果。

![all-reduce 可拆成 reduce-scatter 与 all-gather](assets/all-reduce-relation.jpg)

*对应视频 `00:15:18--00:16:48`。画面并列展示输入、分片归约结果与最终完整结果。*

这种分解不仅是数学关系，也解释了后续方法的通信选择：传统 DDP 需要完整梯度，常用 all-reduce；ZeRO/FSDP 保留分片状态，因而更直接地使用 reduce-scatter 与 all-gather。

> [!QUOTE]
> “先关注 all-reduce，再把它拆成 reduce-scatter 和 all-gather。”——讲者在 `00:16:22--00:16:41` 给出的学习主线（意译）。

### 用 ring 模型估算通信量

令通信张量大小为 $S$ 字节，rank 数量为 $P$，链路有效带宽为 $\beta$，每个通信步骤的启动延迟为 $\alpha$。在经典 ring 模型中：

$$
V_{\text{RS}}\approx \frac{P-1}{P}S,
\qquad
V_{\text{AG}}\approx \frac{P-1}{P}S,
$$

因此

$$
V_{\text{AR}}\approx 2\frac{P-1}{P}S,
$$

而时间可粗略写为

$$
T_{\text{AR}}\approx 2(P-1)\alpha
+2\frac{P-1}{P}\frac{S}{\beta}.
$$

这个公式给出两点直觉：小消息更容易受多次启动的延迟项影响；大消息更容易受总字节量和带宽影响。它是 ring 的教学模型，不代表 NCCL 在所有拓扑、所有消息大小下都固定采用同一算法。

### all-to-all 与 MoE

all-to-all 是更一般的重排：每个 rank 把输入切成 $P$ 片，并把第 $j$ 片发给 rank $j$。输出看起来像把一个“来源 × 目的地”矩阵转置。

![all-to-all 的数据重排及 MoE 应用](assets/all-to-all-moe.jpg)

*对应视频 `00:16:48--00:19:44`。画面展示每个 rank 如何向所有目的 rank 发送不同元素。*

在 Mixture-of-Experts 中，token 会根据路由器选择专家；专家分布在不同设备时，就需要把 token 送到专家所在 rank，再把计算结果送回。all-to-all 能表达这种动态重排，但它也可能产生负载不均和网络热点，这些问题不由 API 本身自动解决。

### 本章小结

- collective 描述“数据如何在一组 rank 之间变换”，让上层算法不必手写每条点对点连接。
- DDP 的核心是 all-reduce；ZeRO/FSDP 更直接地使用 all-gather 与 reduce-scatter。
- ring 模型把通信成本拆成延迟项和带宽项，便于判断瓶颈。
- all-to-all 是 MoE token 路由的关键原语，但负载均衡仍需单独处理。

---

## 硬件拓扑、RDMA 与软件栈

### 拓扑是通信的价格表

代码里写同一个 `all_reduce`，并不表示所有 rank 对之间的代价相同。传统服务器中，GPU 可能先通过 PCIe 到 CPU，再经网卡跨节点；现代训练节点通常先用 NVLink 和 NVSwitch 建立节点内高速域，再通过 InfiniBand 或 Ethernet/RoCE 连接节点。

![现代 GPU 节点的 NVLink、NVSwitch 与节点间网络](assets/hardware-topology.jpg)

*对应视频 `00:23:27--00:26:23`。画面比较节点内高速互联与节点间网络，并给出典型训练拓扑。*

这会直接影响并行策略的放置：

- 张量并行在每层都可能交换激活，对延迟和带宽非常敏感，通常应放在高速域内；
- 流水线并行只在相邻阶段边界传递激活，往往更能容忍较慢链路；
- 数据并行通常每步同步梯度，可跨节点扩展，但仍需尽量重叠通信和反向计算。

### RDMA、RoCE 与“绕过 CPU”

RDMA 是远程直接内存访问的通信语义：数据路径可以减少 CPU 参与和额外复制，使设备内存之间更直接地移动。RoCE 则是在 Ethernet 上承载 RDMA 的一种方式。两者不是同一层级的同义词，也不是某一种特定线缆的名字。

![RDMA、扩展 NVLink 域与 RoCE 的关系](assets/rdma-roce.jpg)

*对应视频 `00:26:23--00:29:47`。画面强调 CPU bypass，并列出扩大 NVLink 域和使用 RoCE 的方向。*

> [!NOTE]
> 课中提到的 GPU 型号、互联带宽和机柜规模是特定硬件代际的例子。数字会随产品与配置改变；稳定不变的是“节点内与节点间链路存在层级”这一推理框架。

### NCCL 与 PyTorch 各负责什么

NCCL 提供面向 GPU 的 collective 实现，并根据拓扑、消息大小和运行环境选择底层算法与传输路径。PyTorch 的 `torch.distributed` 再向用户暴露统一接口，例如 `all_reduce`、`all_gather_into_tensor` 和 `reduce_scatter_tensor`。

![NCCL 与 PyTorch distributed 所处的软件栈](assets/nccl-pytorch-stack.jpg)

*对应视频 `00:30:03--00:37:13`。画面从 NCCL 的底层通信职责过渡到 PyTorch 的 collective 接口。*

可以把分层关系理解为：

```text
并行算法（DDP / TP / PP）
        ↓
PyTorch distributed API
        ↓
NCCL collective 与拓扑感知实现
        ↓
NVLink / NVSwitch / InfiniBand / Ethernet
```

NCCL 会启动 GPU 通信 kernel，但“有 NCCL”并不自动让任意布局都高效。进程映射、拓扑、张量大小、计算重叠和拥塞仍然会影响性能。

### 本章小结

- 同一个 collective 的成本取决于物理拓扑，rank 之间不是天然等价的。
- RDMA 描述减少 CPU 参与和复制的访问语义；RoCE 是在 Ethernet 上承载 RDMA 的方案。
- PyTorch 提供编程接口，NCCL 执行 GPU collective，硬件互联决定底层价格。

---

## 用 `torch.distributed` 验证通信原语

### 最小进程组

课程代码在一台机器上启动多个进程，并用本机地址完成 rendezvous：

```python
import os
import torch.distributed as dist

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "15623"
dist.init_process_group("nccl", rank=rank, world_size=world_size)
```

`MASTER_ADDR` 和 `MASTER_PORT` 只帮助进程彼此发现并建立进程组；这不意味着训练张量都要经过 rank 0 或 master。多节点环境还要为每个进程设置正确的设备和 local rank。

### all-reduce 会原地修改张量

```python
data = torch.tensor([float(rank)], device=device)
dist.all_reduce(
    tensor=data,
    op=dist.ReduceOp.SUM,
    async_op=False,
)
```

若四个 rank 的输入分别为 0、1、2、3，归约和为 6；调用完成后，每个 rank 的 `data` 都变成 6。这里最容易漏掉的是 **in-place**：返回值不是一个新的归约张量，输入本身被覆盖。

![课件中的 rank 查询与 all-reduce 原地调用](assets/all-reduce-code.jpg)

*对应视频 `00:40:11--00:41:34`。画面展示 `get_rank`、`get_world_size` 以及 `dist.all_reduce`。*

### reduce-scatter 与 all-gather 的形状契约

假设 world size 为 $P$，每个输出分片有 $N$ 个元素：

- reduce-scatter 的输入总大小通常是 $P\times N$，输出大小是 $N$；
- all-gather 的本地输入大小是 $N$，完整输出大小是 $P\times N$。

这正好形成一对形状上的对偶。课程先执行 reduce-scatter，再执行 all-gather，借此用代码验证“all-reduce 的两阶段分解”。

```python
# 每个 rank: input 有 P * N 个元素，output 有 N 个元素
dist.reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM)

# 每个 rank: input 有 N 个元素，output 有 P * N 个元素
dist.all_gather_into_tensor(output, input)
```

### `async_op=True` 不是“自动变快”

异步 collective 返回一个 work handle；调用返回只表示通信已被排入执行，并不保证数据已经可用：

```python
work = dist.all_reduce(data, async_op=True)

# 这里执行与 data 的归约结果无关的计算
independent_work()

work.wait()  # 第一次真正使用归约结果前等待
consume(data)
```

若发起后立刻 `wait()`，就没有可重叠的窗口。若在等待前读取结果，则存在正确性问题。异步的价值来自正确安排依赖，而不是布尔参数本身。

### barrier 与 CUDA synchronize 是两层等待

- `torch.cuda.synchronize()` 等待当前进程此前提交到 GPU 的 CUDA 工作完成；
- `dist.barrier()` 等待进程组中的所有 rank 都到达同一逻辑点。

计时时常常两者都需要：先等待本地 GPU 工作真正结束，再确保各 rank 从一致的阶段开始。`barrier()` 不能替代本地设备同步，设备同步也不能保证其他 rank 已经到达。

### 教学代码与生产代码的边界

课程代码刻意小而透明，但不应直接视作生产模板：

- 使用 `localhost`，只演示单节点 rendezvous；
- rank 与本地 GPU 的映射被简化；
- 例子没有容错、超时、弹性伸缩与多进程日志治理；
- 同步写法有助于理解语义，生产训练会更多地使用异步 bucket 和调度。

### 本章小结

- 进程组初始化负责发现和建组，不意味着训练张量经由 master 中转。
- collective 有严格的原地语义和形状契约；理解形状比背 API 更重要。
- 异步通信只有在等待之前存在独立计算时才带来重叠。
- CUDA synchronize 与 distributed barrier 分别解决设备队列和进程到达点问题。

---

## 通信性能如何测量

### 先把 GPU 的异步执行纳入计时

Python 调用 CUDA 算子通常只提交工作。若直接在调用前后读 CPU 时钟，测到的可能只是提交延迟。一个最小 benchmark 通常需要：

```python
# 1. warmup：排除首次初始化等开销
for _ in range(num_warmup):
    dist.all_reduce(data)

# 2. 对齐各 rank，并清空此前 CUDA 工作
torch.cuda.synchronize()
dist.barrier()

# 3. 计时真实执行
start = time.perf_counter()
dist.all_reduce(data)
torch.cuda.synchronize()
duration = time.perf_counter() - start
```

严谨测试还应重复多次并报告分布，而不是只报一次最好结果；还要注明 dtype、消息大小、rank 数、节点数、拓扑、算法、库版本和是否存在其他流量。

### “耗时”与“带宽”回答不同问题

一次 collective 花费 1.6 ms，并不能独立说明快慢，因为消息大小可能完全不同。用 ring all-reduce 的通信量模型，可定义一个算法带宽口径：

$$
\text{effective bandwidth}
=\frac{2(P-1)S/P}{t}.
$$

课件代码先计算所有 rank 合计的发送量，再除以 $P$ 和持续时间，代数上与上式一致：

```python
size_bytes = data.numel() * data.element_size()
sent_bytes = size_bytes * 2 * (world_size - 1)
total_duration = world_size * duration
bandwidth = sent_bytes / total_duration
```

![all-reduce 的有效带宽计算代码](assets/all-reduce-bandwidth-code.jpg)

*对应视频 `00:48:30--00:51:28`。画面展示发送字节数、总时长和带宽计算。*

这里需要明确口径：代码用 $1024^3$ 做单位换算却打印 `GB/s`，严格说更接近 `GiB/s`；而厂商标称带宽常用十进制 GB/s。比较数字前必须统一单位和“有效带宽/链路带宽”的定义。

### 如何读课程中的实测输出

课程执行结果中，四卡 all-reduce 大约落在 1.38–1.60 ms，对应代码口径约 366–426 GB/s；reduce-scatter 大约 2.39–2.61 ms，对应约 450–490 GB/s。

![collective 实测输出与带宽数字](assets/collective-benchmark-output.jpg)

*对应视频 `00:45:03--00:53:20`。画面包含 all-gather 的输出检查，以及 all-reduce、reduce-scatter 的计时结果。*

> [!WARNING]
> 不能据此断言“reduce-scatter 比 all-reduce 更慢”。两段 benchmark 的输入总大小不同：all-reduce 的本地张量约 400 MiB，而 reduce-scatter 的输入约为其 world-size 倍、输出才约 400 MiB。应在相同有效 payload、相同拓扑和相同计时口径下比较。

### 延迟模型的使用边界

$\alpha$–$\beta$ 模型适合建立数量级直觉，但真实 NCCL 还可能根据消息大小和拓扑选 ring、tree 或其他协议。拥塞、PCIe 亲和性、NUMA、GPU Direct、网络分层以及通信与计算竞争资源，都会让实测偏离简单公式。公式用于定位问题，不应代替 profiling。

### 本章小结

- CUDA 异步执行意味着 benchmark 必须 warmup、设备同步并对齐 rank。
- 毫秒数必须和消息大小、rank 数、算法及拓扑一起解释。
- 课程的带宽是特定算法口径，且二进制/十进制单位需要区分。
- 不同输入形状的 benchmark 不能只比较原始耗时。

---

## 数据并行：切 batch，同步 gradient

### 每张卡保存完整模型

数据并行沿 batch 维切分数据。每个 rank 拥有完整模型参数和优化器状态，只处理自己的 local batch，然后把梯度同步成全局梯度。

![数据并行：沿 batch 维切数据，复制完整网络](assets/data-parallelism.png)

*对应视频 `00:55:44--00:56:24`。橙色横线表示数据切分轴，不是前向的数据流方向。*

若 global batch 为 $B$，world size 为 $P$，且均匀切分，则每个 rank 处理 $B/P$ 个样本。令第 $r$ 个 rank 的 local loss 为 $L_r$：

$$
L=\frac{1}{P}\sum_{r=0}^{P-1}L_r,
\qquad
\nabla_\theta L=\frac{1}{P}\sum_{r=0}^{P-1}\nabla_\theta L_r.
$$

因此各 rank 先独立前向和反向，再对参数梯度做平均 all-reduce，就得到 global batch 的平均梯度。这里的同步对象是 **gradient，不是 parameter**；参数之所以保持一致，是因为大家从相同参数出发、使用相同的聚合梯度和更新规则。

### 最小 DDP 训练循环

```python
for x, y in local_batches:
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()

    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

    optimizer.step()
```

![DDP 的前向、反向与梯度 all-reduce](assets/ddp-gradient-sync.jpg)

*对应视频 `00:58:39--01:00:22`。画面强调普通训练与教学版 DDP 的唯一核心差异是梯度同步。*

> [!QUOTE]
> “唯一的差别是在 worker 之间同步梯度。”——讲者在 `00:59:19--00:59:27` 对教学实现的概括（意译）。

真实 `DistributedDataParallel` 不会等所有反向计算结束后再逐参数串行 all-reduce。它通常把梯度装入 bucket，并在某个 bucket 的梯度就绪后立刻发起通信，让前面层的梯度通信与后面层的反向计算重叠。

### 通信量和显存代价

若参数梯度总大小为 $|\theta|$ 字节，ring all-reduce 的每 rank 通信量近似为：

$$
V_{\text{DDP}}\approx 2\frac{P-1}{P}|\theta|.
$$

增加 local batch 可以提高计算/通信比，却不会减少每卡完整模型、梯度和优化器状态的复制。因此，DDP 能很好地扩展吞吐，但不能解决“完整模型状态单卡放不下”。这正是 ZeRO/FSDP 和模型并行的动机。

### batch 切分的边界条件

- local batch 必须有明确的采样策略，生产代码通常使用 distributed sampler，而不是让每个进程先读完整 batch 再手工切片；
- 若样本数不能整除 world size，需要处理不等长 local batch，否则简单平均各 rank 梯度会给样本不同权重；
- global batch 增大到超过 critical batch size 后，继续并行可能不再带来同等统计效率；
- 课程代码只有一个训练 step，因此没有暴露遗漏 `zero_grad()` 时梯度累积的问题，通用循环中必须显式清零或设为 `None`。

### 本章小结

- DDP 切分 batch、复制模型，并通过梯度 all-reduce 保持各副本一致。
- 同步的是梯度，不是每步平均参数。
- DDP 提升吞吐，却不降低每卡完整模型状态的显存占用。
- 生产 DDP 还需要 distributed sampler、梯度 bucket 与通信计算重叠。

---

## 张量并行：切 width，重组 activation

### 沿层宽切参数

张量并行进入每一层内部，把权重矩阵沿某个维度分到多张 GPU。课程用列切分线性层建立直觉。

![张量并行：沿 layer width 切分每一层](assets/tensor-parallelism.png)

*对应视频 `01:03:04--01:03:37`。橙色竖线表示权重沿宽度切分，不是新的网络层。*

令输入和权重形状为

$$
X\in\mathbb{R}^{B\times D},
\qquad
W\in\mathbb{R}^{D\times D}.
$$

把 $W$ 按输出列切成 $P$ 片：

$$
W=[W_0\;W_1\;\cdots\;W_{P-1}],
\qquad
W_r\in\mathbb{R}^{D\times D/P}.
$$

每个 rank 计算

$$
Y_r=XW_r\in\mathbb{R}^{B\times D/P},
$$

再拼接得到

$$
Y=[Y_0\;Y_1\;\cdots\;Y_{P-1}]=XW.
$$

于是每卡只保存约 $1/P$ 的该层权重并执行约 $1/P$ 的矩阵乘，但若下一步需要完整 $Y$，就必须 all-gather 局部激活。

### 前向中的 all-gather

```python
# local_weight: [D, D / P]
# x:            [B, D]
local_activation = x @ local_weight       # [B, D / P]

parts = [torch.empty_like(local_activation) for _ in range(P)]
dist.all_gather(parts, local_activation)
x = torch.cat(parts, dim=-1)              # [B, D]
```

![列张量并行的局部激活、all-gather 与拼接](assets/tensor-all-gather-code.jpg)

*对应视频 `01:05:27--01:07:03`。画面展示局部激活形状和前向重组边界。*

若激活有 $BD$ 个元素，每个元素 $s$ 字节，则在简单 all-gather 模型中，每个 rank 接收量约为

$$
V_{\text{TP,fwd}}\approx \frac{P-1}{P}BDs.
$$

这类通信可能每层发生，所以张量并行对链路的低延迟和高带宽都十分敏感。

### 反向为什么对应 reduce-scatter

反向传播时，每个分片产生对输入的局部贡献：

$$
\frac{\partial L}{\partial X}
=\sum_{r=0}^{P-1}
\frac{\partial L}{\partial Y_r}W_r^\top.
$$

局部贡献必须跨 rank 求和；若上游继续使用分片输入，则求和后还应保留相应分片，这正是 reduce-scatter 的语义。可以把“前向 all-gather、反向 reduce-scatter”理解为一对伴随通信。

> [!IMPORTANT]
> 普通本地张量上的 `.backward()` 不会凭空知道何时跨 rank 通信。只有使用带分布式语义的算子或自定义 autograd 边界，反向通信才会被正确插入。课程把 backward 留作练习，因此示例只证明了 forward 的形状与重组，不是完整训练实现。

### 为什么张量并行更侵入模型

DDP 可以把完整模型包起来；张量并行必须知道每一层如何切、何处保持分片、何处重组，以及不同算子的切分是否相容。优秀实现会交替使用列并行和行并行，让某些中间激活持续保持分片，避免每个子层都重建完整张量。

课程代码还通过反复调用同一个初始化函数保证分片能与基准模型对齐；该函数每次重设随机种子，使层权重重复。这是便于验算的教学技巧，不是常规模型初始化方式。

### 本章小结

- 张量并行沿 layer width 切权重，降低每卡参数存储和局部矩阵乘成本。
- 列切分会产生分片激活；需要完整输出时，前向使用 all-gather 重组。
- 反向的局部输入梯度需要求和，常对应 reduce-scatter。
- TP 进入模型内部且通信频繁，因此通常放在 NVLink/NVSwitch 高速域内。

---

## 流水线并行：切 depth，管理 bubble

### 沿模型深度分阶段

流水线并行把连续层放在不同 rank。上一个阶段算出激活后，通过点对点 `send` 交给下一个阶段；后者 `recv` 后继续前向。

![流水线并行：沿网络深度把连续层分给不同 rank](assets/pipeline-parallelism.png)

*对应视频 `01:09:42--01:10:08`。橙色横线位于层之间，表示 stage 边界。*

两阶段前向的教学骨架如下：

```python
if rank > 0:
    x = torch.empty(activation_shape, device=device)
    dist.recv(x, src=rank - 1)

x = local_layers(x)

if rank < world_size - 1:
    dist.send(x, dst=rank + 1)
```

这种写法清楚展示 stage 边界，却不是完整训练系统：它只有前向和阻塞式 send/recv，没有 loss、反向、权重更新、micro-batch 调度或通信重叠。生产实现还要管理激活生命周期，并保证前向和反向消息不会死锁。

### 为什么完整 batch 会产生空泡

若一次把完整 batch 依次送过 $P$ 个 stage，后面的 stage 在等待输入时空闲，前面的 stage 完成后也会闲置。把 batch 切成 $M$ 个 micro-batch 后，不同 stage 可以同时处理不同 micro-batch。

![两阶段流水线的 micro-batch 与理想空泡利用率](assets/pipeline-bubble.svg)

*对应视频 `01:12:28--01:13:32`。此图是依据课程讲解绘制的理想等时前向模型，不是视频原帧。*

在所有 stage 计算时间相等、只考虑前向、忽略通信的理想模型中，总时间槽数是 $M+P-1$，每个 stage 的有效计算槽数为 $M$，因此

$$
\eta_{\text{ideal}}=\frac{M}{M+P-1},
\qquad
f_{\text{bubble}}=\frac{P-1}{M+P-1}.
$$

例如 $P=2$：$M=1$ 时利用率为 $1/2$；$M=4$ 时为 $4/5$。micro-batch 越多，填充与排空开销所占比例越小。

### micro-batch 不是免费午餐

- micro-batch 太小会降低矩阵乘效率；
- stage 计算量不均时，最慢 stage 决定吞吐；
- 完整训练还要安排反向，1F1B 等调度会改变激活存储和空泡形态；
- send/recv 自身需要时间，并可能与计算争用资源；
- micro-batch 数量、gradient accumulation 与 optimizer step 的语义必须对齐。

因此上面的公式是心智模型，而不是任意 pipeline 实现的性能保证。

### 通信与计算重叠

串行执行时，一步近似花费

$$
T_{\text{serial}}=T_{\text{compute}}+T_{\text{comm}}.
$$

若依赖关系和硬件资源允许理想重叠，则下界更接近

$$
T_{\text{overlap}}\approx
\max(T_{\text{compute}},T_{\text{comm}}).
$$

在 pipeline 中，不同 stage 同时处理不同 micro-batch；在 DDP 中，已就绪梯度 bucket 的 all-reduce 可以与更早层的反向计算并行。两者都依赖足够的独立工作和正确调度。

> [!QUOTE]
> “计算与通信应该同时发生。”——讲者在 `01:14:02--01:14:19` 强调的系统设计目标（意译）。

### 本章小结

- 流水线并行沿 depth 切连续层，只在 stage 边界传递激活。
- micro-batch 让多个 stage 同时工作，减小填充与排空造成的 bubble。
- $M/(M+P-1)$ 只适用于等时、理想前向模型，不能直接代表完整训练利用率。
- 完整 PP 还需要反向调度、激活管理、非阻塞通信和负载均衡。

---

## 从基本策略到真实系统

### 三种主策略的统一比较

| 策略 | 切分轴 | 每卡保留什么 | 主要通信 | 首要收益 | 典型约束 |
|---|---|---|---|---|---|
| 数据并行 | batch | 完整模型，部分样本 | 梯度 all-reduce | 提升吞吐 | 不降低完整模型状态显存 |
| 张量并行 | layer width | 每层的一部分权重 | 高频 all-gather / reduce-scatter | 模型层单卡放不下 | 需要高速低延迟互联 |
| 流水线并行 | model depth | 一组连续层 | stage 间 send/recv | 跨设备放置深模型 | bubble、调度和负载均衡 |

它们并不互斥。一个大规模训练任务可能在每个节点内部做 tensor parallel，在节点组之间做 pipeline parallel，再把整个模型并行副本作为 data-parallel group 复制到更多节点。

### ZeRO/FSDP：把 DDP 的复制状态再切开

传统 DDP 在每卡复制参数、梯度和优化器状态。ZeRO/FSDP 逐步切分这些状态：需要计算某一层时 all-gather 必要参数，反向后用 reduce-scatter 聚合并保留梯度分片。这样以额外通信换取更低的每卡显存。

这也是本课先讲 collective 的原因：不同“并行名词”最终仍能还原为数据布局与通信原语的组合。

### sequence、expert 与混合并行

- **sequence parallelism** 沿序列维切分激活或某些算子，常与 tensor parallel 配合降低激活内存；
- **expert parallelism** 把不同专家放在不同设备，常通过 all-to-all 路由 token；
- **多维并行** 同时沿数据、张量、流水线、序列或专家维切分，需要明确各自的 process group，避免让高频通信跨越慢链路。

课程用简单 MLP，而不是完整 Transformer，是为了把 batch、width、depth 三个正交切分轴暴露出来。换成 Transformer 后，attention、MLP、embedding 等模块的细节更多，但底层 collective 和布局推理仍然适用。

### 并行策略应服从硬件层级

一个实用的放置顺序是：

1. 把最频繁、最延迟敏感的 TP 通信限制在 NVLink/NVSwitch 高速域；
2. 用 PP 跨越相对慢的节点边界，因为它只在 stage 边界传激活；
3. 用 DP 扩展副本数量，并通过 bucket 尽量把梯度通信与反向重叠；
4. 根据显存压力决定是否进一步使用 FSDP/ZeRO、activation checkpointing 或 sequence parallelism。

这不是固定配方。最终配置取决于模型形状、batch size、序列长度、集群拓扑、网络争用和目标时间预算。

### critical batch size 与统计效率

数据并行可以不断扩大 global batch，但优化收益并不会无限线性增长。超过问题相关的 critical batch size 后，更多样本带来的梯度方差降低可能已经很有限；即使硬件吞吐继续提高，达到相同验证损失所需的样本或步数未必等比例减少。

critical batch size 不是一个跨模型恒定的数字。它会随模型、数据、训练阶段、优化器和目标指标变化，不能只根据 GPU 数量提前写死。

### JAX/TPU 的另一种抽象

课程最后提到 JAX/TPU 的 sharding 风格：用户可以声明张量如何映射到设备网格，让编译器推导并插入许多通信。这能提高抽象层次，但并没有消除物理代价；布局若让高频通信跨慢链路，编译器也无法把带宽限制变没。

### 本章小结

- DP、TP、PP 是正交切分轴，可以嵌套形成多维并行。
- ZeRO/FSDP 通过切分 DDP 原本复制的状态，用通信换显存。
- sequence 与 expert parallelism 仍可还原为张量布局和 collective。
- 软件抽象可以自动插入通信，但不能取消硬件拓扑和通信成本。

---

## 总结与延伸

### 一张图回看全课

![讲者的全课收束：并行方法与 memory/recompute/communicate](assets/lecture-summary.jpg)

*对应视频 `01:19:14--01:20:47`。画面完整列出 DDP、FSDP/ZeRO、TP、PP 以及存储、重算、通信之间的权衡。*

本课从 rank 与 collective 出发，依次建立了硬件、编程和算法三层视角：

1. **通信语言**：broadcast、scatter、gather、reduce 负责基础模式；all-gather、reduce-scatter、all-reduce 和 all-to-all 是现代训练的主力。
2. **系统现实**：NCCL 把抽象 collective 映射到 NVLink、NVSwitch、InfiniBand 或 Ethernet/RoCE；拓扑决定实际成本。
3. **切分算法**：DP 切 batch，TP 切 width，PP 切 depth；ZeRO/FSDP 再切分原本复制的模型状态。
4. **性能方法**：用 $\alpha$–$\beta$ 模型估算，用同步正确的 benchmark 验证，再通过 profiler 检查通信是否真正被计算覆盖。

### 最后保留的三角权衡

训练中暂时需要一个张量时，系统通常只有三类选择：

- **存储**：把它留在本地显存，省计算与通信，但消耗容量；
- **重算**：需要时重新计算，用 FLOPs 换显存；
- **通信**：把它放在别的设备或保持分片，需要时交换，用网络换显存。

硬件会继续变快，模型也会继续变大，因此这一权衡不会消失。可靠的并行设计不是寻找一种永远最优的技术，而是让最频繁的数据移动停留在最快的层级，把不可避免的通信隐藏在有用计算后面，并用测量校正公式直觉。

### 建议的后续练习

1. 用四个小向量手算八种 collective 的输入输出，再验证 all-reduce 的两阶段分解。
2. 修改课程 benchmark：固定相同 payload，重复多次，分别报告中位数、P95、算法带宽与总线带宽。
3. 为教学版 tensor parallel 实现一个带 reduce-scatter 的自定义 autograd 边界，并与单卡梯度逐元素比较。
4. 为 pipeline 示例加入 micro-batch 和反向调度，画出 $P=4$ 时不同 $M$ 的时序图。
5. 给定一套具体节点拓扑，设计 TP/PP/DP process group，并说明为什么每种通信被放在对应链路上。

### 本章小结

- 并行训练的核心不是卡数，而是张量布局、通信边界与硬件层级的匹配。
- DP、TP、PP 分别沿 batch、width、depth 切分；ZeRO/FSDP 用更细粒度状态切分降低显存。
- 任何方案都在 memory、recompute、communicate 之间交换成本。
- 公式负责建立直觉，正确同步的 benchmark 和 profiler 负责验证现实。
