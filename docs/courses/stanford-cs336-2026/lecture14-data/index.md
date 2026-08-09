# Stanford CS336 2026 Lecture 14：Data

![课程封面](assets/cover.jpg)

- **视频标题**：Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 14: Data
- **主讲 / 频道**：Stanford Online
- **视频链接**：<https://www.youtube.com/watch?v=5sxHosTLPF8>
- **时长**：01:24:46
- **材料范围**：人工英文字幕、1080p 课程视频与官方 `lecture_14.py`
- **课程定位**：承接 Data I，从数据转换、过滤、去重与混合，一直讲到后训练中的合成数据

> [!IMPORTANT]
> 这堂课最值得带走的不是某个固定阈值，而是一套贯穿全流程的思维：先把目标行为写成可检验的数据标准，再用便宜且可扩展的方法把它外推到海量数据；同时始终检查训练规模、重复轮数和代理目标是否改变了问题本身。

## 1. 从在线资源到可训练文本：Transformation

### 1.1 Data I 到 Data II：从“去哪里找”转向“如何加工”

上一讲的数据链路是：在线服务（例如 GitHub）先产生原始内容，经 dump 或 crawl 形成可下载快照，再加工为 The Stack 一类训练集。它同时涉及服务条款、版权、许可和 fair use。本讲把焦点向下游推进：即使合法取得了网页、论文和代码仓库，它们也还不是模型能直接消费的 token 序列。

课程给出的主流程是：

```text
HTML / PDF / repositories
        ↓ transformation
linearized text
        ↓ filtering
task-relevant, higher-quality subset
        ↓ deduplication
less redundant corpus
        ↓ mixing
training distribution over sources
        ↓
pretraining / mid-training / post-training
```

### 1.2 HTML 到文本是有损的线性化

HTML 页面包含导航、广告、页眉页脚、侧栏、正文、图片和表格。训练语言模型时通常要去掉 boilerplate、保留主内容，并把二维或层级布局压成一维序列。这个过程天然有损：简单表格还能写成 Markdown，复杂的嵌套表格、图片与文字的空间关系、脚注和图注关联则很难完全保留。

常用的规则工具包括 Trafilatura、Resiliparse、jusText 和 lynx。规则工具快，适合运行在海量网页上；但“快”不等于“等价”。官方课件引用 DCLM 对比，说明文本提取器的选择会产生可测的下游差异。

![不同 HTML 文本提取方式的对比](assets/dclm-wet.png)
*图 1：Resiliparse、Trafilatura 与 Common Crawl WET 文本并不等价；这里不擅自扩展 CORE/EXTENDED 指标定义，只保留“转换准确率会传导至模型质量”的结论。（字幕区间：00:03:57--00:04:20）*

> [!WARNING]
> “去掉导航、广告”看似客观，其实已经注入了任务假设。若目标是学习网页结构，被删掉的导航可能反而是信号。无模型的规则过滤也不是“无偏”，它只是把偏置放进了规则与原始分布。

### 1.3 PDF 比 HTML 更难：识别字符不等于恢复语义结构

PDF 更接近一组“把字画在某个坐标”的指令，而不是带有 `<h1>`、`<p>` 等语义标签的文档树。Common Crawl 还可能截断较大的 PDF；扫描件没有字符层，只能先用 OCR 或视觉语言模型识别。FinePDFs 的方案因此包括重新抓取截断文件、使用 RolmOCR/Docling 类工具、清理与过滤。

![PDF 源码与视觉布局之间的鸿沟](assets/finepdfs-layout.jpg)
*图 2：PDF content stream 主要描述位置和绘制操作；标题、双栏阅读顺序、图片与图注关系往往要从几何布局推断。该帧选择了完整揭示状态。（字幕区间：00:04:20--00:06:42）*

识别出页面上的字只是第一步。真正困难的是确定双栏阅读顺序、标题层级、表格单元格、图片与图注配对。数据转换因此不是简单的文件格式转换，而是一个同时权衡速度、结构恢复和信息损失的建模步骤。

### 本章小结

- 原始数据通常是 HTML、PDF 或仓库，而不是纯文本。
- 转换把视觉与层级结构线性化，必然丢失信息。
- 解析工具的错误率会在数十亿页面上累积，并传导到下游模型。
- 取得文本只意味着进入下一阶段，并不代表数据已经可以训练。

## 2. Filtering：用少量目标数据筛选海量原始池

### 2.1 统一抽象：从目标集合 (T) 外推到 (T')

设 (T) 是少量、能代表目标能力的优质数据，(R) 是海量原始池。过滤要从 (R) 中找到与 (T) 相似的新子集 (T')：

$$
T'\subseteq R,\qquad T'\sim T.
$$

- (T)：定义“好数据长什么样”的目标集合。
- (R)：需要被筛选的原始数据。
- (T')：从 (R) 中选出的新样本；它应泛化 (T)，而不是简单复制 (T)。

![目标数据与原始数据的过滤关系](assets/raw-target-schema.png)
*图 3：目标数据 (T) 不必包含在原始池中；过滤器学习其特征，再从 (R) 内找出新的 (T')。（字幕区间：00:06:56--00:07:39）*

工程上有两个同时存在的要求：过滤器要从少量目标数据泛化；还必须极快，因为原始池可能达到 (10^{14}) token。通用流程是先根据 (R,T) 得到评分函数，再按分数保留 (R) 中的样本。

### 2.2 两类评分：生成式与判别式

生成式方法在目标数据上训练 KenLM 一类语言模型，用目标分布的似然评分：

$$
\operatorname{score}(x)=p_T(x).
$$

- (x)：待判断的文档。
- (p_T(x))：目标数据模型认为 (x) 出现的概率。
- (T)：目标语料分布。

判别式方法则训练 fastText 等轻量分类器，直接预测样本是否属于目标集合：

$$
\operatorname{score}(x)=p(T\mid x).
$$

- (p(T\mid x))：给定文档 (x) 时，它属于目标分布的概率。
- `fastText`：常见的快速词特征线性分类器，适合扫描超大语料。

最直接的硬阈值规则是：

$$
T'=\{x\in R:\operatorname{score}(x)\ge \tau\}.
$$

- (	au)：质量或相关性阈值。
- (T')：通过阈值的原始样本子集。

也可以随机保留，使分数越高的样本被选中概率越大。课件给出 GPT-3 风格的示意代码：

```python
def keep_document(score: float) -> bool:
    return np.random.pareto(9) > 1 - score
```

这里 `score` 越高，右侧 `1-score` 越小，文档越容易保留。应把它理解为课程中的随机保留示意，而不是未经论文附录核对就认定为唯一正式公式。

### 2.3 “质量”取决于目标任务

过滤框架可以实例化为：

- 语言识别：英语相对于其他语言；
- 通用质量：参考书籍、Wikipedia 或人工偏好筛网页；
- 领域选择：数学、代码、科学论文；
- 安全过滤：non-toxic 相对于 toxic 内容。

> [!IMPORTANT]
> 课程明确强调：不存在脱离任务的普适“高质量”。一本数学教材对数学推理很有价值，对训练自然对话分布却未必最合适。目标集合 (T) 实际上定义了模型将被鼓励获得什么能力。

### 本章小结

- Filtering 可以统一写成“用目标集合 (T) 从原始池 (R) 中寻找相似的 (T')”。
- KenLM 估计 (p_T(x))，fastText 估计 (p(T\mid x))。
- 过滤器必须同时具备泛化能力和极高吞吐。
- 目标数据定义了质量，因此过滤不可避免地带有任务偏置。

## 3. 过滤案例与尺度效应

### 3.1 语言识别与数学语料

fastText 的语言识别模型支持 176 种语言，训练源包含 Wikipedia、Tatoeba 和 SETimes。Dolma 的一个规则是保留：

$$
p(\text{English}\mid x)\ge 0.5.
$$

- (x)：页面文本。
- (p(\text{English}\mid x))：分类器预测为英语的概率。
- (0.5)：课程给出的 Dolma 页面阈值。

数学语料的例子 OpenWebMath（课件中称 OpenMathText）采用级联管线：先用 LaTeX 命令等规则粗筛，再用在 ProofPile 上训练的 KenLM，保留困惑度小于 15000 的文本；随后用 fastText 判断数学写作。有数学线索时阈值为 0.17，没有线索时要求 0.8。最终得到 14.7B token；用它训练的 1.4B 模型优于用约二十倍未定向数据训练的模型。

这个结果不是说“小数据永远更好”，而是说明面向目标能力的数据可以显著提高单位 token 的学习价值。

### 3.2 GPT-3、LLaMA 与 phi-1：如何构造目标集合

GPT-3 用 Wikipedia、WebText2、Books1、Books2 样本作正例，用 Common Crawl 作负例，基于词特征训练线性分类器。LLaMA/RedPajama 的正例来自被 Wikipedia 引用的外部页面，而不是 Wikipedia 页面本身；负例仍来自 Common Crawl。

phi-1 更清楚地展示了“昂贵 teacher 标注少量样本，便宜 classifier 放大到完整数据池”的方式：

```python
R = "Python subset of the Stack"
prompt = (
    "determine its educational value for a student "
    "whose goal is to learn basic coding concepts"
)
T = "GPT-4 labels a 100K subset of R as positive examples"
```

接着从预训练 CodeGen 模型取得 embedding，在 (T) 上训练随机森林，再扫过完整 (R)。

![phi-1 的教师标注与便宜分类器扩展](assets/phi1-filtering-results.jpg)
*图 4：课程展示的完整 phi-1 管线与 HumanEval 结果；过滤后的 1.3B 模型在 36K steps 达到 17.68%，原始 Python Stack 训练在 96K steps 为 12.19%。（字幕区间：00:15:05--00:16:29）*

> [!NOTE]
> 这个范式与今天的合成数据流水线一脉相承：强模型不必亲自处理所有样本，只需产生足够可靠的监督，再由轻量模型完成规模化筛选。

### 3.3 毒性过滤

Dolma 使用 Jigsaw Toxic Comments。原始标注来自 Wikipedia talk pages，包括 `toxic`、`severe_toxic`、`obscene`、`threat`、`insult`、`identity_hate`。机制与质量过滤完全一致：先得到正负例，再训练便宜分类器扫描 Common Crawl。

需要谨慎的是，安全类别和社会语言现象经常存在上下文依赖、方言偏差和误伤。模型过滤可以减少不希望的内容，但它并不自动解决价值定义与分布偏差。

### 3.4 没有脱离训练预算的唯一最佳阈值

过滤越严格，留下的数据通常越少、平均质量越高；训练初期 loss 可能下降得更快。但有限数据被反复遍历后，收益会饱和甚至过拟合。放宽阈值得到的大数据池早期较差，却能支持更长训练。

![过滤效果随训练 token 数而改变](assets/data-filtering-scale.png)
*图 5：157M 模型在 100 个 WARC 上的曲线。虚线是一轮数据的边界；严格过滤数据早期表现好，但更快耗尽并出现重复训练效应。（字幕区间：00:17:25--00:20:14）*

因此应把过滤阈值、训练 token budget、可用数据规模和 epoch 数联合选择。更长训练中放宽过滤，不意味着低质量数据本身优于高质量数据，而是现实里“无限高质量数据”并不存在。

学生追问置信区间时，讲者承认完整重复训练很昂贵，论文常缺少足够的重复实验；其经验是预训练小规模趋势通常较稳定。这是工程经验，不应替代严格统计结论。

### 本章小结

- 语言、数学、通用质量与毒性过滤只是同一框架的不同目标定义。
- phi-1 展示了强模型标注、小模型扩展的高性价比模式。
- 严格过滤提升早期单位 token 价值，但会更早遇到数据重复与过拟合。
- 最佳阈值随训练预算改变，不能作为与规模无关的常数。

## 4. 去重：从精确匹配到 MinHash

### 4.1 为什么重复会浪费训练

精确重复包括镜像站与 GitHub fork；近似重复包括许可证、服务条款、统一页眉页脚、轻微排版差异，以及只替换少数实体的模板文章。课程展示 C4 审计中的极端案例：同一段商品描述重复 61,036 次。

![C4 中重复 61036 次的商品描述](assets/c4-repeated-description.jpg)
*图 6：模板化网页让看似不同的页面携带几乎相同的信息，提醒我们必须直接审计样本，而不能只看汇总指标。（字幕区间：00:25:45--00:26:17）*

去重的直接收益是减少无信息增量的 token，让相同 FLOPs 覆盖更多新内容；附带收益是降低逐字记忆、版权与隐私风险。与之相邻但更重要的是 decontamination：评测集不能泄漏到训练集。

### 4.2 设计空间与可扩展性

完整去重方案必须回答三个问题：

1. item 是句子、段落、固定 span 还是文档？
2. match 是精确相等、共享子项，还是共享子项比例？
3. action 是全部删除，还是每组保留一个？

过滤可逐样本并行；去重涉及样本之间的关系。对 (N) 个文档做全量两两比较是 (O(N^2))，不可扩展。哈希的价值是把内容映射到较小键空间，让相同或相似项目在近线性扫描中进入同一候选桶。

### 4.3 精确去重

课件用 MurmurHash 演示 MapReduce 风格的分组：

```python
items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]

hash_items = itertools.groupby(
    sorted(items, key=mmh3.hash),
    key=mmh3.hash,
)

deduped_items = [next(group) for h, group in hash_items]
```

- `items`：待去重字符串。
- `mmh3.hash`：快速但非密码学抗碰撞的 MurmurHash。
- `hash_items`：按哈希排序并分组。
- `deduped_items`：每组只取一个元素。

![精确去重代码与 C4 三句 span 规则](assets/exact-dedup-c4-warning.jpg)
*图 7：C4 以三句 span 为 item 做精确匹配；最终完整揭示的警告指出，从文档中间删除三句话可能破坏上下文连贯性。（字幕区间：00:29:38--00:31:21）*

> [!WARNING]
> 玩具代码只按哈希值分组。生产实现应在哈希桶内再次比较原文，以免哈希碰撞误删不同内容。精确去重语义清楚，却无法处理只差少数 token 的近重复。

### 4.4 Jaccard：为近重复定义相似度

对集合 (A,B)，Jaccard 相似度是交集占并集的比例：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

- (A,B)：文档转换出的 token、word 或 character shingles 集合。
- (|A\cap B|)：两个集合共享的元素数。
- (|A\cup B|)：至少出现在一个集合中的元素数。

若 (A=\{1,2,3,4\})，(B=\{1,2,3,5\})，则交集大小为 3，并集大小为 5，所以 (J=0.6)。可以把 (J(A,B)\ge\tau) 定义为近重复；阈值 (	au) 取决于任务，不是固定标准。

![Jaccard 的 0.6 玩具例子](assets/jaccard-example.jpg)
*图 8：画面同时显示集合、实现代码、运行结果 `jaccard=0.6` 与“在线性时间内找近重复”的算法挑战。（字幕区间：00:31:26--00:32:38）*

### 4.5 MinHash：把相似度变成碰撞概率

MinHash 构造随机哈希，使两个集合签名相同的概率等于 Jaccard：

$$
\Pr[h(A)=h(B)]=J(A,B).
$$

- (h)：由随机排列或带 seed 的哈希近似出的 MinHash。
- (h(A),h(B))：各集合元素哈希值中的最小值。
- (J(A,B))：两集合的真实 Jaccard 相似度。

```python
def minhash(S: set[str], seed: int):
    return min(mmh3.hash(x, seed) for x in S)
```

![MinHash 定义与特征矩阵](assets/minhash-definition.jpg)
*图 9：普通哈希希望避免碰撞，MinHash 则让碰撞概率受集合相似度控制；画面展示定义、代码和集合特征矩阵。（字幕区间：00:32:46--00:35:00）*

直觉证明是：随机哈希相当于随机排列全集元素。若最先出现的是交集元素 1、2、3，两个集合的最小元素相同；若是仅属于一侧的 4 或 5，则不同。五个可能的“第一元素”中三个来自交集，因此碰撞概率正好为 (3/5)。

单个 MinHash 碰撞只是高方差的 Bernoulli 事件，不能直接断言相似度已超过 0.99。需要多个哈希，并用结构化组合把概率曲线变成接近阈值的判别器。

### 本章小结

- 去重首先是算力分配问题，同时能降低记忆与泄漏风险。
- item、match 与 action 的选择决定了去重语义。
- 精确哈希分组可扩展，但处理不了轻微改写。
- Jaccard 定义集合相似度；MinHash 让碰撞概率等于 Jaccard，为近重复候选生成奠定基础。

## 5. LSH：用 AND–OR 结构锐化概率阈值

### 5.1 Banding 结构

将 (n) 个 MinHash 拆成 (b) 个 band，每个 band 有 (r) 个哈希，满足 (n=br)。两个文档成为候选，当且仅当“至少一个 band 内的所有 (r) 个哈希都相同”。这就是 band 内 AND、band 间 OR。

![LSH 的 band 内 AND 与 band 间 OR](assets/lsh-bands.jpg)
*图 10：12 个哈希被分为 3 个 band，每 band 4 个；至少一个 band 全匹配即可碰撞。（字幕区间：00:39:07--00:41:03）*

设 Jaccard 相似度为 (s)。一个 band 全部匹配的概率是 (s^r)，所有 band 都不匹配的概率是 ((1-s^r)^b)，所以候选碰撞概率为：

$$
P_{\mathrm{collision}}(s)=1-(1-s^r)^b.
$$

- (s)：两文档的 Jaccard 相似度。
- (r)：每个 band 的哈希数；控制 band 内 AND 的严格程度。
- (b)：band 数；控制 OR 的尝试次数。
- (P_{\mathrm{collision}})：两文档进入同一候选集合的概率。

```python
def get_prob_collision(sim, b, r):
    prob_match = sim ** r
    prob_collision = 1 - (1 - prob_match) ** b
    return prob_collision
```

![LSH 碰撞概率的代码与数值例子](assets/lsh-collision-probability.jpg)
*图 11：当 (s=0.8,b=5,r=10) 时，碰撞概率约 0.4333；画面采用完全显示代码和输出的状态。（字幕区间：00:41:03--00:42:53）*

### 5.2 (r) 与 (b) 如何移动曲线

增加 (r) 会让一个 band 更难全匹配，曲线向右移动并变陡；增加 (b) 会提供更多 OR 机会，曲线向左移动，更容易匹配。

![增加 r 后的碰撞概率](assets/lsh-increase-r.jpg)
*图 12：固定 (b=10)，把 (r) 增至 20 后，中低相似度候选被显著抑制；0.9 相似度的碰撞概率约 0.7264。（字幕区间：00:45:09--00:45:51）*

![增加 b 后的碰撞概率](assets/lsh-increase-b.jpg)
*图 13：保持 (r=20)，把 (b) 增至 20 后，0.9 相似度候选的碰撞概率回升到约 0.9252。（字幕区间：00:45:51--00:46:23）*

参数表可以压缩为：

| (b) | (r) | (s=0.7) | (s=0.8) | (s=0.9) | (s=0.95) |
|---:|---:|---:|---:|---:|---:|
| 10 | 10 | 0.2491 | 0.6789 | 0.9863 | 0.9999 |
| 10 | 20 | 0.0080 | 0.1095 | 0.7264 | 0.9882 |
| 20 | 20 | 0.0158 | 0.2070 | 0.9252 | 0.9999 |

LSH 是概率候选生成器，不是严格判决器。低于阈值仍可能碰撞，高于阈值也并非 100% 召回；候选生成后可以计算真实 Jaccard 再过滤。更尖锐的曲线还需要更多哈希计算、存储和候选索引。

### 5.3 真实配置与相变位置

课件引用真实去重配置 (n=9000,b=20,r=450)，并用下式近似相变位置：

$$
s_\star=\left(\frac{1}{b}\right)^{1/r}.
$$

- (s_\star)：近似相变位置。
- (b)：band 数。
- (r)：每 band 的哈希数。

![真实 LSH 配置与近似阈值](assets/lsh-real-threshold.jpg)
*图 14：课程给出的 (n=9000,b=20,r=450) 与阈值代码；代入得 (s_\star\approx0.993365)。（字幕区间：00:46:42--00:48:06）*

在这个位置 (s_\star^r=1/b)，总体碰撞概率为：

$$
1-\left(1-\frac1b\right)^b\rightarrow1-\frac1e\approx0.632.
$$

- (1/b)：一个固定 band 匹配的概率。
- (b)：band 数。
- 极限值说明这不是严格的 0.5 判别点；当 (b=20) 时约为 0.6415。

最后，去重应跨来源执行。若 Wikipedia、书籍和网页数据集分别内部去重，再直接合并，跨来源收录的同一内容仍会重复。

### 本章小结

- LSH 用 band 内 AND、band 间 OR 把平滑碰撞概率锐化为 S 形曲线。
- (r) 越大越严格，(b) 越大越容易碰撞。
- LSH 只生成候选，仍需理解 false positive、false negative 与复核成本。
- 数据集内部去重不够，最终语料还要做跨来源全局去重和评测集去污染。

## 6. Data Mixing：从经验配比到受约束的小规模优化

### 6.1 混合分布不是“质量排序”

预训练通常同时使用网页、Wikipedia、代码、论文、书籍和多语言数据。Marin 的 token viewer 展示了来源规模差异：通用网页可达数万亿 token，代码、数学和专业语料小得多。

![Marin 数据源的 token 规模](assets/marin-token-viewer.png)
*图 15：不同颜色表示 specialized、web、multilingual、math 与 code；各来源规模跨越很大范围。（字幕区间：00:49:49--00:50:29）*

The Pile 已明确为不同来源指定 weight 与 epoch 数，说明“混合”不是把文件拼在一起，而是定义训练时的采样分布。

![The Pile 的来源权重与 epoch](assets/the-pile-mixture.jpg)
*图 16：Raw Size、Weight、Epochs 与 Effective Size 同时展示了原始规模和训练暴露量的区别。（字幕区间：00:50:34--00:50:58）*

设来源集合为 `Wikipedia`、`CC`、`GitHub`，一个可能的混合是：

```python
p = {"Wikipedia": 0.3, "CC": 0.5, "GitHub": 0.2}
```

课程给出三个基线：凭经验手调（vibes）、均匀采样 (p(s)\propto1)、按来源规模成比例 (p(s)\propto N_s)。

![数据混合的三类基线](assets/mixing-baselines.jpg)
*图 17：仅按直觉、均匀或数据量比例都不足以同时处理能力多样性和有限小来源重复问题。（字幕区间：00:51:57--00:53:39）*

文学、代码和论文并不存在统一的质量序。若全压到某一种“高质量”来源，模型会失去能力多样性；若给小而优质的数据过高权重，又只能重复遍历。

### 6.2 把权重换算为 epoch 数

令总训练 token 暴露量为 (T)，来源 (s) 有 (N_s) 个可用 token，采样权重为 (p_s)，则平均重复轮数是：

$$
E_s=\frac{p_sT}{N_s}.
$$

- (E_s)：来源 (s) 被完整遍历的平均次数。
- (p_s)：训练预算分给来源 (s) 的比例。
- (T)：训练处理的总 token 暴露量，不是 unique token 数。
- (N_s)：来源 (s) 的可用 token 数。

课程的算例中，低质量来源有 10T token，高质量来源只有 10B，1T-token 训练却给两者各 50% 权重：

$$
E_{\mathrm{low}}=0.05,\qquad E_{\mathrm{high}}=50.
$$

![50 对 0.05 个 epoch 的陷阱](assets/mixture-epoching-overfit.jpg)
*图 18：同为 50% 权重，来源容量相差千倍，实际重复轮数也相差千倍。完整代码和两个输出都在画面中。（字幕区间：00:53:45--00:56:04）*

> [!QUOTE]
> **讲者（00:56:40--00:56:52）**：“Best case is wasting compute, and worst case, you're overfitting.”

混合在 batch 中通常按 sequence 采样来源：一条 sequence 来自一个组件，不是逐 token 混合；同一个 batch 可包含多个来源，以降低梯度方差。

### 6.3 UniMax：给每个来源加 epoch 上限

温度式平滑可写为：

$$
p_s\propto N_s^\alpha,\qquad \alpha\in[0,1].
$$

- (alpha=0)：接近均匀采样。
- (alpha=1)：按来源 token 数成比例。
- 中间值：在均匀与比例之间折中。

UniMax 尽量均匀采样，但给每个来源设置最大 (C) 个 epoch。量纲完整的约束是：

$$
\frac{p_sT}{N_s}\le C,
\qquad\text{等价于}\qquad
p_sT\le C N_s.
$$

- (C)：任一来源允许的最大 epoch 数。
- (p_sT)：来源获得的 token 暴露预算。
- (CN_s)：在最多 (C) 轮时可提供的 token 数。

![UniMax 的最大重复轮数约束](assets/unimax-cap.jpg)
*图 19：画面同时显示 50-epoch 陷阱、温度式基线与 UniMax hard cap；课程源码的简写省略了 (N_s)，正文采用量纲完整公式。（字幕区间：00:58:25--01:00:00）*

### 6.4 回归式数据混合

RegMix 类方法把昂贵的大模型配比搜索转成代理优化：

1. 从概率单纯形采样许多候选 mixture，例如 Dirichlet 分布；
2. 每个 mixture 训练一个小代理模型；
3. 收集验证 loss 或下游指标；
4. 拟合回归器 (hat L=f_\theta(p))，寻找预测最优配比；
5. 用该配比训练大模型。

![回归式数据混合的四步流程](assets/regmix.png)
*图 20：先训练小模型群，再拟合 mixture 到目标值的代理函数，搜索最优 mixture，最后用于大规模训练。（字幕区间：01:00:37--01:02:06）*

其数学抽象是：

$$
\hat L=f_\theta(p),\qquad
p^\star=\arg\min_{p\in\Delta^{m-1}}f_\theta(p).
$$

- (p)：(m) 个 domain 的混合权重向量。
- (Delta^{m-1})：概率单纯形，要求权重非负且和为 1。
- (f_\theta)：线性、LightGBM、power law 或 Gaussian Process 等代理模型。
- (hat L)：预测的 loss 或下游目标。
- (p^\star)：预测最优 mixture。

![不同回归式 mixing 方法的设计空间](assets/data-mixing-methods.png)
*图 21：方法在代理尺寸、swarm 构造、回归器、目标粒度、优化器和重复约束上均不同；OlmixBase 一列显式加入 data repetition constraint。（字幕区间：01:03:54--01:04:50）*

> [!WARNING]
> 代理方法包含两个“Hope”：回归器在优化器找到的极值点附近仍准确；小规模最优配比能迁移到大规模。随机测试点预测准确，不保证 argmin 所在的分布边缘也准确。若目标主要是代码评测，优化器还可能牺牲未纳入目标的文学等能力。

### 6.5 Simulated epoching：让小规模像大规模

小实验 token 少，稀缺来源还没被重复，因此可能偏好 `{"low":0.1,"high":0.9}`；同一配比搬到 1T-token 大训练后，高质量小源却会被反复遍历。Simulated epoching 让小实验提前承受相同重复压力：

$$
\rho=\frac{T_{\mathrm{small}}}{T_{\mathrm{large}}},
\qquad N'_s=\rho N_s.
$$

- (T_{\mathrm{small}})：代理实验 token 预算。
- (T_{\mathrm{large}})：目标训练 token 预算。
- (ho)：规模比率。
- (N'_s)：小实验中为来源 (s) 模拟出的可用 token 数。

于是：

$$
\frac{p_sT_{\mathrm{small}}}{N'_s}
=\frac{p_sT_{\mathrm{large}}}{N_s},
$$

- 左侧：小实验中每个来源的重复轮数。
- 右侧：大训练在同一 mixture 下的重复轮数。
- 两者相等，说明模拟保留了 epoching 压力。

```python
small_run_tokens = 10 * 10**9
large_run_tokens = 1 * 10**12
ratio = small_run_tokens / large_run_tokens  # 0.01

downsampled_source_token_counts = {
    source: count * ratio
    for source, count in source_token_counts.items()
}
```

![Simulated epoching 的比例缩小](assets/simulated-epoching.jpg)
*图 22：所有来源按同一比例下采样，小模型中偏爱稀缺来源的 mixture 会提前暴露重复和过拟合。（字幕区间：01:08:25--01:09:31）*

数据源还可以继续细分。单个 Common Crawl 可先按 topic/domain 与 quality 分桶，每个格子再成为 mixture component。这说明 filtering、分桶和 mixing 并非完全独立的阶段。

### 本章小结

- 配比必须与来源容量、训练预算和实际 epoch 数一起审查。
- UniMax 用重复上限防止小来源被过度采样。
- 回归式 mixing 用小模型学习 mixture 到目标的代理关系，但存在极值点与跨尺度失效。
- Simulated epoching 通过同比缩小来源，让小实验复制大训练的重复压力。

## 7. 后训练数据：任务、环境与合成轨迹

### 7.1 通用配方与任务依赖性

预训练与 mid-training 主要发展通用能力；后训练数据更像 evaluation，环境、prompt 和理想响应都由目标能力决定。课程给出的通用配方是：

1. 定义 environments；
2. 定义 tasks / prompts；
3. 从强 teacher 收集 responses。

开放数据集多数使用模型 teacher；人类 teacher 更慢、更贵，前沿实践可采用人机混合。关键不是“合成”这个标签，而是 prompt 是否覆盖目标分布、teacher 是否真的会教学、响应是否经过有效验证。

### 7.2 OpenThoughts：更强模型不一定是更好教师

OpenThoughts 使用 QwQ-32B 生成 1.2M 样例，问题来自 27 个真人与合成来源，包括 StackExchange、NuminaMath、化学与代码数据。

![OpenThoughts 的题目来源](assets/openthoughts-sources.png)
*图 23：不同来源覆盖 coding puzzle、合成推理、多语言代码、code review 与教育材料，展示 prompt 分布本身就是设计变量。（字幕区间：01:15:34--01:16:30）*

![OpenThoughts 的过滤、去重与多响应管线](assets/openthoughts-pipeline.png)
*图 24：问题经过过滤、去重和随机采样，每题生成 16 个响应，约 75K questions 最终形成 1.2M responses。（字幕区间：01:17:13--01:17:42）*

反直觉结论包括：每题采样 16 个响应有效；QwQ-32B 在这个设置中比当时更强的 DeepSeek-R1 更适合作 teacher；简单 answer filtering 没有帮助；较小但高质量的来源可能优于更大、更杂的来源。

> [!IMPORTANT]
> Teacher 的 benchmark 能力和生成“适合学生学习的数据”的能力不是同一个量。后训练数据需要单独评估覆盖度、推理可读性、错误模式与学生模型收益。

### 7.3 SWE-smith：真实环境加合成任务

数学问题几乎不需要外部环境；软件工程任务却绑定仓库、依赖、测试和工具。SWE-smith 从真实仓库出发，建立环境，再通过程序化修改、LM 生成、组合 bug 或 PR mirroring 创造任务，最后用测试验证。

![SWE-smith 半合成任务流水线](assets/swe-smith.png)
*图 25：真实 repository 与 unit tests 保留环境真实性，任务本身可合成；128 个 GitHub 仓库产生约 50K tasks。（字幕区间：01:18:14--01:18:54）*

这种方案属于 semi-synthetic：环境真实，任务可扩展生成。瓶颈是为成千上万旧仓库安装依赖和维护 Docker image。

### 7.4 SWE-Zero：用无执行轨迹换规模

SWE-Zero 的观察是，强模型即使没有 execution feedback，仍能解决相当比例的软件任务；这说明参数中包含一定的代码语义 world model。无执行显然更弱，但足以生成大量有价值的训练轨迹。

![无执行和有执行的 SWE-bench 对比](assets/swezero-noexec.png)
*图 26：MiniMax-M2.5 无执行时为 69.5/57.2，有执行时为 80.2/74.1；其他模型也显示执行反馈仍有明显价值。（字幕区间：01:19:48--01:20:25）*

![标准 OpenHands 与 SWE-Zero 提示约束](assets/swezero-prompt.png)
*图 27：execution-free prompt 明确禁用 Python、pytest、mypy、pip、apt 等命令，也禁止写或运行测试，迫使轨迹依靠静态探索、分析与实现。（字幕区间：01:20:38--01:21:17）*

数据由 Qwen3-Coder-480B teacher 蒸馏，并过滤仍试图执行命令的轨迹。SWE-Zero 生成 300K 条不依赖仓库专属执行的 agent trajectories；随后加入约 13K 条需要 execution feedback 的 SWE-Hero 轨迹。

![SWE-Zero 到 SWE-Hero 的第二阶段收益](assets/swezero-results.png)
*图 28：在 7B、14B 与 32B 尺寸上，加入 SWE-Hero 阶段分别带来约 +5.9、+6.3、+4.7 个百分点，但仍与最大前沿模型有差距。（字幕区间：01:21:17--01:21:45）*

> [!WARNING]
> Execution-free 不等于 execution-equivalent。它的意义是便宜、可扩展，并能提供大量静态代码推理样本；表格同时证明运行测试和环境反馈仍然提高解题能力。

### 7.5 SWE-rebench 与 SWE-ZERO-12M

SWE-rebench 从 450K PRs 开始，结合 GitHub Archive 元数据，先初筛，再尝试安装环境、运行验证，并用 Qwen2.5-72B-Instruct 协助推断依赖与评估 PR 质量，最终得到 21K+ interactive Python SWE tasks，覆盖 3.4K repos。

![SWE-rebench 数据构建流程](assets/swe-rebench.png)
*图 29：GitHub/GHArchive 初筛、环境安装验证、LLM labeling 三步将大规模 PR 池收缩为可用任务。（字幕区间：01:21:45--01:22:13）*

SWE-ZERO-12M 进一步使用 SWE-rebench-v2 的 32K executable 与 120K nonexecutable tasks，以 mini-swe-agent 和 mini-coder-1.7B 扩展到 12M trajectories；课程报告后者 pass@100 为 50.4。讲者称该数据在授课当天刚发布，这是授课时点的陈述，不应当作永久的“当前最新”。

### 本章小结

- 后训练数据更接近评测任务，必须显式定义环境、prompt 与理想响应。
- 强 teacher 不必然产生更适合学生的数据；多响应采样与来源质量同样重要。
- SWE-smith 保留真实环境、合成任务；SWE-Zero 牺牲执行反馈以扩大轨迹规模。
- 软件环境、依赖验证、agent hacking 与质量标注使代码数据远比数学数据昂贵。

## 总结与延伸

### 讲者的收束

课程最后把整条链路压缩为四个动作：

1. **Filtering**：定义“好数据”，训练轻量分类器把标准外推到 Web crawl；
2. **Deduplication**：用哈希与近似匹配减少重复、记忆和 FLOPs 浪费；
3. **Mixing**：在小规模尝试 mixture，但检查 epoching 与跨尺度迁移；
4. **Post-training data**：让数据形态接近 evaluations，大量使用合成 prompts、responses 与 agent trajectories。

讲者最后的现实校正是：论文和课程把流程画得很整洁，真实数据工作却高度 domain-specific，充满逐例查看、安装失败、错误标签、过滤细节和例外处理。算法提供骨架，数据审计决定它在现实中是否成立。

### 全课的统一逻辑

把各章压缩到同一框架，会发现三个不断重复的结构：

- **先定义目标，再做代理。** Filtering 用 \(T\) 定义目标，回归式 mixing 用下游 loss 定义目标，后训练用 task 与 teacher 定义目标。
- **先在小规模估计，再扩展。** fastText 把少量标签扩到 Web，MinHash/LSH 把相似度估计扩到大语料，代理模型把 mixture 搜索扩到大训练。
- **每次扩展都可能改变问题。** 过滤偏置会被放大；LSH 只给概率候选；小模型最优 mixture 会因 epoching 失效；更强 teacher 也可能不是更好的教师。

因此，数据工程的核心能力不是盲目扩大数据量，而是持续追问：代理指标是否仍代表真实目标？训练规模是否改变了数据重复结构？选择和清理是否丢掉了目标能力所需的信号？

### 实践检查清单

- 转换：抽样检查阅读顺序、表格、代码、图注和字符编码。
- 过滤：记录目标集合来源、阈值、保留率、误杀案例与目标偏置。
- 去重：明确 item、shingle、相似度阈值、候选召回率与跨来源范围。
- 混合：把每个权重换算成实际 \(E_s=p_sT/N_s\)，不要只看百分比。
- 代理实验：检查 argmin 附近误差，并用 simulated epoching 模拟目标规模。
- 后训练：分别评价 prompt 覆盖、teacher 教学质量、执行环境真实性与响应过滤。

### 开放问题

1. 如何设计不会被代理回归器或优化器钻空子的 mixture/evaluation target？
2. 除 teacher 自身 benchmark 外，怎样直接度量合成响应的教学价值？
3. Execution-free 轨迹在哪种任务复杂度、代码库规模或依赖条件下开始明显失效？
4. Filtering、质量分桶、mixing 和后训练采样能否联合优化，而不是串行决定？
5. 如何在数据可复现、版权隐私、能力覆盖和训练成本之间建立可审计的折中？

> [!WARNING]
> “更强的优化”不保证“更好的数据”。当代理目标与真实目标不一致时，优化器会更高效地优化错误的东西。最可靠的防线仍是明确目标、保留可追溯来源、检查具体样本，并在目标训练规模下验证。

### 本章小结

- 数据 pipeline 是一组相互耦合的建模决策，而不是中性的清洗步骤。
- 质量、数量、重复轮数与训练规模必须联合分析。
- 合成数据的关键是 task、teacher、environment 与验证机制，不是“合成”二字本身。
- 最终的工程准则是：让小规模实验尽量复制大规模问题，并始终用真实样本和下游行为检查代理假设。
