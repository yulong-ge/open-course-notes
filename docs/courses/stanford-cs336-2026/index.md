# Stanford CS336 2026：Language Modeling from Scratch

这套讲义覆盖 Stanford CS336 Spring 2026 播放列表的 17 节正课与 Dan Fu 嘉宾讲座，从语言模型最底层的 tokenization 和资源核算，一直走到数据、后训练和多模态。

[YouTube 播放列表 :material-youtube:](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV){ .md-button .md-button--primary }
[课程网站 :material-school:](https://stanford-cs336.github.io/){ .md-button }
[教学质量报告 :material-check-decagram:](quality-review.md){ .md-button }

## 学习路线

```text
表示与训练基础
  Lecture 1–4
      ↓
单卡系统与分布式训练
  Lecture 5–8
      ↓
规模规律、推理与评测
  Lecture 9–12
      ↓
数据、后训练与多模态
  Lecture 13–17
      ↓
推理系统与全栈研究案例
  Guest Lecture
```

## 全部讲义

| # | 主题 | 时长 | 讲义 |
|---:|---|---:|---|
| 1 | Overview & Tokenization | 01:19:22 | [阅读](lecture01-overview-tokenization/index.md) |
| 2 | PyTorch & Einops | 01:17:25 | [阅读](lecture02-pytorch-einops/index.md) |
| 3 | Architectures | 01:29:14 | [阅读](lecture03-architectures/index.md) |
| 4 | Attention Alternatives | 01:26:21 | [阅读](lecture04-attention-alternatives/index.md) |
| 5 | GPUs & TPUs | 01:18:39 | [阅读](lecture05-gpus-tpus/index.md) |
| 6 | Kernels, Triton & XLA | 01:26:41 | [阅读](lecture06-kernels-triton-xla/index.md) |
| 7 | Parallelism I | 01:21:03 | [阅读](lecture07-parallelism/index.md) |
| 8 | Parallelism II | 01:20:11 | [阅读](lecture08-parallelism/index.md) |
| 9 | Scaling Laws I | 01:17:57 | [阅读](lecture09-scaling-laws/index.md) |
| 10 | Inference | 01:25:30 | [阅读](lecture10-inference/index.md) |
| 11 | Scaling Laws II | 01:17:04 | [阅读](lecture11-scaling-laws/index.md) |
| 12 | Evaluation | 01:18:34 | [阅读](lecture12-evaluation/index.md) |
| 13 | Data Sources & Datasets | 01:22:02 | [阅读](lecture13-data-sources-datasets/index.md) |
| 14 | Data Processing & Mixtures | 01:24:46 | [阅读](lecture14-data/index.md) |
| 15 | Mid/Post-Training | 01:19:54 | [阅读](lecture15-mid-post-training/index.md) |
| 16 | RLVR | 01:15:50 | [阅读](lecture16-rlvr/index.md) |
| 17 | Alignment & Multimodality | 01:17:39 | [阅读](lecture17-alignment-multimodality/index.md) |
| Guest | Dan Fu: Inference Systems & Full-Stack Research | 01:11:40 | [阅读](guest-lecture-dan-fu/index.md) |

## 讲义标准

- 每节都以完整人工字幕、视频和公开课件为证据来源。
- 长视频按相邻重叠语义段分析，再合并为统一教学叙事。
- 关键帧和课件图经过视觉检查，并与具体视频区间相邻。
- 公式包含动机、假设和符号解释；代码包含作用与 shape 说明。
- 每个主要章节含“本章小结”，整讲以“总结与延伸”收束。
- 同一个 Luna 会话逐份审查初学者可学性，修订后 18/18 PASS。

## 来源与边界

课程视频、课程名称、原课件和视频画面的权利归 Stanford、课程教师及其他相应权利人所有。本项目是独立的非官方中文学习资料，不代表课程团队背书。

公开仓库不包含原始视频、完整字幕或官方课件源文件。讲义保留必要图示、来源语境和视频时间区间，完整权利说明见[许可页面](../../licensing.md)。
