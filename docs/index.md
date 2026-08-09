# Open Course Notes

面向中文学习者的高质量公开课程讲义库。

这里不追求把视频字幕换一种格式重排，而是把完整课程重构成可以独立阅读的教学材料：补齐先修、展开公式、解释张量形状、保留关键图示，并明确区分课堂事实、工程近似和延伸内容。

<div class="course-grid" markdown>

<div class="course-card" markdown>

## Stanford CS336 · 2026

**Language Modeling from Scratch**

从 tokenization、Transformer 架构和 GPU kernels，一路学习并行训练、scaling laws、推理、评测、数据工程、后训练与多模态。

- 17 节正课 + 1 节嘉宾讲座
- 完整中文 Markdown 讲义
- 公式、代码、关键帧与视频时间区间
- Luna 初学者教学质量审查：18/18 PASS

[开始学习 :material-arrow-right:](courses/stanford-cs336-2026/index.md){ .md-button .md-button--primary }

</div>

</div>

## 这个项目如何组织

每门课程位于独立目录，共享站点基础设施：

```text
docs/courses/
├── stanford-cs336-2026/
│   ├── index.md
│   ├── quality-review.md
│   └── lecture.../
└── future-course-id/
    ├── index.md
    └── ...
```

这使后续课程可以继续加入同一个仓库和站点，同时保留各自的来源、结构与审查记录。

## 阅读建议

1. 从课程总览了解依赖关系和学习路线。
2. 按 lecture 顺序阅读；每章末尾的“本章小结”用于自检。
3. 遇到公式时先确认动机和符号，再跟随数值或 shape 例子。
4. 配图下方保留视频区间，可随时回到原讲核对上下文。
5. 将讲义当作学习地图而非实验替代品；实现、测量与复现实验仍然重要。

!!! note "非官方项目"
    本站不是原课程团队的官方项目。课程名称、课件图表和视频画面的权利归相应权利人所有。
