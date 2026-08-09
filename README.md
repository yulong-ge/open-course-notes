# Open Course Notes

[![Deploy MkDocs to GitHub Pages](https://github.com/yulong-ge/open-course-notes/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/yulong-ge/open-course-notes/actions/workflows/deploy-pages.yml)
[![Website](https://img.shields.io/badge/在线阅读-GitHub%20Pages-0969da)](https://yulong-ge.github.io/open-course-notes/)
[![Content license](https://img.shields.io/badge/讲义-CC%20BY--NC--SA%204.0-2b5797)](LICENSE.md)

面向中文学习者的高质量公开课程讲义库。每门课程独立放在 `docs/courses/<course-id>/` 中，共享同一套 MkDocs 站点、搜索、导航、质量检查和 GitHub Pages 部署流程。

## 当前课程

| 课程 | 学期 | 内容 | 状态 |
|---|---|---:|---|
| [Stanford CS336: Language Modeling from Scratch](docs/courses/stanford-cs336-2026/index.md) | Spring 2026 | 17 节正课 + 1 节嘉宾讲座 | 完成 |

在线站点：<https://yulong-ge.github.io/open-course-notes/>

## 为什么采用一个多课程仓库

- 读者只需收藏一个站点，即可搜索和浏览全部课程。
- 新课程复用统一的页面样式、数学公式、代码高亮和自动部署。
- 每门课程保留独立目录、来源说明和审查记录，避免内容互相污染。
- 仓库级工具只维护一次，后续增加课程主要是增加 Markdown 与素材。

## 目录结构

```text
open-course-notes/
├── docs/
│   ├── index.md
│   ├── courses/
│   │   └── stanford-cs336-2026/
│   │       ├── index.md
│   │       ├── quality-review.md
│   │       ├── lecture01-overview-tokenization/
│   │       │   ├── index.md
│   │       │   └── assets/
│   │       └── ...
│   ├── about.md
│   ├── contributing.md
│   └── licensing.md
├── scripts/
│   └── validate_notes.py
├── .github/workflows/deploy-pages.yml
├── mkdocs.yml
├── pyproject.toml
└── uv.lock
```

## 本地预览

项目使用 [uv](https://docs.astral.sh/uv/) 锁定依赖：

```bash
uv sync --locked
uv run python scripts/validate_notes.py
uv run mkdocs serve
```

浏览器打开 <http://127.0.0.1:8000/>。生产构建使用：

```bash
uv run mkdocs build --strict
```

## 自动部署

对 `main` 的内容或配置变更会触发 GitHub Actions：

1. 校验讲义结构、图片链接和本地素材；
2. 使用锁定依赖执行 `mkdocs build --strict`；
3. 上传静态站点 artifact；
4. 部署到 GitHub Pages 项目子目录 `/open-course-notes/`。

Pull Request 同样会执行校验与构建，但不会部署线上站点。

## 内容来源与质量边界

讲义基于公开视频、字幕和公开课件整理，保留课程来源与视频时间区间。CS336 全套讲义还由同一个 Luna 审阅会话逐份评估并完成修订复审。

本仓库不是 Stanford University 或原课程团队的官方项目。公开仓库不再分发原始视频、完整字幕或官方课件源文件；讲义中引用的截图、图表与课程名称归各自权利人所有。

详见 [许可与权利说明](LICENSE.md) 和 [站点内说明](docs/licensing.md)。

## 参与贡献

欢迎修正公式、事实、链接和排版，或提议新增高质量课程。新增课程请遵循 [贡献指南](docs/contributing.md)，保留可核验来源，并确保站点可以严格构建。

