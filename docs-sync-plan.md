# 文档同步与站点构建方案

将 `antgroup/vsag` 仓库中的 markdown 文档同步到 `vsag-io/vsag-io.github.io`，并由后者构建发布为 GitHub Pages 站点。

## 背景与现状

### 源仓库 `antgroup/vsag`

当前 `docs/` 结构（部分）：

```
docs/
├── docs/
│   └── zh/                       # 已有 mdbook 源
│       ├── book.toml
│       └── src/
│           ├── SUMMARY.md
│           ├── README.md
│           ├── guide/ advanced/ development/ resources/ misc/
├── blog/
│   └── zh/
│       └── 2025-07-16-rabitq-intro/
│           ├── rabitq_intro_zh.md
│           └── fig1..6.png
├── brute_force.md / hgraph.md / ivf.md / sindi.md ...   # 散落 md，不同步
├── banner.svg / *.png / *.jpg                           # 散落资源，不同步
└── .github/workflows/docs.yaml                          # 当前同步 workflow
```

当前 workflow 通过 `rsync -av --delete --exclude='.git/' repo-vsag/docs/ repo-vsagio/`
把整个 `docs/` 拍到目标仓库根目录，会覆盖目标仓库的 `Makefile`、`en/`、`zh/`、
`assets/`、`.github/` 等基础设施，属于破坏性同步，不可用。

### 目标仓库 `vsag-io/vsag-io.github.io`

```
├── Makefile                       # mdbook build zh + en → book/{zh,en}
├── assets/index.html              # 语言选择落地页
├── en/ (book.toml + src/)         # mdbook 英文书
├── zh/ (book.toml + src/)         # mdbook 中文书
└── .github/workflows/deploy.yml   # mdbook build + github-pages-deploy-action
```

`deploy.yml` 在 push `main` 后执行 `make build`，把 `book/` 发布到 GitHub Pages。

## 目标

最终站点 URL 布局：

- `vsag-io.github.io/` → 语言/板块落地页（保留 `assets/index.html`）
- `vsag-io.github.io/docs/zh/` 与 `vsag-io.github.io/docs/en/` → 文档 mdbook
- `vsag-io.github.io/blogs/zh/` 与 `vsag-io.github.io/blogs/en/` → 博客 mdbook

## 核心原则

1. **职责分离**：
   - 源仓库 `antgroup/vsag` 只维护 markdown 源内容。
   - 目标仓库 `vsag-io/vsag-io.github.io` 负责 mdbook 配置、构建、部署。

2. **同步范围最小化**：
   sync workflow 只写目标仓库的 `docs/` 和 `blogs/` 两个目录，保留 `Makefile`、
   `.github/`、`assets/`、`LICENSE`、`README.md` 等基础设施。

3. **写入方式**：
   推到目标仓库的 `sync` 分支，不自动合并。人工 review 后合并到 `main`，触发
   目标仓库的 `deploy.yml` 完成构建与发布。

## 源仓库目录约定

保持顶层 `docs/` 目录名不变，内部严格按以下约定组织：

```
docs/
├── docs/                          # 文档 mdbook 源（对应站点 /docs/*）
│   ├── zh/
│   │   ├── book.toml
│   │   └── src/
│   │       ├── SUMMARY.md
│   │       └── ...
│   └── en/                        # 新增（骨架）
│       ├── book.toml
│       └── src/
│           └── SUMMARY.md
├── blog/                          # 博客 mdbook 源（对应站点 /blogs/*）
│   ├── zh/                        # 补齐 mdbook 结构
│   │   ├── book.toml
│   │   └── src/
│   │       ├── SUMMARY.md
│   │       └── 2025-07-16-rabitq-intro/
│   │           ├── rabitq_intro_zh.md
│   │           └── fig1..6.png
│   └── en/                        # 新增（骨架）
│       ├── book.toml
│       └── src/
│           └── SUMMARY.md
└── （其他散落 md/图片文件暂不变，也不被同步）
```

注意：

- 源目录名 `docs/docs/` 的双层嵌套保持不变，只是内部可读性略差；URL 路径由目标
  仓库的目录结构决定，不受源目录名影响。
- `docs/blog/zh/` 目前没有 `book.toml` 和 `SUMMARY.md`，需要补齐才能作为 mdbook
  构建。已有的博文目录和图片可以原地保留，只新增 mdbook 元文件。

## 目标仓库改造

需要人工一次性完成以下改动（不要通过 sync workflow 做，避免破坏性）：

1. 目录移动：
   - `en/` → `docs/en/`
   - `zh/` → `docs/zh/`
2. 新建博客 mdbook 骨架：
   - `blogs/zh/book.toml` + `blogs/zh/src/SUMMARY.md`
   - `blogs/en/book.toml` + `blogs/en/src/SUMMARY.md`
3. 更新 `Makefile`：

   ```make
   build:
   	mdbook build docs/zh   -d ../../book/docs/zh
   	mdbook build docs/en   -d ../../book/docs/en
   	mdbook build blogs/zh  -d ../../book/blogs/zh
   	mdbook build blogs/en  -d ../../book/blogs/en
   	cp assets/index.html book/index.html

   serve-docs-zh:
   	mdbook serve docs/zh -d ../../book/docs/zh --open
   serve-docs-en:
   	mdbook serve docs/en -d ../../book/docs/en --open
   serve-blogs-zh:
   	mdbook serve blogs/zh -d ../../book/blogs/zh --open
   serve-blogs-en:
   	mdbook serve blogs/en -d ../../book/blogs/en --open
   ```

4. 视情况更新 `assets/index.html` 内的跳转链接（指向 `/docs/zh/`、`/docs/en/`、
   `/blogs/zh/`、`/blogs/en/`）。
5. `.github/workflows/deploy.yml` 无需改动，`folder: book` 依然有效。
6. 合并后验证 Pages 输出的 URL 可达。

## 源仓库 workflow 改写

替换 `.github/workflows/docs.yaml` 为：

```yaml
name: Sync Docs to Repo VSAGIO

on:
  push:
    branches:
      - main
    paths:
      - 'docs/**'

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repo VSAG
        uses: actions/checkout@v4
        with:
          path: repo-vsag

      - name: Checkout Repo VSAGIO
        uses: actions/checkout@v4
        with:
          repository: vsag-io/vsag-io.github.io
          token: ${{ secrets.GH_PAT }}
          path: repo-vsagio

      - name: Sync docs books (zh, en)
        run: |
          mkdir -p repo-vsagio/docs
          rsync -av --delete repo-vsag/docs/docs/zh/ repo-vsagio/docs/zh/
          rsync -av --delete repo-vsag/docs/docs/en/ repo-vsagio/docs/en/

      - name: Sync blogs books (zh, en)
        run: |
          mkdir -p repo-vsagio/blogs
          rsync -av --delete repo-vsag/docs/blog/zh/ repo-vsagio/blogs/zh/
          rsync -av --delete repo-vsag/docs/blog/en/ repo-vsagio/blogs/en/

      - name: Commit and Push to sync branch
        run: |
          cd repo-vsagio
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          if [[ -z $(git status --porcelain) ]]; then
            echo "No changes to commit."
            exit 0
          fi
          git checkout -B sync
          git add docs blogs
          git commit -m "docs: sync from vsag@${{ github.sha }}"
          git push --force origin sync
```

要点：

- `rsync --delete` 作用域限定在 `docs/{zh,en}/` 和 `blogs/{zh,en}/`，不影响目标仓库
  的 `Makefile`、`.github/`、`assets/`、`LICENSE`、`README.md`。
- `git add docs blogs` 仅暂存这两个目录，提供二次保险。
- 继续使用 "推到 `sync` 分支、人工合并到 `main`" 的流程。

## Token 权限

继续使用当前的 Fine-grained Personal Access Token `GH_PAT`，权限保持最小：

- Resource owner: `vsag-io`
- Repository access: 仅 `vsag-io/vsag-io.github.io`
- Repository permissions:
  - Contents: Read and write（唯一需要的权限）
  - 其余保持 No access

配置位置：`antgroup/vsag` 的 Settings → Secrets and variables → Actions → `GH_PAT`。

## 执行顺序

1. **目标仓库 `vsag-io/vsag-io.github.io` 一次性改造**：
   - 移动 `en/` → `docs/en/`，`zh/` → `docs/zh/`
   - 新建 `blogs/zh/` 和 `blogs/en/` mdbook 骨架
   - 更新 `Makefile`
   - 视需要调整 `assets/index.html` 链接
   - 合并到 `main`，验证 `deploy.yml` 成功构建并发布

2. **源仓库 `antgroup/vsag` 配合改造**：
   - 新建 `docs/docs/en/`（`book.toml` + 占位 `src/SUMMARY.md`、`src/README.md`）
   - 为 `docs/blog/zh/` 补齐 mdbook 结构（`book.toml` + `src/SUMMARY.md`），将
     已有博文路径纳入 SUMMARY
   - 新建 `docs/blog/en/` 骨架
   - 用上面的新内容替换 `.github/workflows/docs.yaml`

3. **验证链路**：
   - 在源仓库 push 一次 `docs/**` 下的小改动
   - 检查目标仓库 `sync` 分支是否出现预期 diff
   - 人工 review 并合并 `sync` → `main`
   - 观察 `deploy.yml` 运行结果与站点 URL 可达性

## 风险与注意

- 目标仓库首次改造属于破坏性改动，必须人工做，不要让 sync workflow 触发。
- `docs/blog/zh/2025-07-16-rabitq-intro/rabitq_intro_zh.md` 这种命名在 mdbook
  `SUMMARY.md` 中可以直接引用，无需改名。
- 当前 workflow 第 29 行的 `--exclude='.git/'` 在新版本中不再需要，因为同步源
  路径不包含目标仓库的 `.git`。
- 若未来要让 sync workflow 自动开 PR 而非 push 分支，需要给 `GH_PAT` 追加
  `Pull requests: Read and write` 权限，并引入 `peter-evans/create-pull-request`。
