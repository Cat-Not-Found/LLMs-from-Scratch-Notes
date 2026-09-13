# LLMs-from-Scratch 复现笔记

[中文](#llms-from-scratch-复现笔记) | [English](#english)

这是一个**个人学习项目**：我按照 Sebastian Raschka 的
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 和《Build a Large
Language Model (From Scratch)》一书，从零手写复现了书中前六章的全部代码，并用
Jupyter Notebook 记录为学习笔记。这里不是官方代码，也不是一份新的教程，只是一份
带注释的复现记录，语法和表述仍在学习中。

> 本项目基于原书/原始仓库的 Apache-2.0 许可代码改写（大量代码来自原书），
> 详见 [致谢与原作](#致谢与原作)。

## 目录结构

```
.
├── ch01/                          # 第 1 章：处理文本数据
│   └── ch01.ipynb
├── ch02/                          # 第 2 章：注意力机制
│   └── ch02.ipynb
├── ch03/                          # 第 3 章：实现 GPT 模型
│   └── ch03.ipynb
├── ch04/                          # 第 4 章：在无标注数据上预训练
│   └── ch04.ipynb
├── ch05/                          # 第 5 章：分类任务的微调
│   ├── ch05.ipynb
│   └── app.py                     # 5.9 节的 Chainlit 本地界面
├── ch06/                          # 第 6 章：指令微调
│   └── ch06.ipynb
├── data/                          # 数据集与运行产生的文件（见下）
├── utils/                         # 各章共用的模块
│   ├── paths.py                   # 统一的路径辅助函数
│   ├── previous_chapters.py       # 前几章的模型/训练函数汇总
│   └── gpt_download.py            # 下载并加载 OpenAI GPT-2 权重
├── scripts/                       # 独立脚本（书中 bonus 内容）
│   ├── gpt_with_kv_cache.py       # 带 KV cache 的生成实现（第 3.8 节）
│   └── hyperparam_search.py       # 超参数搜索
├── gpt2/                          # 下载的 GPT-2 权重（不纳入版本控制）
├── requirements.txt
└── README.md
```

## 下载与安装

### 1. 克隆仓库

```bash
git clone git@github.com:Cat-Not-Found/LLMs-from-Scratch-Notes.git
cd LLMs-from-Scratch-Notes
```

> 没有配置 SSH 的话，也可以用 HTTPS 地址：
> `git clone https://github.com/Cat-Not-Found/LLMs-from-Scratch-Notes.git`。

### 2. 创建并激活 conda 环境

本项目的代码在 conda 环境 `LLMs`（Python 3.10.4）下全部跑通：

```bash
conda create -n LLMs python=3.10.4 -y
conda activate LLMs
```

如果习惯用 `venv`，等价写法是 `python3.10 -m venv .venv && source .venv/bin/activate`
（Windows 为 `.venv\Scripts\activate`）。

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

- 默认会安装 **CPU 版** `torch`。如果你想用 NVIDIA 显卡，请到
  <https://pytorch.org/get-started/locally/> 选择对应的 CUDA 版本，直接复制页面上给出的
  `pip install` 命令（不同 torch 版本的 wheel 源地址 tag 不同，请以官网为准）。
- `requirements.txt` 里的版本是本项目实际运行过的版本，若只想读 notebook，也可以适当放宽版本限制。
- `tensorflow` 仅用于读取 OpenAI 发布的 GPT-2 TensorFlow 权重（第 4.7 节起）。
- `triton` 是第 5 章训练时 `torch` 的运行依赖；`bitsandbytes` 提供第 6 章的 8-bit AdamW。
- `chainlit` 仅用于第 5.9 节的本地聊天界面，不需要可以删掉。
- `thop` 仅用于第 3.7 节的 FLOPs 统计。

可选：如果你想在 VS Code / Jupyter 里直接运行 notebook，再安装并注册内核：

```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name LLMs --display-name "Python (LLMs)"
```

### 4. 用 Jupyter Lab / VS Code 打开 notebook 并选择内核

```bash
jupyter lab
```

在 VS Code 中打开任意 `chXX/chXX.ipynb`，右上角选择内核 **Python (LLMs)**；需要时也可以用
`conda activate LLMs` 后运行 `python -m jupyter lab`。

### 5. 首次运行时会自动下载的数据

`gpt2/` 权重、SMS 数据集与 `the-verdict.txt` 都比较大，因此没有放进仓库；第一次运行对应
notebook 时会自动联网下载（见下方表格）。如果网络不稳定导致下载失败，可以手动下载后放到
对应目录，再重新运行该单元格。

- 本项目在 **CPU** 上也能完整跑通，只是第 4~6 章训练会比较慢。
- 第 6 章默认使用 `gpt2-medium (355M)`，显存/内存不足时，把 notebook 里的
  `CHOOSE_MODEL` 改成 `"gpt2-small (124M)"`，或减小 `batch_size`。

### 6. 可选：启动第 5.9 节的本地聊天界面

先跑完 `ch05/ch05.ipynb`（生成 `ch05/review_classifier.pth`），然后：

```bash
cd ch05
chainlit run app.py
```

浏览器打开 <http://localhost:8000> 即可输入短信/评论文字，让微调后的分类器判断是否为 spam。
`ch05/app.py` 里已经处理好了仓库路径，因此在 `ch05/` 目录下执行或在仓库根目录用
`chainlit run ch05/app.py` 都可以。

## 运行方式

用 Jupyter Lab / VS Code 打开各章 notebook，**把工作目录设为该章所在文件夹**
（例如运行 `ch01/ch01.ipynb` 时工作目录为 `ch01/`），然后从上到下依次执行。

每个 notebook 的第一个代码单元或首次用到公共模块的单元会执行：

```python
sys.path.append(str(Path.cwd().parent))
from utils.paths import data_path, repo_path
```

这样无论从仓库根目录还是章节目录启动，`data/` 与 `gpt2/` 都能被正确找到。
路径辅助函数定义在 `utils/paths.py`：`data_path(...)` 返回仓库 `data/` 下的绝对路径，
`repo_path(...)` 返回仓库内的绝对路径，两者都会**自动创建缺失的父目录**——因为 `data/`
不属于版本控制，全新克隆后第一次下载/保存文件时就是靠它们建目录的。

### 各章运行顺序

| 章节 | 内容 | 运行后产生 |
| --- | --- | --- |
| ch01 | BPE 分词、滑动窗口、Token/位置嵌入 | `data/the-verdict.txt`（自动下载） |
| ch02 | 自注意力、因果注意力、多头注意力 | — |
| ch03 | 124M GPT-2 结构、LayerNorm/GELU/残差、FLOPs、KV cache | — |
| ch04 | 交叉熵/困惑度、训练循环、解码策略、加载 GPT-2 预训练权重 | `data/model.pth`、`data/model_and_optimizer.pth`、`gpt2/124M/` |
| ch05 | 垃圾短信分类微调、分类头、Chainlit 界面 | `data/train.csv`、`data/validation.csv`、`data/test.csv`、`ch05/review_classifier.pth`、`data/sms_spam_collection/` |
| ch06 | 指令微调、模型评估 | `data/instruction-data.json`、`data/instruction-data-with-response.json`、`gpt2/355M/` |

请按章节顺序执行：ch04 依赖 ch01 下载的文本，ch05 依赖 ch06 之外的上一章权重，
ch06 依赖 ch05 生成的数据划分。

### 未纳入版本控制的大文件

为了保持仓库体积合理，以下内容被 `.gitignore` 排除，但**均可由代码自动下载或生成**：

| 文件/目录 | 获取方式 |
| --- | --- |
| `data/the-verdict.txt` | 运行 `ch01/ch01.ipynb` 自动下载 |
| `data/sms_spam_collection/`、`data/sms_spam_collection.zip` | 运行 `ch05/ch05.ipynb` 自动下载 |
| `gpt2/`（约 1.8 GB） | 运行 `ch04/ch04.ipynb` / `ch06/ch06.ipynb` 中的 `download_and_load_gpt2(...)` 自动下载 |
| `ch05/review_classifier.pth`（约 523 MB） | 运行 `ch05/ch05.ipynb` 训练后生成 |
| `data/*.pth` | 运行 `ch04/ch04.ipynb` 训练后生成 |

小体积的数据划分（`data/train.csv`、`validation.csv`、`test.csv`）与 `data/instruction-data.json`
已随仓库提供，方便直接阅读第 6 章；`instruction-data-with-response.json` 由第 6 章运行后生成。

## 与原始仓库的差异

- 代码按章节整理到 `ch01/` ~ `ch06/` 目录，公共模块放在 `utils/`。
- 少量**路径引用**被改写为基于 `utils/paths.py` 的写法，以便在任意工作目录下运行；
  模型与算法的实现逻辑没有改动。
- Markdown 笔记是我自己写的，只做了拼写和语法层面的小范围修改。
- `ch05/app.py` 来自原书第 6 章示例界面，仅调整了导入路径与模型权重路径。

## 致谢与原作

本项目是个人学习复现，**绝大部分模型代码、图示与数据集来自原作者**，请以原作为准：

- 原书：Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024.
- 原始仓库：<https://github.com/rasbt/LLMs-from-scratch>（Apache-2.0）
- 数据集：[The Verdict](https://en.wikipedia.org/wiki/The_Verdict)（Edith Wharton）、
  [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)（UCI）、
  [Alpaca 指令数据](https://crfm.stanford.edu/2023/03/13/alpaca.html)。

代码部分沿用原仓库的 Apache-2.0 许可，见 [`LICENSE.txt`](LICENSE.txt)。

---

## English

This repository is a **personal study project**. I hand-wrote a from-scratch reproduction of
the first six chapters of Sebastian Raschka's
[*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch),
using Jupyter notebooks as my learning notes. It is **not** the official code and **not** a new
tutorial — just an annotated record of working through the book while learning to write English
notes. The model code, figures and datasets come from the original author; see
[Acknowledgements](#acknowledgements).

### Repository layout

| Path | Description |
| --- | --- |
| `ch01/` – `ch06/` | One notebook per chapter, following the book |
| `data/` | Datasets, generated data splits and trained artifacts |
| `utils/` | Shared modules: `paths.py`, `previous_chapters.py`, `gpt_download.py` |
| `scripts/` | Standalone bonus scripts: KV-cache generation, hyperparameter search |
| `gpt2/` | Downloaded OpenAI GPT-2 weights (not versioned) |

### Download & installation

**1. Clone the repository**

```bash
git clone git@github.com:Cat-Not-Found/LLMs-from-Scratch-Notes.git
cd LLMs-from-Scratch-Notes
```

> Without SSH configured, use the HTTPS URL instead:
> `git clone https://github.com/Cat-Not-Found/LLMs-from-Scratch-Notes.git`.

**2. Create and activate a conda environment**

Everything was executed with conda environment `LLMs` (Python 3.10.4):

```bash
conda create -n LLMs python=3.10.4 -y
conda activate LLMs
```

With `venv` the equivalent is `python3.10 -m venv .venv && source .venv/bin/activate`
(on Windows: `.venv\Scripts\activate`).

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

- This installs the **CPU** build of `torch`. For an NVIDIA GPU, pick your CUDA version on
  <https://pytorch.org/get-started/locally/> and copy the `pip install` command shown there
  (the wheel index tag differs between torch releases, so always use the official command).
- The versions in `requirements.txt` are exactly the ones used here; feel free to relax them
  if you only want to read the notebooks.
- `tensorflow` is only needed to read OpenAI's GPT-2 checkpoints (from section 4.7 on).
- `triton` is required by `torch` during the chapter 5 training run, and `bitsandbytes`
  provides the 8-bit AdamW used in chapter 6.
- `chainlit` is only needed for the optional local UI in section 5.9.
- `thop` is only used for the FLOPs measurement in section 3.7.

Optional — if you want to run the notebooks directly in VS Code / Jupyter, install and
register a kernel:

```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name LLMs --display-name "Python (LLMs)"
```

**4. Open a notebook in Jupyter Lab or VS Code and pick the kernel**

```bash
jupyter lab
```

Open any `chXX/chXX.ipynb`, select the kernel **Python (LLMs)** in the top-right corner, or run
`python -m jupyter lab` after `conda activate LLMs`.

**5. What is downloaded on the first run**

The `gpt2/` weights, the SMS dataset and `the-verdict.txt` are large, so they are not committed;
the corresponding notebook downloads them automatically the first time it runs (see the table
below). If a download fails, fetch the file manually, drop it in the right folder and re-run
that cell.

- Everything also runs on **CPU**, although training in chapters 4–6 is much slower.
- Chapter 6 uses `gpt2-medium (355M)` by default. If memory is tight, change `CHOOSE_MODEL` to
  `"gpt2-small (124M)"` in the notebook, or lower `batch_size`.

**6. Optional: launch the local chat UI of section 5.9**

First run `ch05/ch05.ipynb` (which produces `ch05/review_classifier.pth`), then:

```bash
cd ch05
chainlit run app.py
```

Open <http://localhost:8000>, type a message and let the finetuned classifier label it as
spam or ham. `ch05/app.py` resolves the repository paths itself, so running it from `ch05/`
or with `chainlit run ch05/app.py` from the repository root both work.

### Running the notebooks

Open a notebook and **set the working directory to its chapter folder** (for example,
`ch01/` when running `ch01/ch01.ipynb`), then execute the cells from top to bottom.
Cells that use shared modules start with:

```python
sys.path.append(str(Path.cwd().parent))
from utils.paths import data_path, repo_path
```

so that `data/` and `gpt2/` resolve no matter where you start Jupyter. Both helpers also create
the parent directory if it is missing, which is what makes the first download/save work on a
fresh clone (the `data/` folder is not tracked by git).

### Chapter order

| Chapter | Content | Produced by the run |
| --- | --- | --- |
| ch01 | BPE tokenization, sliding window, token/positional embeddings | `data/the-verdict.txt` (downloaded) |
| ch02 | Self-attention, causal attention, multi-head attention | — |
| ch03 | 124M GPT-2 architecture, LayerNorm/GELU/residuals, FLOPs, KV cache | — |
| ch04 | Cross-entropy/perplexity, training loop, decoding strategies, loading GPT-2 weights | `data/model.pth`, `data/model_and_optimizer.pth`, `gpt2/124M/` |
| ch05 | Spam-classification fine-tuning, classification head, Chainlit UI | `data/train.csv`, `data/validation.csv`, `data/test.csv`, `ch05/review_classifier.pth`, `data/sms_spam_collection/` |
| ch06 | Instruction fine-tuning, model evaluation | `data/instruction-data.json`, `data/instruction-data-with-response.json`, `gpt2/355M/` |

Run the chapters in order: ch04 reads the text downloaded in ch01, ch05 loads the model
weights produced in ch04, and ch06 uses the data splits written by ch05.

### Files that are not versioned

`gpt2/`, `data/the-verdict.txt`, `data/sms_spam_collection/`, `ch05/review_classifier.pth`
and every `*.pth` checkpoint are listed in `.gitignore` because they are large, and all of
them are downloaded or generated automatically by the notebooks. The small data splits
(`data/train.csv`, `validation.csv`, `test.csv`) and `data/instruction-data.json` are committed
so that chapter 6 can be read right away; `instruction-data-with-response.json` is written by
chapter 6 when it runs.

### Acknowledgements

The overwhelming majority of the model code, the figures and the datasets are the original
author's work — please refer to the original source:

- Book: Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2024.
- Original repository: <https://github.com/rasbt/LLMs-from-scratch> (Apache-2.0)
- Datasets: [The Verdict](https://en.wikipedia.org/wiki/The_Verdict) (Edith Wharton),
  [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (UCI),
  [Alpaca instruction data](https://crfm.stanford.edu/2023/03/13/alpaca.html).

The code follows the Apache-2.0 license of the original repository; see [`LICENSE.txt`](LICENSE.txt).
