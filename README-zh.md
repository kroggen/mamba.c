# mamba.c

<p align="center">
  <img src="assets/mamba-c.png" width="300" height="300" alt="Mamba C">
</p>

<p align="right"><a href="https://github.com/kroggen/mamba.c/blob/learning/README.md">English</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ja.md">日本語</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ru.md">Русский</a></p>

纯C语言推断Mamba 1、2 & 3模型

受到[llama2.c](https://github.com/karpathy/llama2.c)的启发并使用其代码

这只实现了Mamba SSM的循环模式

您可以将其与[相关的pytorch实现](https://github.com/kroggen/mamba-cpu/tree/recurrent-only)进行比较

不支持批处理。代码最小化以便学习。

即便如此，它在CPU上的速度比pytorch还要快！！！

## 快速开始

一旦Mamba-3模型权重公开发布（见下方的[模型](#模型)）：

```
python3 tokenizer.py
python3 export.py state-spaces/mamba3-130m model.bin
make fast
./mamba model.bin -n 20 -i "Customer Support should" -t 0.0
```
Python仅用于将分词器和模型导出为更简单的格式（需要transformers和pytorch）

## 模型

> **注意:** 截至2026年3月，Mamba-3模型权重尚未公开发布。
> 论文（[arXiv:2603.15569](https://arxiv.org/abs/2603.15569)）于2026年3月16日提交。
> [state-spaces](https://huggingface.co/state-spaces) HuggingFace组织目前仅托管Mamba-1和Mamba-2检查点。
> 请关注该页面以获取未来的Mamba-3发布。

当权重可用时，导出脚本期望具有`mamba3.py`中使用的`backbone.layers.N.mixer.*` / `backbone.layers.N.mlp.*`布局的标准HuggingFace检查点。
然后您可以运行：

```
python3 export.py state-spaces/mamba3-130m model.bin
```

或手动：

```
python3 export.py /path/to/local/mamba3-model model.bin
```

## 内部状态

由于它是一个循环模型，因此可以保存内部状态，然后稍后返回到该状态

要获取内部状态的副本：

```c
  int state_size;
  char* state = get_internal_state(mamba, &state_size);
```

要设置内部状态：

```c
  set_internal_state(mamba, state, state_size);
```


## 分支

代码有3个版本，每个版本在一个单独的分支上：

* `learning` - 非常基础
* `fused` - 将基本功能融合成更大的功能（你可以[比较](https://github.com/kroggen/mamba.c/compare/learning..fused)它们）
* `cuda` - 简单的GPU实现，易于理解

Mamba 2的代码也可用：

* `mamba2-learning` - 非常基础（[与mamba1比较](https://github.com/kroggen/mamba.c/compare/learning..mamba2-learning)）
* `mamba2-fused` - 融合函数（[与learning比较](https://github.com/kroggen/mamba.c/compare/mamba2-learning..mamba2-fused) | [与mamba1比较](https://github.com/kroggen/mamba.c/compare/fused..mamba2-fused)）

以及Mamba 3（ICLR 2026）：

* `mamba3-learning` - 非常基础（[与mamba2比较](https://github.com/kroggen/mamba.c/compare/mamba2-learning..mamba3-learning)）
* `mamba3-fused` - 融合函数（[与learning比较](https://github.com/kroggen/mamba.c/compare/mamba3-learning..mamba3-fused) | [与mamba2比较](https://github.com/kroggen/mamba.c/compare/mamba2-fused..mamba3-fused)）

Mamba-3相对于Mamba-2的关键变化：
- **梯形离散化**: `h_t = α*h_{t-1} + β*B̄_{t-1}x_{t-1} + γ*B̄_t*x_t`（需要跟踪`prev_Bx`）
- **数据依赖RoPE**: B和C由从输入θ和步长Δ导出的累积角度旋转
- **QK归一化**: 投影后对B和C应用RMSNorm（取代门控RMSNorm输出归一化）
- **可学习BC偏置**: 在QK-norm后添加到B和C的头部特定偏置，初始化为1
- **无短卷积**: 梯形规则 + 偏置使conv1d不必要
- **Llama风格架构**: 每个层为`RMSNorm → SSM → 残差 → RMSNorm → SwiGLU MLP → 残差`


## 注释

分词器可能需要对特殊字符进行更多的工作

欢迎贡献并发送PR



## 许可证

MIT