# mamba.c

<p align="center">
  <img src="assets/cute-mamba.png" width="300" height="300" alt="Cute Mamba">
</p>

<p align="right"><a href="https://github.com/kroggen/mamba.c/blob/learning/README.md">English</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ja.md">日本語</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ru.md">Русский</a></p>

纯C语言推断Mamba模型

受到[llama2.c](https://github.com/karpathy/llama2.c)的启发并使用其代码

这只实现了Mamba SSM的循环模式

您可以将其与[相关的pytorch实现](https://github.com/kroggen/mamba-cpu/tree/recurrent-only)进行比较

不支持批处理。代码最小化以便学习。

即便如此，它在CPU上的速度比pytorch还要快！！！

## 快速开始

```
python3 tokenizer.py
python3 export.py state-spaces/mamba-130m model.bin
make fast
./mamba model.bin -n 20 -i "Customer Support should" -t 0.0
```
您可以在导出部分选择另一个模型

## 模型

您可以使用存储在[HuggingFace](https://huggingface.co/state-spaces)上的这些模型：

* `state-spaces/mamba-130m`
* `state-spaces/mamba-370m`
* `state-spaces/mamba-790m`
* `state-spaces/mamba-1.4b`
* `state-spaces/mamba-2.8b`
* `state-spaces/mamba-2.8b-slimpj`

您可以将模型名称作为`export.py`脚本的参数

请注意，导出脚本将下载模型（如果尚未下载）到hugingface缓存目录。

您也可以选择指定模型文件的路径，如果您手动下载了它。例如：

```
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/config.json?download=true -O config.json
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/pytorch_model.bin?download=true -O pytorch_model.bin
python3 export.py . model.bin
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

主要有2个分支：

* `learning` - 非常基础
* `fused` - 将基本功能融合成更大的功能

您可以[比较](https://github.com/kroggen/mamba.c/compare/learning..fused)它们


## 注释

分词器可能需要对特殊字符进行更多的工作

欢迎贡献并发送PR



## 许可证

MIT