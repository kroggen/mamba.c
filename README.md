# mamba.c

<p align="center">
  <img src="assets/mamba-c.png" width="300" height="300" alt="Mamba C">
</p>

<p align="right"><a href="https://github.com/kroggen/mamba.c/blob/learning/README-zh.md">中文</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ja.md">日本語</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ru.md">Русский</a></p>

Inference of Mamba 1, 2 & 3 models in pure C

Inspired by and using code from [llama2.c](https://github.com/karpathy/llama2.c)

This implements only the recurrent mode of Mamba SSM

You can compare it with the [related pytorch implementation](https://github.com/kroggen/mamba-cpu/tree/recurrent-only)

No support for batches. The code is minimal for learning purposes.

Even so, it is faster than pytorch on CPU!!!


## Fast Start

Once Mamba-3 model weights are publicly released (see [Models](#models) below):

```
python3 tokenizer.py
python3 export.py state-spaces/mamba3-130m model.bin
make fast
./mamba model.bin -n 20 -i "Customer Support should" -t 0.0
```
Python is only used to export the tokenizer and the model to a simpler format (requires transformers and pytorch)

## Models

> **Note:** As of March 2026, no Mamba-3 model weights have been publicly released yet.
> The paper ([arXiv:2603.15569](https://arxiv.org/abs/2603.15569)) was submitted on March 16, 2026.
> The [state-spaces](https://huggingface.co/state-spaces) HuggingFace org currently only hosts Mamba-1 and Mamba-2 checkpoints.
> Watch that page for future Mamba-3 releases.

When weights become available, the export script expects a standard HuggingFace checkpoint with the
`backbone.layers.N.mixer.*` / `backbone.layers.N.mlp.*` layout used in `mamba3.py`.
You can then run:

```
python3 export.py state-spaces/mamba3-130m model.bin
```

Or manually:

```
python3 export.py /path/to/local/mamba3-model model.bin
```

## Internal State

As it is a recurrent model, it is possible to save the internal state and then return to that state later

To get a copy of the internal state:

```c
  int state_size;
  char* state = get_internal_state(mamba, &state_size);
```

To set the internal state:

```c
  set_internal_state(mamba, state, state_size);
```


## Branches

The code is available on 3 versions, each on a separate branch:

* `learning` - very basic
* `fused` - fuse the basic functions into bigger ones (you can [compare](https://github.com/kroggen/mamba.c/compare/learning..fused) them)
* `cuda` - simple GPU implementation, easy to understand

There is also code for Mamba 2:

* `mamba2-learning` - very basic ([compare with mamba1](https://github.com/kroggen/mamba.c/compare/learning..mamba2-learning))
* `mamba2-fused` - fused functions ([compare with learning](https://github.com/kroggen/mamba.c/compare/mamba2-learning..mamba2-fused) | [compare with mamba1](https://github.com/kroggen/mamba.c/compare/fused..mamba2-fused))

And for Mamba 3 (ICLR 2026):

* `mamba3-learning` - very basic ([compare with mamba2](https://github.com/kroggen/mamba.c/compare/mamba2-learning..mamba3-learning))

Mamba-3 key changes vs Mamba-2:
- **Trapezoidal discretization**: `h_t = α*h_{t-1} + β*B̄_{t-1}x_{t-1} + γ*B̄_t*x_t` (requires tracking `prev_Bx`)
- **Data-dependent RoPE**: B and C are rotated by cumulative angles derived from input θ and step size Δ
- **QK-normalization**: RMSNorm applied to B and C after projection (replaces the gated RMSNorm output norm)
- **Learnable BC bias**: head-specific bias added to B and C after QK-norm, initialized to ones
- **No short convolution**: the trapezoidal rule + bias makes conv1d unnecessary
- **Llama-style architecture**: each layer is `RMSNorm → SSM → residual → RMSNorm → SwiGLU MLP → residual`


## Notes

The tokenizer may need some more work for special characters

Feel free to contribute and send a PR



## License

MIT
