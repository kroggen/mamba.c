# mamba.c

<p align="center">
  <img src="assets/mamba-c.png" width="300" height="300" alt="Mamba C">
</p>

<p align="right"><a href="https://github.com/kroggen/mamba.c/blob/learning/README-zh.md">中文</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ja.md">日本語</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ru.md">Русский</a></p>

Inference of Mamba 1 & 2 models in pure C

Inspired by and using code from [llama2.c](https://github.com/karpathy/llama2.c)

This implements only the recurrent mode of Mamba SSM

You can compare it with the [related pytorch implementation](https://github.com/kroggen/mamba-cpu/tree/recurrent-only)

No support for batches. The code is minimal for learning purposes.

Even so, it is faster than pytorch on CPU!!!


## Fast Start

```
python3 tokenizer.py
python3 export.py state-spaces/mamba2-130m model.bin
make fast
./mamba model.bin -n 20 -i "Customer Support should" -t 0.0
```
Python is only used to export the tokenizer and the model to a simpler format (requires transformers and pytorch)

You can select another model on the export part

## Models

You can use these Mamba 2 models stored on [HuggingFace](https://huggingface.co/state-spaces):

* `state-spaces/mamba2-130m`
* `state-spaces/mamba2-370m`
* `state-spaces/mamba2-780m`
* `state-spaces/mamba2-1.3b`
* `state-spaces/mamba2-2.7b`

You can specify the model name as an argument to the `export.py` script

Note that the export script will download the model (if it's not already downloaded) to the hugingface cache directory.

Optionally you can also specify the path to the model file, if you downloaded it manually. Example:

```
wget https://huggingface.co/state-spaces/mamba2-130m/resolve/main/config.json?download=true -O config.json
wget https://huggingface.co/state-spaces/mamba2-130m/resolve/main/pytorch_model.bin?download=true -O pytorch_model.bin
python3 export.py . model.bin
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


## Notes

The tokenizer may need some more work for special characters

Feel free to contribute and send a PR



## License

MIT
