# mamba.c

<p align="center">
  <img src="assets/mamba-c.png" width="300" height="300" alt="Mamba C">
</p>

<p align="right"><a href="https://github.com/kroggen/mamba.c/blob/learning/README-zh.md">中文</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ja.md">日本語</a> | <a href="https://github.com/kroggen/mamba.c/blob/learning/README-ru.md">Русский</a></p>

Inference of Mamba models in pure C

Inspired by and using code from [llama2.c](https://github.com/karpathy/llama2.c)

This implements only the recurrent mode of Mamba SSM

You can compare it with the [related pytorch impementation](https://github.com/kroggen/mamba-cpu/tree/recurrent-only)

No support for batches. The code is minimal for learning purposes.

Even so, it is faster than pytorch on CPU!!!


## Fast Start

```
python3 tokenizer.py
python3 export.py state-spaces/mamba-130m model.bin
make fast
./mamba model.bin -n 20 -i "Customer Support should" -t 0.0
```
You can select another model on the export part

## Models

You can use these models stored on [HuggingFace](https://huggingface.co/state-spaces):

* `state-spaces/mamba-130m`
* `state-spaces/mamba-370m`
* `state-spaces/mamba-790m`
* `state-spaces/mamba-1.4b`
* `state-spaces/mamba-2.8b`
* `state-spaces/mamba-2.8b-slimpj`

You can specify the model name as an argument to the `export.py` script

Note that the export script will download the model (if it's not already downloaded) to the hugingface cache directory.

Optionally you can also specify the path to the model file, if you downloaded it manually. Example:

```
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/config.json?download=true -O config.json
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/pytorch_model.bin?download=true -O pytorch_model.bin
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

There are mainly 2 branches:

* `learning` - very basic
* `fused` - fuse the basic functions into bigger ones

You can [compare](https://github.com/kroggen/mamba.c/compare/learning..fused) them


## Notes

The tokenizer may need some more work for special characters

Feel free to contribute and send a PR



## License

MIT
