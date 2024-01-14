# Mamba C

<p align="center">
  <img src="assets/cute-mamba.png" width="300" height="300" alt="Cute Mamba">
</p>

Inference of Mamba models in pure C

Inspired and using code from [llama2.c](https://github.com/karpathy/llama2.c)

This implements only the recurrent mode of Mamba SSM

You can compare it with the [related pytorch impementation](https://github.com/kroggen/mamba-cpu/tree/recurrent-only)

No support for batches. The code is minimal for learning purposes.

Even so, it is way faster than pytorch on CPU!!!


## Fast Start

```
python3 tokenizer.py
python3 export.py state-spaces/mamba-130m model.bin
make fast
./mamba model.bin -n 20 -i "Customer Support should" -t 0.0
```
You can select another model on the export part

## Models

Currently these are the available models:

* [state-spaces/mamba-130m](https://huggingface.co/state-spaces/mamba-130m)
* [state-spaces/mamba-370m](https://huggingface.co/state-spaces/mamba-370m)
* [state-spaces/mamba-790m](https://huggingface.co/state-spaces/mamba-790m)
* [state-spaces/mamba-1.4b](https://huggingface.co/state-spaces/mamba-1.4b)
* [state-spaces/mamba-2.8b](https://huggingface.co/state-spaces/mamba-2.8b)
* [state-spaces/mamba-2.8b-slimpj](https://huggingface.co/state-spaces/mamba-2.8b-slimpj)

You can specify the model name as an argument to the `export.py` script

Note that the export script will download the model (if it's not already downloaded) to the hugingface cache directory.

Optionally you can also specify the path to the model file, if you downloaded it manually. Example:

```
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/config.json?download=true -O config.json
wget https://huggingface.co/state-spaces/mamba-130m/resolve/main/pytorch_model.bin?download=true -O pytorch_model.bin
python3 export.py . model.bin
```


## Notes

The tokenizer may need some more work for special characters

Feel free to contribute and send a PR



## License

MIT
