import os
import struct
import argparse
import json

import numpy as np
import torch

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

# -----------------------------------------------------------------------------
# model export functions

def write_weights(file, model, key):
    """ writes the layer weights to file """
    print(f"writing {key} {list(model[key].shape)[::-1]}")
    serialize_fp32(file, model[key])

def write_layer_weights(file, model, layer, n_layers):
    """ writes the layer weights to file """
    print(f"writing {layer % n_layers} {list(model[layer % 0].shape)[::-1]}")
    for n in range(n_layers):
        serialize_fp32(file, model[layer % n])

def model_export(model, config, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    """
    version = 1

    out_file = open(filepath, 'wb')

    # first write the header (256 bytes)

    # write magic, uint32 of "Mamb"
    out_file.write(struct.pack('I', 0x4d616d62))
    # write version
    out_file.write(struct.pack('i', version))

    # write the params (7 integers + 1 byte)
    d_inner = model['layers.0.mixer.D'].shape[0]
    dt_rank = model['layers.0.mixer.dt_proj.weight'].shape[1]
    d_state = model['layers.0.mixer.A_log'].shape[1]
    d_conv = model['layers.0.mixer.conv1d.weight'].shape[2]

    shared_classifier = torch.equal(model['embedding.weight'], model['lm_head.weight'])

    print(f"writing header\n  layers: {config.n_layers}\n  vocab_size: {config.vocab_size}\n  d_model: {config.d_model}\n  d_inner: {d_inner}\n  dt_rank: {dt_rank}\n  d_state: {d_state}\n  d_conv: {d_conv}\n  shared classifier: {shared_classifier}")

    header = struct.pack('iiiiiiii', config.n_layers, config.vocab_size, config.d_model,
                         d_inner, dt_rank, d_state, d_conv, int(shared_classifier))
    out_file.write(header)

    # pad the rest with zeros
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)

    '''
    Example of the model structure:
    embedding.weight - [50280, 768]
    layers.0.mixer.D - [1536]
    layers.0.mixer.in_proj.weight - [3072, 768]
    layers.0.mixer.conv1d.weight - [1536, 1, 4]
    layers.0.mixer.conv1d.bias - [1536]
    layers.0.mixer.x_proj.weight - [80, 1536]
    layers.0.mixer.dt_proj.weight - [1536, 48]
    layers.0.mixer.dt_proj.bias - [1536]
    layers.0.mixer.A_log - [1536, 16]
    layers.0.mixer.out_proj.weight - [768, 1536]
    layers.0.norm.weight - [768]
    norm_f.weight - [768]
    lm_head.weight - [50280, 768]
    '''

    # convert the A_log to A
    for n in range(config.n_layers):
        model[f'layers.{n}.mixer.A'] = -torch.exp(model.pop(f'layers.{n}.mixer.A_log'))

    # write the weights

    # write the embedding weights
    write_weights(out_file, model, 'embedding.weight')

    # layer weights
    write_layer_weights(out_file, model, 'layers.%d.mixer.in_proj.weight', config.n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.conv1d.weight', config.n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.conv1d.bias', config.n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.x_proj.weight', config.n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.dt_proj.weight', config.n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.dt_proj.bias', config.n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.A', config.n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.D', config.n_layers)
    write_layer_weights(out_file, model, 'layers.%d.mixer.out_proj.weight', config.n_layers)
    write_layer_weights(out_file, model, 'layers.%d.norm.weight', config.n_layers)

    # final norm weights
    write_weights(out_file, model, 'norm_f.weight')

    # final classifier weights
    if not shared_classifier:
        write_weights(out_file, model, 'lm_head.weight')

    # write to binary file
    out_file.close()
    print(f"done. saved to {filepath}")


# -----------------------------------------------------------------------------
# Load / import functions

def load_model(path):
    print(f"loading model from {path}")

    # load the model
    if os.path.isdir(path):
        filepath = os.path.join(path, 'pytorch_model.bin')
    else:
        filepath = path
    model = torch.load(filepath, map_location='cpu')

    # remove the 'backbone.' prefix from the keys
    unwanted_prefix = 'backbone.'
    for k,v in list(model.items()):
        if k.startswith(unwanted_prefix):
            model[k[len(unwanted_prefix):]] = model.pop(k)

    # get the path to the config file
    if os.path.isdir(path):
        config_path = os.path.join(path, 'config.json')
    else:
        config_path = os.path.join(os.path.dirname(path), 'config.json')
    # load the config
    with open(config_path) as f:
        config = json.load(f)
    # rename config.n_layers to config.n_layers
    config['n_layers'] = config.pop('n_layer')
    config = argparse.Namespace(**config)    

    return model, config


def get_model_from_huggingface(model_name: str):
    """Download model from HuggingFace and get the path to the model file.
    The model name can be one of the following:
        'state-spaces/mamba-130m'
        'state-spaces/mamba-370m'
        'state-spaces/mamba-790m'
        'state-spaces/mamba-1.4b'
        'state-spaces/mamba-2.8b'
        'state-spaces/mamba-2.8b-slimpj'
    """
    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

    config_path = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    model_path = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)

    return model_path

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="model name or folder where the model files are located", default="state-spaces/mamba-130m")
    parser.add_argument("destination", type=str, help="full path to the output file", default="model.bin")
    args = parser.parse_args()

    # if the source starts with 'state-spaces/mamba-' then load the model from HuggingFace
    if args.source.startswith('state-spaces/mamba-'):
        model_path = get_model_from_huggingface(args.source)
    else:
        model_path = args.source

    model, config = load_model(model_path)

    if model is None:
        parser.error("Can't load input model!")

    # export
    model_export(model, config, args.destination)
