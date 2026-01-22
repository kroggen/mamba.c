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
    t = tensor.detach().cpu().contiguous().view(-1).to(torch.float32)
    # Use torch's storage to get raw bytes directly
    file.write(t.numpy().tobytes())

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
    Export the Mamba2 model weights in full float32 .bin file to be read from C.
    """
    version = 2

    out_file = open(filepath, 'wb')

    # first write the header (256 bytes)

    # write magic, uint32 of "Mmb2" (Mamba2)
    out_file.write(struct.pack('I', 0x4d6d6232))
    # write version
    out_file.write(struct.pack('i', version))

    # Mamba2 config extraction - infer from model weights if not in config
    d_model = config.d_model
    n_layers = config.n_layer
    vocab_size = config.vocab_size

    # Infer parameters from model weights
    # nheads from A_log shape
    nheads = model['backbone.layers.0.mixer.A_log'].shape[0]
    # d_inner from out_proj or norm.weight
    d_inner = model['backbone.layers.0.mixer.norm.weight'].shape[0]
    # headdim = d_inner / nheads
    headdim = d_inner // nheads
    # conv_dim from conv1d.weight, d_state = (conv_dim - d_inner) / 2
    conv_dim = model['backbone.layers.0.mixer.conv1d.weight'].shape[0]
    d_state = (conv_dim - d_inner) // 2
    # d_conv from conv1d.weight
    d_conv = model['backbone.layers.0.mixer.conv1d.weight'].shape[2]

    d_in_proj = 2 * d_inner + 2 * d_state + nheads

    shared_classifier = torch.equal(model['backbone.embedding.weight'], model['lm_head.weight'])

    print(f"writing header")
    print(f"  n_layers: {n_layers}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model (dim): {d_model}")
    print(f"  d_inner: {d_inner}")
    print(f"  d_state: {d_state}")
    print(f"  d_conv: {d_conv}")
    print(f"  headdim: {headdim}")
    print(f"  nheads: {nheads}")
    print(f"  shared classifier: {shared_classifier}")

    # write the params: n_layers, vocab_size, dim, d_inner, d_state, d_conv, headdim, shared_classifier
    # Note: nheads is computed (d_inner / headdim), rounded_vocab_size is computed
    header = struct.pack('iiiiiiii', n_layers, vocab_size, d_model,
                         d_inner, d_state, d_conv, headdim, int(shared_classifier))
    out_file.write(header)

    # pad the rest with zeros
    pad = 256 - out_file.tell()
    assert pad >= 0
    out_file.write(b'\0' * pad)

    '''
    Mamba2 model structure example:
    backbone.embedding.weight - [vocab_size, d_model]
    backbone.layers.0.mixer.in_proj.weight - [d_in_proj, d_model]
    backbone.layers.0.mixer.conv1d.weight - [conv_dim, 1, d_conv]
    backbone.layers.0.mixer.conv1d.bias - [conv_dim]
    backbone.layers.0.mixer.dt_bias - [nheads]
    backbone.layers.0.mixer.A_log - [nheads]
    backbone.layers.0.mixer.D - [nheads]
    backbone.layers.0.mixer.norm.weight - [d_inner]
    backbone.layers.0.mixer.out_proj.weight - [d_model, d_inner]
    backbone.layers.0.norm.weight - [d_model]
    backbone.norm_f.weight - [d_model]
    lm_head.weight - [vocab_size, d_model]
    '''

    # Convert A_log to A = -exp(A_log) for faster inference
    for n in range(n_layers):
        A_log = model.pop(f'backbone.layers.{n}.mixer.A_log').float()  # convert to float32
        model[f'backbone.layers.{n}.mixer.A'] = -torch.exp(A_log)

    # write the weights

    # write the embedding weights
    write_weights(out_file, model, 'backbone.embedding.weight')

    # layer weights
    write_layer_weights(out_file, model, 'backbone.layers.%d.mixer.in_proj.weight', n_layers)

    # conv1d weight needs reshaping: (conv_dim, 1, d_conv) -> (conv_dim, d_conv)
    for n in range(n_layers):
        conv_weight = model[f'backbone.layers.{n}.mixer.conv1d.weight'].squeeze(1)
        print(f"writing backbone.layers.{n}.mixer.conv1d.weight {list(conv_weight.shape)[::-1]}")
        serialize_fp32(out_file, conv_weight)

    write_layer_weights(out_file, model, 'backbone.layers.%d.mixer.conv1d.bias', n_layers)
    write_layer_weights(out_file, model, 'backbone.layers.%d.mixer.dt_bias', n_layers)
    write_layer_weights(out_file, model, 'backbone.layers.%d.mixer.A', n_layers)
    write_layer_weights(out_file, model, 'backbone.layers.%d.mixer.D', n_layers)
    write_layer_weights(out_file, model, 'backbone.layers.%d.mixer.norm.weight', n_layers)
    write_layer_weights(out_file, model, 'backbone.layers.%d.mixer.out_proj.weight', n_layers)
    write_layer_weights(out_file, model, 'backbone.layers.%d.norm.weight', n_layers)

    # final norm weights
    write_weights(out_file, model, 'backbone.norm_f.weight')

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
        if not os.path.exists(filepath):
            filepath = os.path.join(path, 'model.safetensors')
    else:
        filepath = path

    if filepath.endswith('.safetensors'):
        from safetensors.torch import load_file
        model = load_file(filepath)
    else:
        model = torch.load(filepath, map_location='cpu')

    # get the path to the config file
    if os.path.isdir(path):
        config_path = os.path.join(path, 'config.json')
    else:
        config_path = os.path.join(os.path.dirname(path), 'config.json')
    # load the config
    with open(config_path) as f:
        config = json.load(f)
    config = argparse.Namespace(**config)

    return model, config


def get_model_from_huggingface(model_name: str):
    """Download model from HuggingFace and get the path to the model directory.
    The model name can be one of the following:
        'state-spaces/mamba2-130m'
        'state-spaces/mamba2-370m'
        'state-spaces/mamba2-780m'
        'state-spaces/mamba2-1.3b'
        'state-spaces/mamba2-2.7b'
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=model_name)
    return local_dir

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="model name or folder where the model files are located", default="state-spaces/mamba2-130m")
    parser.add_argument("destination", type=str, help="full path to the output file", default="model.bin")
    args = parser.parse_args()

    # if the source starts with 'state-spaces/mamba2-' then load the model from HuggingFace
    if args.source.startswith('state-spaces/mamba2-'):
        model_path = get_model_from_huggingface(args.source)
    else:
        model_path = args.source

    model, config = load_model(model_path)

    if model is None:
        parser.error("Can't load input model!")

    # export
    model_export(model, config, args.destination)
