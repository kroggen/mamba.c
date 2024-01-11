import os
import struct

from transformers import AutoTokenizer

def export():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

    print(f"vocab size: {tokenizer.vocab_size} - BOS: {tokenizer.bos_token} - EOS: {tokenizer.eos_token}")
    print(f"unk token: {tokenizer.unk_token} - pad token: {tokenizer.pad_token} - mask token: {tokenizer.mask_token}")

    # get all the tokens
    tokens = []
    for i in range(50277):        # tokenizer.vocab_size is returning 50254, but ...
        t = tokenizer.decode([i]) # ... there are 50277 tokens in the tokenizer.json file
        b = t.encode('utf-8')     # bytes of this token, utf-8 encoded
        tokens.append(b)

    # record the max token length
    max_token_length = max(len(t) for t in tokens)

    # write to a binary file
    tokenizer_bin = "tokenizer.bin"
    with open(tokenizer_bin, 'wb') as f:
        # write header: text "MbTk" as integer
        f.write(struct.pack("I", 0x4d62546b))
        # write version
        f.write(struct.pack("I", 1))
        # write the number of tokens
        f.write(struct.pack("I", len(tokens)))
        # write the max token length
        f.write(struct.pack("I", max_token_length))
        for bytes in tokens:
            # write the length of the token
            f.write(struct.pack("I", len(bytes)))
            # write the token text
            f.write(bytes)

if __name__ == "__main__":
    export()
