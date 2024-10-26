import argparse

import torch
from safetensors import safe_open


"""

model.embed_tokens.weight                       [128256, 2048]  
model.layers.0.input_layernorm.weight           [2048]
model.layers.0.mlp.down_proj.weight             [2048, 8192]
model.layers.0.mlp.gate_proj.weight             [8192, 2048]
model.layers.0.mlp.up_proj.weight               [8192, 2048]
model.layers.0.post_attention_layernorm.weight  [2048]
model.layers.0.self_attn.k_proj.weight          [512, 2048] 
model.layers.0.self_attn.o_proj.weight          [2048, 2048]
model.layers.0.self_attn.q_proj.weight          [2048, 2048]
model.layers.0.self_attn.v_proj.weight          [512, 2048] 
model.norm.weight   [2048]

"""

dtype_map = {
    "fp16": torch.float16,
    "fp32": torch.float32
}

def write_tensor(name, checkpoint, fout, dtype):
    print(f"\rConverting: {name}", end="")

    tensor = checkpoint.get_tensor(name)
    tensor = tensor.to(dtype_map[dtype])
    tensor_bytes = tensor.numpy().tobytes()
    fout.write(tensor_bytes)


def convert_llama32(dtype):
    print(f"Converting to dtype: {dtype}")
    fname = f"llama32-1B.{dtype}.bin"
    with safe_open("model.safetensors", framework="pt", device="cpu") as ckpt, open(fname, "wb") as fout:

        magic_no = b"llama32f"
        fout.write(magic_no)

        write_tensor("model.embed_tokens.weight", ckpt, fout, dtype)
        
        num_layers = 16
        for i in range(num_layers):
            write_tensor(f"model.layers.{i}.input_layernorm.weight", ckpt, fout , dtype)
            write_tensor(f"model.layers.{i}.self_attn.q_proj.weight", ckpt, fout, dtype)
            write_tensor(f"model.layers.{i}.self_attn.k_proj.weight", ckpt, fout, dtype)
            write_tensor(f"model.layers.{i}.self_attn.v_proj.weight", ckpt, fout, dtype)
            write_tensor(f"model.layers.{i}.self_attn.o_proj.weight", ckpt, fout, dtype)
            write_tensor(f"model.layers.{i}.post_attention_layernorm.weight", ckpt, fout, dtype)
            write_tensor(f"model.layers.{i}.mlp.gate_proj.weight", ckpt, fout, dtype)
            write_tensor(f"model.layers.{i}.mlp.up_proj.weight", ckpt, fout, dtype)
            write_tensor(f"model.layers.{i}.mlp.down_proj.weight", ckpt, fout, dtype)

        write_tensor("model.norm.weight", ckpt, fout, dtype)
    print(f"\nConversion complete: {fname}")



parser = argparse.ArgumentParser()
parser.add_argument("dtype", help="output dtype.", choices=dtype_map.keys())

args = parser.parse_args()
convert_llama32(args.dtype)
