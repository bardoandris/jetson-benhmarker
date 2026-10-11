import torch
from transformers import AutoModelForCausalLM

moondream = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    device_map={"": "cuda"},
)
moondream.compile()