import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

HAGGINGFACE_TOKEN = "hf_xxxxxx"
model_id = "xxxxxx"
device = "xxxxxx"

# プロンプト
prompt = "xxxxxx"

# パイプラインの作成
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=HAGGINGFACE_TOKEN)
pipe = pipe.to(device)

# パイプラインの実行
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=20).images[0]

image.save("test.png")
