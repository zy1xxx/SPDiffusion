from SPD_Pipeline import SPDiffusionPipeline
import torch
model_name="SG161222/RealVisXL_V4.0"
device="cuda"
pipe = SPDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16,variant='fp16', use_safetensors=False,safety_checker=None)
pipe=pipe.to(device)
generator = torch.Generator(device=device).manual_seed(2048)
image=pipe("A red book and a yellow vase",run_sdxl=True,generator=generator,cross_threshold=0.9,self_threshold=0.1,st_step=2).images[0]
image.save("result.png")