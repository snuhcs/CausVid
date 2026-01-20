from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os

from torch.profiler import profile, ProfilerActivity, record_function

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipeline = InferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)
print("Generator: ", sum(p.numel() for p in pipeline.generator.parameters()))
print("Text Encoder: ", sum(p.numel() for p in pipeline.text_encoder.parameters()))
print("VAE: ", sum(p.numel() for p in pipeline.vae.parameters()))

dataset = TextDataset(args.prompt_file_path)

sampled_noise = torch.randn(
    [1, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
)

os.makedirs(args.output_folder, exist_ok=True)

for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]

    torch.cuda.memory._record_memory_history()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        record_shapes=True,
        with_stack=True,
        with_modules=True,
        profile_memory=True,
    ) as prof:
        with record_function("autoregressive_inference"):
            video = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts
            )[0].permute(0, 2, 3, 1).cpu().numpy()

    torch.cuda.memory._dump_snapshot(os.path.join(args.output_folder, f"memory_snapshot_{prompt_index:03d}.json"))
    prof.export_chrome_trace(os.path.join(args.output_folder, f"output_{prompt_index:03d}.json"))

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{prompt_index:03d}.mp4"), fps=16)
