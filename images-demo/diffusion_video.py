from __future__ import annotations

import numpy as np

from modal import Image, Secret, Stub
from typing import List

cache_path = "/vol/cache"
model = "damo-vilab/text-to-video-ms-1.7b"
stub = Stub("damo_vilab")


def download_models():
    import torch
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    pipe.save_pretrained(cache_path, safe_serialization=True)


image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "diffusers[torch]>=0.15.1",
        "ftfy",
        "torchvision",
        "torch>=2.0",
        "transformers>=4.27.0",
        "triton",
        "safetensors",
        "opencv-python",
        "imageio[ffmpeg]",
        "openai",
    )
    .run_commands(
        [
            "apt-get update",
            "apt-get install --yes ffmpeg libsm6 libxext6",
        ]
    )
    .run_function(
        download_models,
        secrets=[Secret.from_name("huggingface-secret")],
    )
)
stub.image = image


def to_video(frames: list[np.ndarray], fps: int) -> str:
    import imageio
    import tempfile

    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = imageio.get_writer(out_file.name, format="FFMPEG", fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return out_file.name


@stub.function(gpu="A10G", timeout=900, secret=Secret.from_name("openai-secret"))
def run_inference(
    prompt: str,
    query: str,
    num_frames: int = 24,
    num_candidates: int = 3,
) -> List[bytes]:
    import openai
    import random
    import torch
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

    scenes = (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "you are an expert barista"},
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n"
                        "Only generate a list of steps delimited by a new line character "
                        "without any numbering or - at the beginning of each line, "
                        "for the query delimited by back ticks "
                        "NOTE: do not include a summary line at the beginning saying 'follow these steps'\n"
                        f"```{query}```\n"
                    ),
                },
            ],
        )
        .choices[0]
        .message.content.split("\n")
    )

    delim = "\n"
    print(
        f"going to generate {num_candidates} candidate videos for the following steps: \n{delim.join(scenes)}"
    )

    video_candidates = []
    for _ in range(num_candidates):
        seed = random.randint(0, 1000_000)
        all_frames = []
        for scene in scenes:
            generator = torch.Generator().manual_seed(seed)
            pipe = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float16,
                variant="fp16",
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()

            frames = pipe(
                scene,
                num_inference_steps=64,
                num_frames=num_frames,
                generator=generator,
            ).frames
            all_frames.extend(frames)
        video_path = to_video(all_frames, 8)
        print(f"wrote video to video_path: {video_path}")
        with open(video_path, "rb") as f:
            video_candidates.append(f.read())
    return video_candidates


@stub.local_entrypoint()
def entrypoint():
    prompt = (
        "You are generating prompts to a text-to-video model that takes a prompt and generates a video. "
        "Generate a 4 step description. "
        "Each step should be an input to the video generation model. "
        "All the 4 steps should describe a scene independent of each other. "
        "The model cannot take previous prompt for context. "
        "The model is not good at generating complex scenes, "
        "so the prompts should be very simple, with simple colors and plain background. "
        "Also, the prompts should be very short, at a maximum of 8 words each. "
        "Do not use words like 'steep'."
    )
    query = "how to make tea?"

    candidates = run_inference.call(prompt, query, num_frames=32)
    for num, candidate in enumerate(candidates):
        with open(f"/tmp/output_{num}.mp4", "wb") as f:
            f.write(candidate)
