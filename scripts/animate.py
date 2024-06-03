import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionImg2ImgPipeline

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging

from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat
from clip_interrogator import Config, Interrogator

import csv, pdb, glob, math, gc
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config  = OmegaConf.load(args.config)
    samples = []

    # create validation pipeline
    logging.set_verbosity_error()
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)
        model_config.C = model_config.get("C", args.C)

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()

        # load controlnet model
        controlnet = controlnet_images = None
        video_as_control = False
        controlnet_image_index = model_config.get("controlnet_image_indexs", [0])
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""

            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()

            image_paths = model_config.controlnet_images if model_config.controlnet_images is not None else []
            if isinstance(image_paths, str): image_paths = [image_paths]

            video_path = None
            video_extension = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.m4v', '.webm', '.mpeg', '.mpg')
            for path in image_paths:
                if (os.path.splitext(path)[1].lower() in video_extension):
                    video_as_control = True
                    video_path = path
                    break

            if video_as_control: image_paths = [video_path]
            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0),
                    ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x

            if video_as_control:
                cap = cv2.VideoCapture(video_path)
                video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                model_config.L = video_length
                controlnet_image_index = range(video_length)
                controlnet_images = []
                not_valid = []
                for i in range(model_config.L):
                    ret, frame = cap.read()
                    if not ret:
                        not_valid.append(i)
                        continue

                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    controlnet_images.append(pil_image)
                controlnet_image_index = [val if val >= 0 else video_length + val for val in controlnet_image_index]
                controlnet_image_index = [idx for idx in controlnet_image_index if idx not in not_valid]

                if args.stride > 1:
                    controlnet_image_index = controlnet_image_index[::args.stride]
                    controlnet_images = controlnet_images[::args.stride]
            else:
                controlnet_image_index = [val if val >= 0 else model_config.L + val for val in controlnet_image_index]
                controlnet_images = [Image.open(path).convert("RGB") for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                time_str = datetime.datetime.now().strftime("T%H-%M-%S")
                image.save(f"{savedir}/control_images/{i}_{time_str}.png")

            # transform image style
            if model_config.get("dreambooth_path", "") != "":
                print(f"transform controlnet images with {model_config.dreambooth_path} ...")
                img2img_seed = model_config.get("seed", [-1])
                if img2img_seed[0] != -1: torch.manual_seed(img2img_seed[0])
                img2img_n_prompt = list(model_config.n_prompt)[0]

                # image to text
                image_prompts = []
                clip_config = Config(clip_model_name="ViT-L-14/openai")
                clip_config.apply_low_vram_defaults()
                ci = Interrogator(clip_config)
                for image in controlnet_images:
                    image_prompts.append(ci.interrogate_fast(image))
                del ci, clip_config
                torch.cuda.empty_cache()
                gc.collect()

                with open(f"{savedir}/prompts.txt", 'w') as prompt_file:
                    prompt_file.write("\n".join(image_prompts))

                # build img2img
                img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.pretrained_model_path).to("cuda")
                img2img_pipe = load_weights(img2img_pipe, dreambooth_model_path=model_config.dreambooth_path).to("cuda")
                os.makedirs(os.path.join(savedir, "transformed_images"), exist_ok=True)

                for i, image in enumerate(controlnet_images):
                    image = image.resize((model_config.W, model_config.H))
                    transformed_image = img2img_pipe(
                        prompt=image_prompts[i], image=image, strength=0.3, negative_prompt=img2img_n_prompt,
                        num_inference_steps=model_config.steps, guidance_scale=model_config.guidance_scale).images[0]
                    time_str = datetime.datetime.now().strftime("T%H-%M-%S")
                    transformed_image.save(f"{savedir}/transformed_images/{i}_{time_str}.png")
                    controlnet_images[i] = transformed_image

                del img2img_pipe
                torch.cuda.empty_cache()
                gc.collect()

            controlnet_images = [image_norm(image_transforms(image)) for image in controlnet_images]
            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")

                # split to batch to save memory
                batch_size = 16
                if (controlnet_images.shape[0] > batch_size):
                    dataloader = DataLoader(controlnet_images, batch_size=batch_size)
                    encoded_images = []
                    with torch.no_grad():
                        for batch in dataloader:
                            batch_encoded = vae.encode(batch * 2. - 1.).latent_dist.sample() * 0.18215
                            encoded_images.append(batch_encoded)

                    encoded_images = torch.cat(encoded_images, dim=0)
                    controlnet_images = rearrange(encoded_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)
                else:
                    controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                    controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ).to("cuda")

        prompts      = model_config.prompt
        n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt

        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

        config[model_idx].random_seed = []
        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):

            # manually set random seed for reproduction
            if random_seed != -1: torch.manual_seed(random_seed)
            else: torch.seed()
            config[model_idx].random_seed.append(torch.initial_seed())

            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            sample = pipeline(
                prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = model_config.steps,
                guidance_scale      = model_config.guidance_scale,
                width               = model_config.W,
                height              = model_config.H,
                video_length        = model_config.L,
                context_frames      = model_config.C,

                controlnet_images = controlnet_images,
                controlnet_image_index = controlnet_image_index,
                video_control = video_as_control,
                stride = args.stride,
            ).videos
            samples.append(sample)

            prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
            print(f"save to {savedir}/sample/{prompt}.gif")

            sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference-config",      type=str, default="configs/inference/inference-v3.yaml")
    parser.add_argument("--config",                type=str, required=True)

    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--C", type=int, default=16, help="module context length")
    parser.add_argument("--stride", type=int, default=1, help="stride while video control")

    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    main(args)
