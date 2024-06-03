from diffusers import StableDiffusionImg2ImgPipeline
from clip_interrogator import Config, Interrogator
from PIL import Image
import os, argparse
import datetime


def transform_images(model_path, controlnet_images, savedir, prompt="", strength=0.5):
    image_prompts = image_to_text(controlnet_images, savedir)
    # 加载Stable Diffusion Img2Img模型
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path)
    os.makedirs(os.path.join(savedir, "transformed_images"), exist_ok=True)

    for i, image in enumerate(controlnet_images):
        transformed_image = img2img_pipe(prompt=image_prompts[i], image=image, strength=strength).images[0]
        time_str = datetime.datetime.now().strftime("T%H-%M-%S")
        transformed_image.save(f"{savedir}/transformed_images/{i}_{time_str}.png")
        controlnet_images[i] = transformed_image

def image_to_text(controlnet_images, savedir):
    image_prompts = []
    clip_config = Config(clip_model_name="ViT-L-14/openai")
    clip_config.apply_low_vram_defaults()
    ci = Interrogator(clip_config)
    for image in controlnet_images:
        image_prompts.append(ci.interrogate(image))
    with open(f"{savedir}/prompts.txt", 'w') as prompt_file:
        prompt_file.write("\n".join(image_prompts))
    return image_prompts

def read_images(paths):
    images = []
    if isinstance(paths, str):  # 处理单个路径
        if os.path.isdir(paths):  # 如果是文件夹
            for filename in os.listdir(paths):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # 确保只读取图像文件
                    img_path = os.path.join(paths, filename)
                    images.append(Image.open(img_path))
        else:  # 如果是单个文件
            images.append(Image.open(paths))
    elif isinstance(paths, list):  # 处理文件路径列表
        for path in paths:
            images.append(Image.open(path))
    else:
        raise ValueError("paths must be a string or a list of strings.")

    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform controlnet images using a pre-trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--image_path", type=str, required=True, nargs="+", help="Path to the input image.")
    parser.add_argument("--savedir", type=str, required=True, help="Directory to save the transformed images.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for the transformation.")
    parser.add_argument("--strength", type=float, default=0.5, help="Strength of the transformation.")

    args = parser.parse_args()

    # 读取输入图像
    controlnet_images = read_images(args.image_path)

    # 调用转换函数
    transform_images(args.model_path, controlnet_images, args.savedir, args.prompt, args.strength)
