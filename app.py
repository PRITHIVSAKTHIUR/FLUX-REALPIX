import spaces
import gradio as gr
import torch
from PIL import Image
from diffusers import DiffusionPipeline
import random
import uuid
from typing import Tuple
import numpy as np
import time
import zipfile

DESCRIPTION = """## flux realism
"""

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

MAX_SEED = np.iinfo(np.int32).max

base_model = "black-forest-labs/FLUX.1-dev"
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

lora_repo = "strangerzonehf/Flux-Super-Realism-LoRA"
trigger_word = "Super Realism"

pipe.load_lora_weights(lora_repo)
pipe.to("cuda")

style_list = [
    {
        "name": "3840 x 2160",
        "prompt": "hyper-realistic 8K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "",
    },
    {
        "name": "2560 x 1440",
        "prompt": "hyper-realistic 4K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "",
    },
    {
        "name": "HD+",
        "prompt": "hyper-realistic 2K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "",
    },
    {
        "name": "Style Zero",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
DEFAULT_STYLE_NAME = "3840 x 2160"
STYLE_NAMES = list(styles.keys())

def apply_style(style_name: str, positive: str) -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n

@spaces.GPU
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    style_name: str = DEFAULT_STYLE_NAME,
    num_inference_steps: int = 30,
    num_images: int = 1,
    zip_images: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    positive_prompt, style_negative_prompt = apply_style(style_name, prompt)
    
    if use_negative_prompt:
        final_negative_prompt = style_negative_prompt + " " + negative_prompt
    else:
        final_negative_prompt = style_negative_prompt
    
    final_negative_prompt = final_negative_prompt.strip()
    
    if trigger_word:
        positive_prompt = f"{trigger_word} {positive_prompt}"
    
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    start_time = time.time()
    
    images = pipe(
        prompt=positive_prompt,
        negative_prompt=final_negative_prompt if final_negative_prompt else None,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        generator=generator,
        output_type="pil",
    ).images
    
    end_time = time.time()
    duration = end_time - start_time
    
    image_paths = [save_image(img) for img in images]
    
    zip_path = None
    if zip_images:
        zip_name = str(uuid.uuid4()) + ".zip"
        with zipfile.ZipFile(zip_name, 'w') as zipf:
            for i, img_path in enumerate(image_paths):
                zipf.write(img_path, arcname=f"Img_{i}.png")
        zip_path = zip_name
    
    return image_paths, seed, f"{duration:.2f}", zip_path

examples = [
    "Super Realism, High-resolution photograph, woman, UHD, photorealistic, shot on a Sony A7III --chaos 20 --ar 1:2 --style raw --stylize 250",
    "Woman in a red jacket, snowy, in the style of hyper-realistic portraiture, caninecore, mountainous vistas, timeless beauty, palewave, iconic, distinctive noses --ar 72:101 --stylize 750 --v 6",
    "Super Realism, Headshot of handsome young man, wearing dark gray sweater with buttons and big shawl collar, brown hair and short beard, serious look on his face, black background, soft studio lighting, portrait photography --ar 85:128 --v 6.0 --style",
    "Super-realism, Purple Dreamy, a medium-angle shot of a young woman with long brown hair, wearing a pair of eye-level glasses, stands in front of a backdrop of purple and white lights. The womans eyes are closed, her lips are slightly parted, as if she is looking up at the sky. Her hair is cascading over her shoulders, framing her face. She is wearing a sleeveless top, adorned with tiny white dots, and a gold chain necklace around her neck. Her left earrings are dangling from her ears, adding a pop of color to the scene."
]

css = '''
.gradio-container {
    max-width: 590px !important;
    margin: 0 auto !important;
}
h1 {
    text-align: center;
}
footer {
    visibility: hidden;
}
'''

with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        prompt = gr.Text(
            label="Prompt",
            show_label=False,
            max_lines=1,
            placeholder="Enter your prompt",
            container=False,
        )
        run_button = gr.Button("Run", scale=0, variant="primary")
    result = gr.Gallery(label="Result", columns=1, show_label=False, preview=True)

    with gr.Accordion("Additional Options", open=False):
        style_selection = gr.Dropdown(
            label="Quality Style",
            choices=STYLE_NAMES,
            value=DEFAULT_STYLE_NAME,
            interactive=True,
        )
        use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False)
        negative_prompt = gr.Text(
            label="Negative prompt",
            max_lines=1,
            placeholder="Enter a negative prompt",
            visible=False,
        )
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row():
            width = gr.Slider(
                label="Width",
                minimum=512,
                maximum=2048,
                step=64,
                value=1024,
                                #value=1280,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=2048,
                step=64,
                value=1024,
                                #value=832,
            )
        guidance_scale = gr.Slider(
            label="Guidance Scale",
            minimum=0.1,
            maximum=20.0,
            step=0.1,
            value=3.0,
        )
        num_inference_steps = gr.Slider(
            label="Number of inference steps",
            minimum=1,
            maximum=40,
            step=1,
            value=30,
        )
        num_images = gr.Slider(
            label="Number of images",
            minimum=1,
            maximum=5,
            step=1,
            value=1,
        )
        zip_images = gr.Checkbox(label="Zip generated images", value=False)
        
        gr.Markdown("### Output Information")
        seed_display = gr.Textbox(label="Seed used", interactive=False)
        generation_time = gr.Textbox(label="Generation time (seconds)", interactive=False)
        zip_file = gr.File(label="Download ZIP")

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed_display, generation_time, zip_file],
        fn=generate,
        cache_examples=False,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    gr.on(
        triggers=[
            prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
            style_selection,
            num_inference_steps,
            num_images,
            zip_images,
        ],
        outputs=[result, seed_display, generation_time, zip_file],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=120).launch(mcp_server=True, ssr_mode=False, show_error=True)
