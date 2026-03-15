"""
FreeInpaint Gradio App
======================
A Gradio-based interactive demo for FreeInpaint image inpainting.

Users can:
1. Upload an image
2. Draw a mask on the area to be inpainted (white brush = area to inpaint)
3. Enter a text prompt describing what to generate
4. Optionally enable FreeInpaint optimization (requires reward model paths)
5. View the inpainted result

Usage:
    python app.py
    python app.py --model stable-diffusion-v1-5/stable-diffusion-inpainting
    python app.py --share  # creates a public Gradio link
"""

import argparse
import os
import sys

import gradio as gr
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusers import DPMSolverMultistepScheduler
from examples.freeinpaint.pipe.pipeline_stable_diffusion_inpaint_optno_guidance import (
    StableDiffusionInpaintOptNoGuidancePipeline,
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_pil_rgb(img):
    """Convert various image formats to a PIL RGB image."""
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.convert("RGB")


def _resize_to_multiple(image: Image.Image, multiple: int = 8) -> Image.Image:
    """Resize image so that both dimensions are multiples of *multiple*."""
    w, h = image.size
    w = (w // multiple) * multiple
    h = (h // multiple) * multiple
    w = max(w, multiple)
    h = max(h, multiple)
    return image.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    enable_freeinpaint: bool,
    image_reward_path: str,
    clip_model_path: str,
    inpaint_reward_config: str,
    inpaint_reward_model: str,
    opt_noise_steps: int,
    reward_guidance_scale: float,
    overall_reward_scale: float,
    prompt_reward_scale: float,
    harmonic_reward_scale: float,
    self_attn_loss_scale: float,
):
    """Load the inpainting pipeline."""
    global pipe

    if not model_path.strip():
        return "❌ Please provide a model path or Hugging Face model ID."

    try:
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionInpaintOptNoGuidancePipeline.from_pretrained(
            model_path.strip(),
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        # Freeze parameters – the pipeline is used inference-only
        for component in [pipe.unet, pipe.vae, pipe.text_encoder]:
            for param in component.parameters():
                param.requires_grad = False

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)

        # ------------------------------------------------------------------
        # FreeInpaint optimisation settings
        # ------------------------------------------------------------------
        pipe.opt_noise_steps = opt_noise_steps if enable_freeinpaint else 0
        pipe.initno_lr = 1e-1
        pipe.self_attn_loss_scale = self_attn_loss_scale

        if enable_freeinpaint:
            pipe.reward_guidance_scale = reward_guidance_scale
            pipe.overall_reward_scale = overall_reward_scale
            pipe.prompt_reward_scale = prompt_reward_scale
            pipe.harmonic_reward_scale = harmonic_reward_scale

            # Load reward models
            from examples.freeinpaint.metrics.guidance import ImageRewardScore, PromptRewardScore
            from examples.freeinpaint.metrics.prefpaint import InpaintReward

            pipe.overall_reward = ImageRewardScore(image_reward_path, device, dtype=dtype)
            pipe.prompt_reward = PromptRewardScore(clip_model_path, device, dtype=dtype)
            harmonic_reward = InpaintReward(inpaint_reward_config, device, dtype=dtype)
            harmonic_reward = harmonic_reward.load_model(harmonic_reward, inpaint_reward_model)
            pipe.harmonic_reward = harmonic_reward
        else:
            # Disable all reward / optimisation guidance
            pipe.reward_guidance_scale = 0

        return f"✅ Model loaded from '{model_path}' on {device}."

    except Exception as exc:
        pipe = None
        return f"❌ Failed to load model: {exc}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inpaint(
    image_dict,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
):
    """Run inpainting on the uploaded image + drawn mask."""
    if pipe is None:
        return None, "❌ No model loaded. Please load a model first."

    if image_dict is None:
        return None, "❌ Please upload an image."

    # ------------------------------------------------------------------
    # Extract image and mask from the Gradio sketch component.
    # When type="pil" and tool="sketch", Gradio returns a dict:
    #   {"image": PIL.Image, "mask": PIL.Image}
    # where mask pixels are WHITE on the area the user painted.
    # ------------------------------------------------------------------
    if isinstance(image_dict, dict):
        image = image_dict.get("image")
        mask = image_dict.get("mask")
    else:
        # Fallback: the sketch tool sometimes returns just the composite image
        image = image_dict
        mask = None

    if image is None:
        return None, "❌ Could not read image. Please upload an image and draw a mask."

    image = _to_pil_rgb(image)

    if mask is None:
        return None, "❌ No mask drawn. Please draw on the area you want to inpaint."

    mask = _to_pil_rgb(mask)

    # Verify that the mask is non-empty (user actually drew something)
    mask_arr = np.array(mask)
    if mask_arr.max() == 0:
        return None, "❌ No mask drawn. Please draw on the area you want to inpaint."

    # Ensure image and mask have the same size and are multiples of 8
    target_size = _resize_to_multiple(image, multiple=8).size
    image = image.resize(target_size, Image.LANCZOS)
    mask = mask.resize(target_size, Image.NEAREST)

    generator = torch.Generator(device=device).manual_seed(int(seed))

    try:
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt if (negative_prompt and negative_prompt.strip()) else None,
            generator=generator,
        ).images[0]
    except Exception as exc:
        return None, f"❌ Inference failed: {exc}"

    return result, "✅ Done!"


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def build_demo():
    with gr.Blocks(title="FreeInpaint Demo") as demo:
        gr.Markdown(
            """
# 🎨 FreeInpaint – Interactive Inpainting Demo

**How to use:**
1. Enter a model path (or Hugging Face model ID) and click **Load Model**.
2. Upload an image and **draw on the area** you want to replace (white brush = inpaint region).
3. Describe what to generate in the prompt and click **Inpaint ▶**.

> *Default model:* `stable-diffusion-v1-5/stable-diffusion-inpainting`
            """
        )

        # ------------------------------------------------------------------ #
        # Model loading                                                        #
        # ------------------------------------------------------------------ #
        with gr.Accordion("⚙️ Model & Settings", open=True):
            with gr.Row():
                model_path_input = gr.Textbox(
                    value="stable-diffusion-v1-5/stable-diffusion-inpainting",
                    label="Model path / HuggingFace ID",
                    placeholder="e.g. stable-diffusion-v1-5/stable-diffusion-inpainting",
                    scale=4,
                )
                load_btn = gr.Button("Load Model", variant="primary", scale=1)
            load_status = gr.Textbox(
                label="Model status",
                interactive=False,
                placeholder="Model not loaded",
            )

        # ------------------------------------------------------------------ #
        # FreeInpaint optimisation (advanced)                                  #
        # ------------------------------------------------------------------ #
        with gr.Accordion("🚀 FreeInpaint Optimisation (optional)", open=False):
            gr.Markdown(
                "Enable FreeInpaint's reward-guided optimisation for better quality. "
                "Requires pre-downloaded reward model checkpoints."
            )
            enable_freeinpaint = gr.Checkbox(label="Enable FreeInpaint optimisation", value=False)
            with gr.Row():
                image_reward_path = gr.Textbox(label="ImageReward model path", placeholder="path/to/ImageReward")
                clip_model_path = gr.Textbox(
                    label="CLIP model path",
                    placeholder="path/to/openai-clip-vit-large-patch14",
                )
            with gr.Row():
                inpaint_reward_config = gr.Textbox(
                    label="InpaintReward config path",
                    placeholder="path/to/prefpaintReward/configs.yaml",
                )
                inpaint_reward_model = gr.Textbox(
                    label="InpaintReward model path",
                    placeholder="path/to/prefpaintReward/prefpaintReward.pt",
                )
            with gr.Row():
                opt_noise_steps = gr.Slider(0, 100, value=40, step=1, label="InitNO optimisation steps")
                reward_guidance_scale = gr.Slider(0, 50, value=25, step=1, label="Reward guidance scale")
            with gr.Row():
                overall_reward_scale = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Overall reward scale")
                prompt_reward_scale = gr.Slider(0.0, 10.0, value=4.0, step=0.1, label="Prompt reward scale")
            with gr.Row():
                harmonic_reward_scale = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Harmonic reward scale")
                self_attn_loss_scale = gr.Slider(0.0, 20.0, value=5.0, step=0.5, label="Self-attention loss scale")

        # ------------------------------------------------------------------ #
        # Main inpainting UI                                                   #
        # ------------------------------------------------------------------ #
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    source="upload",
                    tool="sketch",
                    type="pil",
                    label="🖼️ Upload image & draw mask (paint white over the area to inpaint)",
                    brush_radius=20,
                )
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe what to generate in the masked area…",
                    lines=2,
                )
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What to avoid in the output…",
                    lines=1,
                )
                with gr.Row():
                    num_steps = gr.Slider(10, 100, value=50, step=1, label="Inference steps")
                    guidance_scale = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
                seed_input = gr.Number(value=42, label="Seed", precision=0)
                inpaint_btn = gr.Button("Inpaint ▶", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="🖼️ Inpainted result", type="pil")
                status_output = gr.Textbox(label="Status", interactive=False)

        # ------------------------------------------------------------------ #
        # Event handlers                                                        #
        # ------------------------------------------------------------------ #
        load_btn.click(
            fn=load_model,
            inputs=[
                model_path_input,
                enable_freeinpaint,
                image_reward_path,
                clip_model_path,
                inpaint_reward_config,
                inpaint_reward_model,
                opt_noise_steps,
                reward_guidance_scale,
                overall_reward_scale,
                prompt_reward_scale,
                harmonic_reward_scale,
                self_attn_loss_scale,
            ],
            outputs=[load_status],
        )

        inpaint_btn.click(
            fn=run_inpaint,
            inputs=[
                image_input,
                prompt_input,
                negative_prompt_input,
                num_steps,
                guidance_scale,
                seed_input,
            ],
            outputs=[output_image, status_output],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreeInpaint Gradio demo")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID or local path to pre-load at startup",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--server-port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server hostname")
    args = parser.parse_args()

    demo = build_demo()

    # Optionally pre-load the model at startup so the UI is ready immediately
    if args.model:
        status = load_model(
            model_path=args.model,
            enable_freeinpaint=False,
            image_reward_path="",
            clip_model_path="",
            inpaint_reward_config="",
            inpaint_reward_model="",
            opt_noise_steps=0,
            reward_guidance_scale=0,
            overall_reward_scale=0.1,
            prompt_reward_scale=4.0,
            harmonic_reward_scale=1.0,
            self_attn_loss_scale=5.0,
        )
        print(status)

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )
