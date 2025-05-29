import gradio as gr
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from transformers import pipeline
import torch

def load_canny_models():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def load_depth_models():
    depth_estimator = pipeline("depth-estimation")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe, depth_estimator

def load_inpaint_models():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint",
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def get_depth_map(image, depth_estimator):
    # Convert PIL image to RGB if it's not
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Get depth map
    depth_map = depth_estimator(image)["depth"]
    
    # Convert to numpy array and normalize
    depth_map = np.array(depth_map)
    depth_map = depth_map[:, :, None]
    depth_map = np.concatenate([depth_map, depth_map, depth_map], axis=2)
    
    # Convert to tensor and normalize
    depth_map = torch.from_numpy(depth_map).float() / 255.0
    depth_map = depth_map.permute(2, 0, 1).unsqueeze(0)
    
    return depth_map

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1]
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

def process_canny(image, low_threshold, high_threshold, prompt, negative_prompt, num_inference_steps, guidance_scale):
    # Convert to numpy array and apply Canny edge detection
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    # Generate new image
    pipe = load_canny_models()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]
    
    # Create image grid
    grid = make_image_grid([Image.fromarray(np.array(image)), canny_image, output], rows=1, cols=3)
    
    return canny_image, output, grid

def process_depth(image, prompt, negative_prompt, num_inference_steps, guidance_scale, strength):
    # Get depth map
    pipe, depth_estimator = load_depth_models()
    depth_map = get_depth_map(image, depth_estimator)
    
    # Generate new image
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        control_image=depth_map,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength
    ).images[0]
    
    # Create image grid
    grid = make_image_grid([image, output], rows=1, cols=2)
    
    return output, grid

def process_inpaint(image, mask, prompt, negative_prompt, num_inference_steps, guidance_scale, eta):
    # Resize images to 512x512
    image = image.resize((512, 512))
    mask = mask.resize((512, 512))
    
    # Create control image
    control_image = make_inpaint_condition(image, mask)
    
    # Generate new image
    pipe = load_inpaint_models()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        control_image=control_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        eta=eta
    ).images[0]
    
    # Create image grid
    grid = make_image_grid([image, mask, output], rows=1, cols=3)
    
    return output, grid

# Create the Gradio interface
with gr.Blocks(title="ControlNet Image Generation") as demo:
    gr.Markdown("# ControlNet Image Generation")
    gr.Markdown("⚠️ Running on CPU - Generation will be slower")
    
    with gr.Tabs():
        with gr.TabItem("Canny Edge Detection"):
            with gr.Row():
                with gr.Column():
                    canny_input = gr.Image(label="Input Image", type="pil")
                    
                    with gr.Accordion("Parameters", open=True):
                        low_threshold = gr.Slider(0, 255, value=100, label="Low Threshold")
                        high_threshold = gr.Slider(0, 255, value=200, label="High Threshold")
                        prompt = gr.Textbox(
                            label="Prompt",
                            value="the mona lisa, masterpiece, best quality, extremely detailed"
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
                        )
                        num_inference_steps = gr.Slider(20, 50, value=30, label="Number of Inference Steps")
                        guidance_scale = gr.Slider(1.0, 20.0, value=7.5, label="Guidance Scale")
                    
                    canny_generate_btn = gr.Button("Generate Image")
                
                with gr.Column():
                    canny_output = gr.Image(label="Canny Edge Detection", type="pil")
                    canny_generated = gr.Image(label="Generated Image", type="pil")
                    canny_grid = gr.Image(label="Image Grid", type="pil")
            
            canny_generate_btn.click(
                fn=process_canny,
                inputs=[
                    canny_input,
                    low_threshold,
                    high_threshold,
                    prompt,
                    negative_prompt,
                    num_inference_steps,
                    guidance_scale
                ],
                outputs=[canny_output, canny_generated, canny_grid]
            )
        
        with gr.TabItem("Depth Image-to-Image"):
            with gr.Row():
                with gr.Column():
                    depth_input = gr.Image(label="Input Image", type="pil")
                    
                    with gr.Accordion("Parameters", open=True):
                        depth_prompt = gr.Textbox(
                            label="Prompt",
                            value="lego batman and robin, masterpiece, best quality, extremely detailed"
                        )
                        depth_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
                        )
                        depth_num_inference_steps = gr.Slider(20, 50, value=30, label="Number of Inference Steps")
                        depth_guidance_scale = gr.Slider(1.0, 20.0, value=7.5, label="Guidance Scale")
                        strength = gr.Slider(0.0, 1.0, value=0.8, label="Strength")
                    
                    depth_generate_btn = gr.Button("Generate Image")
                
                with gr.Column():
                    depth_generated = gr.Image(label="Generated Image", type="pil")
                    depth_grid = gr.Image(label="Image Grid", type="pil")
            
            depth_generate_btn.click(
                fn=process_depth,
                inputs=[
                    depth_input,
                    depth_prompt,
                    depth_negative_prompt,
                    depth_num_inference_steps,
                    depth_guidance_scale,
                    strength
                ],
                outputs=[depth_generated, depth_grid]
            )
        
        with gr.TabItem("Inpainting"):
            with gr.Row():
                with gr.Column():
                    inpaint_input = gr.Image(label="Input Image", type="pil")
                    inpaint_mask = gr.Image(label="Mask Image", type="pil")
                    
                    with gr.Accordion("Parameters", open=True):
                        inpaint_prompt = gr.Textbox(
                            label="Prompt",
                            value="corgi face with large ears, detailed, pixar, animated, disney"
                        )
                        inpaint_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
                        )
                        inpaint_num_inference_steps = gr.Slider(20, 50, value=20, label="Number of Inference Steps")
                        inpaint_guidance_scale = gr.Slider(1.0, 20.0, value=7.5, label="Guidance Scale")
                        eta = gr.Slider(0.0, 1.0, value=1.0, label="Eta")
                    
                    inpaint_generate_btn = gr.Button("Generate Image")
                
                with gr.Column():
                    inpaint_generated = gr.Image(label="Generated Image", type="pil")
                    inpaint_grid = gr.Image(label="Image Grid", type="pil")
            
            inpaint_generate_btn.click(
                fn=process_inpaint,
                inputs=[
                    inpaint_input,
                    inpaint_mask,
                    inpaint_prompt,
                    inpaint_negative_prompt,
                    inpaint_num_inference_steps,
                    inpaint_guidance_scale,
                    eta
                ],
                outputs=[inpaint_generated, inpaint_grid]
            )

if __name__ == "__main__":
    demo.launch() 