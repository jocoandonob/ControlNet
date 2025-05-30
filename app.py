import gradio as gr
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    UniPCMultistepScheduler
)
from transformers import pipeline
import torch
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from controlnet_aux import OpenposeDetector

def create_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def load_model_with_retry(model_class, model_id, **kwargs):
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(
                model_id,
                **kwargs,
                use_auth_token=False,
                local_files_only=False,
                resume_download=True,
                max_retries=3
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to load model {model_id} after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2

def load_canny_models():
    try:
        controlnet = load_model_with_retry(
            ControlNetModel,
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe = load_model_with_retry(
            StableDiffusionControlNetPipeline,
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe
    except Exception as e:
        raise Exception(f"Error loading Canny models: {str(e)}")

def load_depth_models():
    try:
        depth_estimator = pipeline("depth-estimation")
        controlnet = load_model_with_retry(
            ControlNetModel,
            "lllyasviel/control_v11f1p_sd15_depth",
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe = load_model_with_retry(
            StableDiffusionControlNetImg2ImgPipeline,
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe, depth_estimator
    except Exception as e:
        raise Exception(f"Error loading depth models: {str(e)}")

def load_inpaint_models():
    try:
        controlnet = load_model_with_retry(
            ControlNetModel,
            "lllyasviel/control_v11p_sd15_inpaint",
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe = load_model_with_retry(
            StableDiffusionControlNetInpaintPipeline,
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe
    except Exception as e:
        raise Exception(f"Error loading inpainting models: {str(e)}")

def load_sdxl_models():
    try:
        controlnet = load_model_with_retry(
            ControlNetModel,
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        vae = load_model_with_retry(
            AutoencoderKL,
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe = load_model_with_retry(
            StableDiffusionXLControlNetPipeline,
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe
    except Exception as e:
        raise Exception(f"Error loading SDXL models: {str(e)}")

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

def process_guess(image, low_threshold, high_threshold, guidance_scale):
    # Convert to numpy array and apply Canny edge detection
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    # Generate new image
    pipe = load_canny_models()
    output = pipe(
        prompt="",  # Empty prompt for guess mode
        image=canny_image,
        guess_mode=True,
        guidance_scale=guidance_scale
    ).images[0]
    
    # Create image grid
    grid = make_image_grid([Image.fromarray(np.array(image)), canny_image, output], rows=1, cols=3)
    
    return canny_image, output, grid

def process_sdxl(image, low_threshold, high_threshold, prompt, negative_prompt, num_inference_steps, guidance_scale, controlnet_conditioning_scale):
    # Convert to numpy array and apply Canny edge detection
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    # Generate new image
    pipe = load_sdxl_models()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale
    ).images[0]
    
    # Create image grid
    grid = make_image_grid([Image.fromarray(np.array(image)), canny_image, output], rows=1, cols=3)
    
    return canny_image, output, grid

def load_multi_controlnet_models():
    try:
        # Load pose detection model
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        
        # Load controlnet models
        controlnets = [
            load_model_with_retry(
                ControlNetModel,
                "thibaud/controlnet-openpose-sdxl-1.0",
                torch_dtype=torch.float32
            ),
            load_model_with_retry(
                ControlNetModel,
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=torch.float32,
                use_safetensors=True
            ),
        ]
        
        # Load VAE
        vae = load_model_with_retry(
            AutoencoderKL,
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        
        # Load pipeline
        pipe = load_model_with_retry(
            StableDiffusionXLControlNetPipeline,
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnets,
            vae=vae,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe, openpose
    except Exception as e:
        raise Exception(f"Error loading MultiControlNet models: {str(e)}")

def process_multi_controlnet(image, low_threshold, high_threshold, prompt, negative_prompt, num_inference_steps, guidance_scale, pose_scale, canny_scale, num_images):
    try:
        # Get pose detection
        pipe, openpose = load_multi_controlnet_models()
        pose_image = openpose(image)
        
        # Get Canny edges
        image_np = np.array(image)
        canny_image = cv2.Canny(image_np, low_threshold, high_threshold)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        canny_image = Image.fromarray(canny_image)
        
        # Resize images to 1024x1024
        pose_image = pose_image.resize((1024, 1024))
        canny_image = canny_image.resize((1024, 1024))
        
        # Generate images
        generator = torch.manual_seed(1)
        outputs = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=[pose_image, canny_image],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images,
            controlnet_conditioning_scale=[pose_scale, canny_scale]
        ).images
        
        # Create image grid
        grid_images = [image, canny_image, pose_image] + [img.resize((512, 512)) for img in outputs]
        grid = make_image_grid(grid_images, rows=2, cols=3)
        
        return canny_image, pose_image, outputs[0], grid
    except Exception as e:
        raise Exception(f"Error in MultiControlNet processing: {str(e)}")

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
        
        with gr.TabItem("Guess Mode"):
            with gr.Row():
                with gr.Column():
                    guess_input = gr.Image(label="Input Image", type="pil")
                    
                    with gr.Accordion("Parameters", open=True):
                        guess_low_threshold = gr.Slider(0, 255, value=100, label="Low Threshold")
                        guess_high_threshold = gr.Slider(0, 255, value=200, label="High Threshold")
                        guess_guidance_scale = gr.Slider(1.0, 20.0, value=3.0, label="Guidance Scale")
                    
                    guess_generate_btn = gr.Button("Generate Image")
                
                with gr.Column():
                    guess_output = gr.Image(label="Canny Edge Detection", type="pil")
                    guess_generated = gr.Image(label="Generated Image", type="pil")
                    guess_grid = gr.Image(label="Image Grid", type="pil")
            
            guess_generate_btn.click(
                fn=process_guess,
                inputs=[
                    guess_input,
                    guess_low_threshold,
                    guess_high_threshold,
                    guess_guidance_scale
                ],
                outputs=[guess_output, guess_generated, guess_grid]
            )
        
        with gr.TabItem("SDXL ControlNet"):
            with gr.Row():
                with gr.Column():
                    sdxl_input = gr.Image(label="Input Image", type="pil")
                    
                    with gr.Accordion("Parameters", open=True):
                        sdxl_low_threshold = gr.Slider(0, 255, value=100, label="Low Threshold")
                        sdxl_high_threshold = gr.Slider(0, 255, value=200, label="High Threshold")
                        sdxl_prompt = gr.Textbox(
                            label="Prompt",
                            value="aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
                        )
                        sdxl_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="low quality, bad quality, sketches"
                        )
                        sdxl_num_inference_steps = gr.Slider(20, 50, value=30, label="Number of Inference Steps")
                        sdxl_guidance_scale = gr.Slider(1.0, 20.0, value=7.5, label="Guidance Scale")
                        controlnet_conditioning_scale = gr.Slider(0.0, 1.0, value=0.5, label="ControlNet Conditioning Scale")
                    
                    sdxl_generate_btn = gr.Button("Generate Image")
                
                with gr.Column():
                    sdxl_output = gr.Image(label="Canny Edge Detection", type="pil")
                    sdxl_generated = gr.Image(label="Generated Image", type="pil")
                    sdxl_grid = gr.Image(label="Image Grid", type="pil")
            
            sdxl_generate_btn.click(
                fn=process_sdxl,
                inputs=[
                    sdxl_input,
                    sdxl_low_threshold,
                    sdxl_high_threshold,
                    sdxl_prompt,
                    sdxl_negative_prompt,
                    sdxl_num_inference_steps,
                    sdxl_guidance_scale,
                    controlnet_conditioning_scale
                ],
                outputs=[sdxl_output, sdxl_generated, sdxl_grid]
            )
        
        with gr.TabItem("MultiControlNet"):
            with gr.Row():
                with gr.Column():
                    multi_input = gr.Image(label="Input Image", type="pil")
                    
                    with gr.Accordion("Parameters", open=True):
                        multi_low_threshold = gr.Slider(0, 255, value=100, label="Low Threshold")
                        multi_high_threshold = gr.Slider(0, 255, value=200, label="High Threshold")
                        multi_prompt = gr.Textbox(
                            label="Prompt",
                            value="a giant standing in a fantasy landscape, best quality"
                        )
                        multi_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="monochrome, lowres, bad anatomy, worst quality, low quality"
                        )
                        multi_num_inference_steps = gr.Slider(20, 50, value=25, label="Number of Inference Steps")
                        multi_guidance_scale = gr.Slider(1.0, 20.0, value=7.5, label="Guidance Scale")
                        pose_scale = gr.Slider(0.0, 2.0, value=1.0, label="Pose Control Scale")
                        canny_scale = gr.Slider(0.0, 2.0, value=0.8, label="Canny Control Scale")
                        num_images = gr.Slider(1, 4, value=3, step=1, label="Number of Images")
                    
                    multi_generate_btn = gr.Button("Generate Image")
                
                with gr.Column():
                    multi_canny = gr.Image(label="Canny Edge Detection", type="pil")
                    multi_pose = gr.Image(label="Pose Detection", type="pil")
                    multi_generated = gr.Image(label="Generated Image", type="pil")
                    multi_grid = gr.Image(label="Image Grid", type="pil")
            
            multi_generate_btn.click(
                fn=process_multi_controlnet,
                inputs=[
                    multi_input,
                    multi_low_threshold,
                    multi_high_threshold,
                    multi_prompt,
                    multi_negative_prompt,
                    multi_num_inference_steps,
                    multi_guidance_scale,
                    pose_scale,
                    canny_scale,
                    num_images
                ],
                outputs=[multi_canny, multi_pose, multi_generated, multi_grid]
            )

if __name__ == "__main__":
    demo.launch() 