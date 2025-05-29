import gradio as gr
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

def load_models():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float32,  # Use float32 for CPU
        use_safetensors=True
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,  # Use float32 for CPU
        use_safetensors=True
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def process_image(image, low_threshold, high_threshold, prompt, negative_prompt, num_inference_steps, guidance_scale):
    # Convert to numpy array and apply Canny edge detection
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    # Generate new image
    pipe = load_models()
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

# Create the Gradio interface
with gr.Blocks(title="ControlNet Canny Edge Detection") as demo:
    gr.Markdown("# ControlNet Canny Edge Detection")
    gr.Markdown("⚠️ Running on CPU - Generation will be slower")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            
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
            
            generate_btn = gr.Button("Generate Image")
        
        with gr.Column():
            canny_output = gr.Image(label="Canny Edge Detection", type="pil")
            generated_output = gr.Image(label="Generated Image", type="pil")
            grid_output = gr.Image(label="Image Grid", type="pil")
    
    generate_btn.click(
        fn=process_image,
        inputs=[
            input_image,
            low_threshold,
            high_threshold,
            prompt,
            negative_prompt,
            num_inference_steps,
            guidance_scale
        ],
        outputs=[canny_output, generated_output, grid_output]
    )

if __name__ == "__main__":
    demo.launch() 