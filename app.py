import streamlit as st
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

st.set_page_config(page_title="ControlNet Canny Edge Detection", layout="wide")
st.title("ControlNet Canny Edge Detection")

# Initialize session state for storing the model
@st.cache_resource
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
    # Remove CPU offload since we're using CPU
    return pipe

# Sidebar controls
st.sidebar.header("Parameters")
low_threshold = st.sidebar.slider("Low Threshold", 0, 255, 100)
high_threshold = st.sidebar.slider("High Threshold", 0, 255, 200)
prompt = st.sidebar.text_area("Prompt", "the mona lisa, masterpiece, best quality, extremely detailed")
negative_prompt = st.sidebar.text_area("Negative Prompt", "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry")
num_inference_steps = st.sidebar.slider("Number of Inference Steps", 20, 50, 30)
guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 20.0, 7.5)

# Add a warning about CPU usage
st.sidebar.warning("⚠️ Running on CPU - Generation will be slower")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and process the uploaded image
    original_image = Image.open(uploaded_file)
    
    # Convert to numpy array and apply Canny edge detection
    image = np.array(original_image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    # Generate new image
    if st.button("Generate Image"):
        with st.spinner("Generating image... This may take a few minutes on CPU"):
            try:
                pipe = load_models()
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=canny_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
                
                # Create and display the image grid
                grid = make_image_grid([original_image, canny_image, output], rows=1, cols=3)
                st.image(grid, caption="Original | Canny Edge | Generated", use_column_width=True)
                
                # Display individual images below the grid
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(original_image, caption="Original Image", use_column_width=True)
                with col2:
                    st.image(canny_image, caption="Canny Edge Detection", use_column_width=True)
                with col3:
                    st.image(output, caption="Generated Image", use_column_width=True)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload an image to get started.") 