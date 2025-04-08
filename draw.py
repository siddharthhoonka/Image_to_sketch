import io
import base64
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
from api import token 
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from groq import Groq

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Global variables for models
# ------------------------------
sd_pipe = None
groq_client = Groq(api_key=token)

# ------------------------------
# Pydantic model for input payload
# ------------------------------
class ImagePayload(BaseModel):
    image_base64: str  # Expecting a base64 string (data URL without the header is fine)

# ------------------------------
# Helper Functions
# ------------------------------
def image_from_base64(b64_string: str) -> Image.Image:
    # Remove header if present (e.g., "data:image/png;base64,...")
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_image_description(image: Image.Image) -> str:
    """Use the Groq vision model to get a description of the image."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Create a chat message that includes both text and image (as a data URI)
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image? Describe it in a way that the rough sketch should be recreated artistically to a refined image in 25 words. Colours are to be assumed, unless shown in the images. Be coloour accurate for objects that are coloured. Mostly image will be black and white, so you can fill and describe colours accordingly"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-11b-vision-preview",
    )
    return chat_completion.choices[0].message.content

def generate_image(prompt: str, pipe: StableDiffusionXLPipeline) -> Image.Image:
    """Generate an image from the given prompt using 2 diffusion steps."""
    result = pipe(prompt, num_inference_steps=2, guidance_scale=0)
    return result.images[0]

def load_sd_pipeline() -> StableDiffusionXLPipeline:
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use the correct checkpoint for your setup

    # Load UNet model
    unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
    state_dict = load_file(hf_hub_download(repo, ckpt), device="cuda")
    unet.load_state_dict(state_dict)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, 
        unet=unet, 
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to("cuda")

    # Use trailing timesteps
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe

# ------------------------------
# Startup event to load models once
# ------------------------------
@app.on_event("startup")
def startup_event():
    global sd_pipe, groq_client
    try:
        sd_pipe = load_sd_pipeline()
        groq_client = groq_client
    except Exception as e:
        raise RuntimeError("Failed to load models") from e

# ------------------------------
# Serve the Frontend HTML at the root route
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail="index.html not found") from e

# ------------------------------
# Endpoint: Process drawing
# ------------------------------
@app.post("/process")
def process_drawing(payload: ImagePayload):
    try:
        # Convert input to PIL image
        image = image_from_base64(payload.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data") from e

    try:
        # Get description from the vision model
        description = get_image_description(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Vision model failed") from e

    try:
        # Generate image from description using Stable Diffusion
        generated_image = generate_image(description, sd_pipe)
        gen_image_b64 = image_to_base64(generated_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Stable Diffusion generation failed") from e

    return {"description": description, "generated_image_base64": gen_image_b64}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
