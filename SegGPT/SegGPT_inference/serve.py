from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
import torch
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
from seggpt_engine import run_one_image, imagenet_mean, imagenet_std

app = FastAPI()

# Global variables for model and device
model = None
device = None
OUTPUT_DIR = "temp_outputs"

def init_model(model_path: str, device_name: str = "cuda"):
    global model, device
    device = torch.device(device_name)
    # Load your model here
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/segment")
async def segment_image(
    input_image: UploadFile = File(...),
    prompt_image: UploadFile = File(...),
    prompt_target: UploadFile = File(...),
):
    try:
        # Read and process input image
        input_content = await input_image.read()
        img = Image.open(io.BytesIO(input_content)).convert("RGB")
        input_np = np.array(img)
        size = img.size
        img = np.array(img.resize((448, 448))) / 255.

        # Read and process prompt image
        prompt_content = await prompt_image.read()
        prompt_img = Image.open(io.BytesIO(prompt_content)).convert("RGB")
        prompt_img = prompt_img.resize((448, 448))
        prompt_np = np.array(prompt_img) / 255.

        # Read and process target image
        target_content = await prompt_target.read()
        target_img = Image.open(io.BytesIO(target_content)).convert("RGB")
        target_img = target_img.resize((448, 448))
        target_np = np.array(target_img) / 255.

        # Prepare batch
        tgt = np.concatenate((target_np, target_np), axis=0)
        img_input = np.concatenate((prompt_np, img), axis=0)

        # Normalize
        img_input = (img_input - imagenet_mean) / imagenet_std
        tgt = (tgt - imagenet_mean) / imagenet_std

        # Run inference
        img_input = np.expand_dims(img_input, axis=0)
        tgt = np.expand_dims(tgt, axis=0)
        output = run_one_image(img_input, tgt, model, device)

        # Post-process
        output = torch.nn.functional.interpolate(
            output[None, ...].permute(0, 3, 1, 2),
            size=[size[1], size[0]],
            mode='nearest',
        ).permute(0, 2, 3, 1)[0].numpy()
        
        output = input_np * (0.6 * output / 255 + 0.4)
        output = output.astype(np.uint8)

        # Save output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.png")
        output_img = Image.fromarray(output)
        output_img.save(output_path)

        return FileResponse(output_path)

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    init_model(args.model_path, args.device)
    uvicorn.run(app, host="0.0.0.0", port=args.port)