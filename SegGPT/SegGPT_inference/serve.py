from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
from seggpt_engine import run_one_image, imagenet_mean, imagenet_std
from seggpt_inference import prepare_model  # Import the prepare_model function
import time

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model and device
model = None
device = None
OUTPUT_DIR = "temp_outputs"

def init_model(model_path: str, arch: str = 'seggpt_vit_large_patch16_input896x448', device_name: str = "cuda"):
    """Initialize the model using the prepare_model function"""
    global model, device
    device = torch.device(device_name)
    
    # Use prepare_model function to properly load the model
    model = prepare_model(model_path, arch=arch, seg_type='instance')
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
        start_time = time.time()
        
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

        # Read and process target image (as PNG mask)
        target_content = await prompt_target.read()
        target_img = Image.open(io.BytesIO(target_content)).convert("RGB")
        target_img = target_img.resize((448, 448))
        target_np = np.array(target_img) / 255.

        preprocess_time = time.time()

        # Prepare batch
        tgt = np.concatenate((target_np, target_np), axis=0)
        img_input = np.concatenate((prompt_np, img), axis=0)

        # Normalize
        img_input = (img_input - imagenet_mean) / imagenet_std
        tgt = (tgt - imagenet_mean) / imagenet_std

        # Run inference
        img_input = np.expand_dims(img_input, axis=0)
        tgt = np.expand_dims(tgt, axis=0)
        
        inference_start = time.time()
        output = run_one_image(img_input, tgt, model, device)
        inference_time = time.time() - inference_start

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

        total_time = time.time() - start_time
        print(f"Timing breakdown:")
        print(f"- Preprocessing: {preprocess_time - start_time:.2f}s")
        print(f"- Inference: {inference_time:.2f}s")
        print(f"- Total: {total_time:.2f}s")

        return FileResponse(
            output_path,
            headers={
                "X-Process-Time": str(total_time),
                "X-Inference-Time": str(inference_time)
            }
        )

    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SegGPT Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin-bottom: 20px;
            }
            .drop-zone {
                border: 2px dashed #ccc;
                padding: 20px;
                text-align: center;
                background-color: white;
                border-radius: 8px;
                min-height: 200px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
                position: relative;
            }
            .drop-zone.dragover {
                border-color: #007bff;
                background-color: #f8f9fa;
                transform: scale(1.02);
            }
            .drop-zone.has-file {
                border-color: #28a745;
            }
            .preview {
                max-width: 100%;
                max-height: 300px;
                margin-top: 10px;
            }
            #result {
                grid-column: span 3;
                text-align: center;
            }
            #resultImage {
                max-width: 100%;
                max-height: 500px;
            }
            button {
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin: 20px 0;
                transition: background-color 0.3s ease;
            }
            button:disabled {
                background-color: #ccc !important;
                cursor: not-allowed;
                opacity: 0.7;
            }
            button:not(:disabled):hover {
                background-color: #0056b3;
            }
            .loading {
                display: none;
                margin: 20px 0;
            }
            .error {
                color: red;
                margin: 10px 0;
            }
            .timing {
                color: #666;
                font-size: 14px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <h1>SegGPT Demo</h1>
        <div class="container">
            <div class="drop-zone" id="inputZone">
                <p>Drop or click to upload input image</p>
                <input type="file" accept="image/*" style="display: none" id="inputFile">
                <img id="inputPreview" class="preview">
            </div>
            <div class="drop-zone" id="promptZone">
                <p>Drop or click to upload prompt image</p>
                <input type="file" accept="image/*" style="display: none" id="promptFile">
                <img id="promptPreview" class="preview">
            </div>
            <div class="drop-zone" id="targetZone">
                <p>Drop or click to upload target mask (PNG)</p>
                <input type="file" accept="image/png" style="display: none" id="targetFile">
                <img id="targetPreview" class="preview">
            </div>
        </div>
        <div style="text-align: center;">
            <button id="processButton" disabled>Process Images</button>
            <div class="loading" id="loading">Processing...</div>
            <div class="error" id="error"></div>
            <div class="timing" id="timing"></div>
        </div>
        <div id="result">
            <img id="resultImage">
        </div>
        <script>
            function setupDropZone(zoneId, fileId, previewId) {
                const zone = document.getElementById(zoneId);
                const fileInput = document.getElementById(fileId);
                const preview = document.getElementById(previewId);

                // Click to upload
                zone.addEventListener('click', () => fileInput.click());
                
                // File input change
                fileInput.addEventListener('change', (e) => {
                    console.log(`${fileId} changed:`, e.target.files[0]?.name);
                    if (fileInput.files[0]) {
                        handleFile(fileInput.files[0], preview);
                        updateProcessButton();
                    }
                });

                // Drag and drop
                zone.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    zone.classList.add('dragover');
                });
                
                zone.addEventListener('dragleave', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    zone.classList.remove('dragover');
                });
                
                zone.addEventListener('drop', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    zone.classList.remove('dragover');
                    
                    const file = e.dataTransfer.files[0];
                    if (file) {
                        fileInput.files = e.dataTransfer.files; // Set the files to the input
                        handleFile(file, preview);
                        console.log(`${fileId} dropped:`, file.name);
                        updateProcessButton();
                    }
                });
            }

            function handleFile(file, preview) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    preview.parentElement.classList.add('has-file');
                    console.log('File loaded:', file.name);
                };
                reader.readAsDataURL(file);
            }

            function updateProcessButton() {
                const inputFile = document.getElementById('inputFile').files[0];
                const promptFile = document.getElementById('promptFile').files[0];
                const targetFile = document.getElementById('targetFile').files[0];
                const button = document.getElementById('processButton');
                
                console.log('Files status:', {
                    input: !!inputFile,
                    prompt: !!promptFile,
                    target: !!targetFile
                });
                
                if (inputFile && promptFile && targetFile) {
                    button.disabled = false;
                    button.style.backgroundColor = '#007bff';
                } else {
                    button.disabled = true;
                    button.style.backgroundColor = '#ccc';
                }
            }

            async function processImages() {
                const loading = document.getElementById('loading');
                const error = document.getElementById('error');
                const timing = document.getElementById('timing');
                const button = document.getElementById('processButton');
                const resultImage = document.getElementById('resultImage');

                loading.style.display = 'block';
                error.textContent = '';
                timing.textContent = '';
                button.disabled = true;
                resultImage.style.display = 'none';

                const startTime = performance.now();

                const formData = new FormData();
                formData.append('input_image', document.getElementById('inputFile').files[0]);
                formData.append('prompt_image', document.getElementById('promptFile').files[0]);
                formData.append('prompt_target', document.getElementById('targetFile').files[0]);

                try {
                    const response = await fetch('/segment', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) throw new Error('Processing failed');

                    const blob = await response.blob();
                    resultImage.src = URL.createObjectURL(blob);
                    resultImage.style.display = 'block';

                    const endTime = performance.now();
                    const totalTime = ((endTime - startTime) / 1000).toFixed(2);
                    const serverProcessTime = response.headers.get('X-Process-Time');
                    const inferenceTime = response.headers.get('X-Inference-Time');
                    
                    timing.innerHTML = `
                        Total time: ${totalTime}s<br>
                        Server processing: ${parseFloat(serverProcessTime).toFixed(2)}s<br>
                        Model inference: ${parseFloat(inferenceTime).toFixed(2)}s
                    `;
                } catch (err) {
                    error.textContent = err.message;
                } finally {
                    loading.style.display = 'none';
                    button.disabled = false;
                }
            }
            setupDropZone('inputZone', 'inputFile', 'inputPreview');
            setupDropZone('promptZone', 'promptFile', 'promptPreview');
            setupDropZone('targetZone', 'targetFile', 'targetPreview');
            document.getElementById('processButton').addEventListener('click', processImages);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_arch", type=str, default="seggpt_vit_large_patch16_input896x448")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    init_model(args.model_path, args.model_arch, args.device)
    uvicorn.run(app, host="0.0.0.0", port=args.port)