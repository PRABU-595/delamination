from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import sys
import os
from pathlib import Path
from PIL import Image
import io
import torchvision.transforms as transforms

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.integrated.framework import IntegratedDelaminationFramework

app = FastAPI(title="Delamination ML App", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves static frontend
static_path = Path(__file__).parent / "ui"
static_path.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(static_path), html=True), name="ui")

# Load Model (Global)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        print("Loading Model...")
        config = {
            'snpi_net': {'adaptive_kernel': {'input_dim': 6}},
            'cad_former': {'d_model': 128, 'n_layers': 4},
            'al_vtfd': {}
        }
        MODEL = IntegratedDelaminationFramework(config).to(DEVICE)
        
        # Load weights
        checkpoint_path = Path("src/training/checkpoints/mega_run/mega_best_model.pt")
        # Fallback to absolute if running from diff dir
        if not checkpoint_path.exists():
             checkpoint_path = Path(__file__).parent.parent.parent / "experiments/checkpoints/mega_run/mega_best_model.pt"

        if checkpoint_path.exists():
            print(f"Loading weights from {checkpoint_path}")
            MODEL.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            MODEL.eval()
        else:
            print("WARNING: No checkpoint found! Using random weights.")
            MODEL.eval()
            
    return MODEL

@app.get("/")
async def root():
    return {"message": "Delamination ML API is running. Go to /ui/index.html"}

@app.post("/predict")
async def predict_delamination(
    file: UploadFile = File(...),
    E11: float = Form(140.0),
    E22: float = Form(10.0),
    G12: float = Form(5.0),
    nu12: float = Form(0.3),
    G1c: float = Form(0.3),
    G2c: float = Form(1.0),
    max_load: float = Form(1000.0)
):
    try:
        model = get_model()
        
        # Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Prepare Physics Inputs from Form Data
        batch_size = 1
        # Physics inputs: [E11, E22, G12, nu12, G1c, G2c]
        physics_inputs = torch.tensor([[E11, E22, G12, nu12, G1c, G2c]], device=DEVICE)
        
        # Loading History: Linear ramp up to max_load
        # Shape: [1, 100]
        loading_history = torch.linspace(0, max_load, 100, device=DEVICE).unsqueeze(0)
        
        # Laminate Config (Default: Quasi-isotropic [0/90/45/-45])
        # We keep this as zeros or a standard encoding for now as user didn't request input
        laminate_config = torch.zeros(batch_size, 4, 64, device=DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model.predict_delamination(
                laminate_config, loading_history,
                physics_inputs=physics_inputs,
                meso_data=input_tensor
            )
            
        return {
            "delamination_severity": float(outputs['delamination_area'].item()),
            "growth_rate_mm_per_cycle": float(outputs['growth_rate'].item()),
            "uncertainty": float(outputs['uncertainty'].mean().item()),
            "status": "CRITICAL" if outputs['delamination_area'].item() > 0.5 else "HEALTHY"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
