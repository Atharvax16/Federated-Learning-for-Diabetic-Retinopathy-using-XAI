# HealEdge - Federated Learning for Diabetic Retinopathy Detection with XAI

A privacy-preserving federated learning system that enables multiple hospitals to collaboratively train a deep learning model for **Diabetic Retinopathy (DR) grading** without sharing sensitive patient data. The system uses **Federated Averaging (FedAvg)**, **Focal Loss** for class imbalance, and **Grad-CAM** for explainability.

## What This Project Does

Diabetic Retinopathy is a leading cause of blindness worldwide. Early detection through retinal image analysis is critical, but training accurate AI models requires large, diverse datasets — which hospitals cannot easily share due to privacy regulations.

**HealEdge solves this by:**

1. **Federated Learning** — Each hospital trains a local model on its own private data and only shares model weight updates (never raw images) with a central server.
2. **FedAvg Aggregation** — The central server aggregates weight updates from all hospitals into a single global model that benefits from the collective data.
3. **Focal Loss** — Handles severe class imbalance in DR datasets (e.g., "No DR" makes up ~49% of samples while "Severe DR" is only ~5%).
4. **Explainable AI (XAI)** — Grad-CAM heatmaps highlight which regions of retinal images the model focuses on, enabling clinicians to trust and verify predictions.

### DR Classification Grades

| Grade | Diagnosis | Description |
|-------|-----------|-------------|
| 0 | No DR | Healthy retina |
| 1 | Mild | Microaneurysms only |
| 2 | Moderate | More than just microaneurysms |
| 3 | Severe | Extensive intraretinal abnormalities |
| 4 | Proliferative | Neovascularization or vitreous hemorrhage |

## Architecture

```
┌──────────────────────────────────────────────┐
│  Hospital Clients (Federated Nodes)          │
│  • Local training on private patient data    │
│  • ResNet50 with Focal Loss                  │
│  • Send weight updates to central server     │
└─────────────────┬────────────────────────────┘
                  │  FedAvg Aggregation
                  ▼
┌──────────────────────────────────────────────┐
│  Central Server (FastAPI)                    │
│  • Aggregates weights from all hospitals     │
│  • Distributes global model back             │
│  • Serves inference API & dashboard          │
└─────────────────┬────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
 Clinical     Training      Network
 Dashboard    Lab            Map
 (Inference)  (Metrics)     (Topology)
```

**Backend:** Python, FastAPI, PyTorch, timm
**Frontend:** Next.js 16, React 19, TypeScript, Tailwind CSS, Radix UI, Recharts

## Project Structure

```
├── federated/                     # Core federated learning system
│   ├── server/
│   │   ├── main.py               # FastAPI server & training orchestration
│   │   ├── aggregator.py         # FedAvg aggregation algorithm
│   │   └── schemas.py            # Pydantic request/response models
│   ├── client/
│   │   └── hospital.py           # Hospital client simulator
│   └── common/
│       ├── model.py              # ResNet50, EfficientNet-B4, FocalLoss
│       └── config.py             # Shared constants & hyperparameters
│
├── app/                           # Next.js frontend
│   ├── page.tsx                   # Root page
│   ├── layout.tsx                 # Root layout
│   └── globals.css                # Global styles & animations
│
├── components/                    # React UI components
│   ├── app-shell.tsx              # Main app with tab navigation
│   ├── clinical-dashboard.tsx     # Image upload & DR inference
│   ├── model-lab.tsx              # Training metrics & confusion matrix
│   ├── global-network.tsx         # Network topology visualization
│   └── ui/                        # Radix UI component library
│
├── templates/index.html           # Fallback vanilla JS dashboard
├── run.py                         # Entry point — starts FastAPI server
├── best_model.pt                  # Pretrained EfficientNet-B4 weights
├── requirements.txt               # Python dependencies
├── package.json                   # Node.js dependencies
└── 01_aptos_baseline_resnet_class_imbalanced.ipynb  # Training notebook
```

## How to Run

### Prerequisites

- Python 3.9+
- Node.js 18+
- pip and npm (or pnpm)

### 1. Clone the Repository

```bash
git clone https://github.com/Atharvax16/Federated-Learning-for-Diabetic-Retinopathy-using-XAI.git
cd Federated-Learning-for-Diabetic-Retinopathy-using-XAI
```

### 2. Start the Backend (FastAPI Server)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

The backend runs at **http://localhost:8000** and provides the REST API for federated training, hospital management, and image inference.

### 3. Start the Frontend (Next.js)

```bash
# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

The frontend runs at **http://localhost:3000**.

### 4. Use the Application

Open **http://localhost:3000** in your browser. The app has three main tabs:

#### Clinical Dashboard
- Upload a retinal image (drag-and-drop or file select)
- Get an instant DR grade prediction with confidence scores
- View Grad-CAM heatmap showing which regions the model focused on

#### Training Lab
1. Click **"Setup Demo Hospitals"** to create 4 simulated hospitals with varying dataset sizes
2. Click **"Run Demo"** for a quick ~20-second simulation with pre-computed metrics, or **"Run Real Training"** for actual federated learning
3. Monitor live training metrics: loss, accuracy, AUC, F1 score
4. View per-hospital accuracy comparison and per-class recall
5. Inspect the confusion matrix

#### Network Map
- Visualize the federated network topology
- See animated encrypted weight propagation between nodes

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/hospitals/register` | Register a new hospital |
| `GET` | `/api/hospitals` | List all registered hospitals |
| `DELETE` | `/api/hospitals/{id}` | Remove a hospital |
| `POST` | `/api/hospitals/setup-demo` | Create 4 demo hospitals |
| `POST` | `/api/training/configure` | Set training hyperparameters |
| `GET` | `/api/training/config` | Get current configuration |
| `POST` | `/api/training/start` | Start real federated training |
| `POST` | `/api/training/start-demo` | Start demo simulation |
| `POST` | `/api/training/stop` | Stop ongoing training |
| `GET` | `/api/training/status` | Get training state & metrics |
| `GET` | `/api/training/logs` | Get last 100 log entries |
| `GET` | `/api/model/download` | Download global model weights |
| `POST` | `/api/predict` | Upload image for DR prediction |

## ML Models & Training

### Models
- **Training (Federated):** ResNet50 with ImageNet pretrained weights, 5-class output head
- **Inference:** EfficientNet-B4 loaded from `best_model.pt`

### Handling Class Imbalance
DR datasets are heavily imbalanced. The system offers three loss functions:
- **Focal Loss** (default): Down-weights easy examples, focuses on hard-to-classify cases — `FL = (1-pt)^γ × CE`, γ=2.0
- **Weighted Cross Entropy**: Class weights inversely proportional to frequency
- **Standard Cross Entropy**: Baseline

### Federated Training Pipeline
```
For each round (default 10):
  1. Server distributes global model weights to all hospitals
  2. Each hospital trains locally (2 epochs, lr=1e-4, AdamW)
  3. Each hospital returns updated weights + validation metrics
  4. Server aggregates using FedAvg (weighted by sample count)
  5. Global model is updated and metrics are logged
```

### Results (APTOS Dataset — from notebook)
- **Validation Accuracy:** 79.5%
- **Validation AUC:** 0.907
- **Validation F1 (Macro):** 0.658
- Per-class Recall: No DR 98.3% | Mild 85.0% | Moderate 60.6% | Severe 45.5% | Proliferative 53.6%

### Explainability
- **Grad-CAM:** Visualizes model attention on retinal images using gradients from the final convolutional layer
- **Saliency Maps:** Pixel-level gradient importance for interpretability

## Configurable Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_rounds` | 10 | Number of federated training rounds |
| `local_epochs` | 2 | Training epochs per hospital per round |
| `local_lr` | 1e-4 | Local learning rate |
| `batch_size` | 32 | Training batch size |
| `loss_type` | focal | Loss function (focal / weighted_ce / ce) |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| ML Framework | PyTorch, torchvision, timm |
| Backend | FastAPI, Uvicorn, Python |
| Frontend | Next.js 16, React 19, TypeScript |
| Styling | Tailwind CSS v4, Framer Motion |
| UI Components | Radix UI (30+ components) |
| Charts | Recharts, Chart.js |
| Data Science | NumPy, Pandas, scikit-learn, Pillow |

## Key Design Decisions

- **Privacy-Preserving**: Only model weight updates are shared — raw patient images never leave the hospital
- **Focal Loss for Imbalance**: Critical for medical ML where rare severe cases (Grade 3-4) must not be missed
- **Dual Dashboard**: Modern Next.js frontend + fallback vanilla JS dashboard for compatibility
- **Demo Mode**: Pre-computed metrics enable instant demonstration without GPU resources
- **Modular Architecture**: Hospital clients, aggregation server, and inference pipeline are decoupled for easy extension
