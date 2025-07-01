# 🧘‍♀️ Yoga Pose Detection with GCN + Residual + LSTM + MLP (Streamlit App)

A robust, real-time Yoga Pose Classification system using **Graph Convolutional Networks (GCN)** with **residual connections**, **LSTM** for temporal modeling, and **MLP** for classification. Built with **MediaPipe**, **PyTorch Geometric**, and **Streamlit**, the system provides accurate pose prediction along with **real-time correction suggestions**.

---

## 📂 Project Overview

| Component       | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `GCN_Residual_layers_MLP_LSTM_2_layers.ipynb` | Jupyter notebook used to define, train, evaluate, and visualize the full deep model |
| `realtime_code.py`       | Full pipeline for inference: model loading, keypoint extraction, pose classification, and feedback |
| `app.py`                 | Streamlit app interface supporting webcam, image, and video input |
| `best_gcn_residual_mlp_lstm_model.pth` | Trained weights loaded during prediction |
| `requirements.txt`       | All necessary packages and versions |

---

## 🔥 Features

- ✅ Real-time pose prediction using webcam
- ✅ Upload image or video for inference and corrections
- ✅ Intelligent correction tips using anatomical logic
- ✅ Robust GCN-based modeling even with partial visibility
- ✅ Annotated video output and download option
- ✅ Supports 13 yoga poses (Tree, Warrior, Cobra, etc.)

---

## 📦 Installation

# Optional: create a virtual environment

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies

pip install -r requirements.txt

🚀 **Launch the App**

streamlit run app.py

🧠 **Model Architecture (from GCN_Residual_layers_MLP_LSTM_2_layers.ipynb)**

The core of this system is a hybrid deep model:

🔷 **Input**

33 MediaPipe Pose keypoints per frame

Each keypoint has 3 channels: (x, y, visibility) → in_channels = 3

🔶 **Components**

**Layer	Details**

- GCN Layers (x4)	GCNConv layers with residual connections using torch_geometric.nn.GCNConv
- BatchNorm + Dropout	After each GCN layer for regularization
- LSTM (x2)	Models temporal dynamics using torch.nn.LSTM
- Global Pooling	Mean, Max, Add pooling from PyTorch Geometric
- MLP Classifier	4 fully connected layers with BatchNorm, ReLU, Dropout
- Output	5-class Softmax prediction for yoga poses

➕ **Residual Connections**

Each GCN layer adds its output to the previous, like:

res2 = F.relu(self.bn2(self.gcn2(res1, edge_index))) + res1

➕ **LSTM**

The LSTM consumes the padded sequence of nodes per person, captures temporal info:

packed = pack_padded_sequence(...)

lstm_out, _ = self.lstm(packed)

➕ **Final Classification**

Fused graph-pooled + LSTM outputs go through MLP:

Linear(hidden_dim*3 + hidden_dim) → ReLU → Dropout → Softmax

🧠 **Inference Pipeline (realtime_code.py)**

This script implements the real-time detection logic. It includes:

✅ DeepGCN_ResMLP_LSTM

A PyTorch nn.Module loading the trained model architecture with:

GCNConv layers

Residual connections

BatchNorm + Dropout

LSTM (2-layer)

MLP for classification

✅ extract_keypoints_from_frame()

Uses MediaPipe to extract 33 pose landmarks from RGB frames.

Normalizes and filters low-visibility points.

Returns (x, y, visibility) in shape (33, 3).

✅ create_anatomical_edge_index()

Builds a fixed graph of joints (nodes) with biologically inspired connections.

Includes long-range edges to improve message passing (e.g., wrist → hip).

✅ predict_pose()

Converts frame → graph data → feeds into the model.

Uses PyTorch Geometric’s Data() format.

Returns predicted class and confidence.

✅ get_pose_corrections()

Computes angles between joints using vector math.

Provides pose-specific tips, e.g.,:

"Straighten left leg by 12°"

"Raise right arm higher above head"

Each pose has custom logic for checking alignment.

🌐 **Streamlit Interface (app.py)**

✅ **Input Modes Supported**

Mode	        Description

- Webcam	   -> Real-time prediction with overlays

- Image Upload ->Classifies a static image

- Video Upload ->Processes each frame and annotates

✅ **Output**

Pose label (e.g., "tree")

Confidence score (e.g., 92.2%)

Up to 3 correction tips

Bright annotations (green/yellow/red) for visibility

Video download button (after processing)

✅ **Key UI Logic**

Uses cv2.putText() to overlay pose, confidence, and tips.

Uses Streamlit’s st.image() and st.download_button() for interaction.

Handles video upload via tempfile and safe cleanup via atexit.

🧾 **Supported Poses**

The system is trained to classify the following 5 yoga poses:

['downdog', 'goddess', 'plank', 'tree', 'warrior2'] 

🎨 **Sample Output**

Pose: tree (92.2%)

Corrections:

Bend left knee 51°

Raise left arm higher above head

📦 **Dependencies (from requirements.txt)**

streamlit

torch, torchvision, torchaudio

torch-geometric

opencv-python

mediapipe

numpy, pandas, matplotlib, scikit-learn

tqdm

