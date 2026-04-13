[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/oAity9uG)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=23223122)



## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Open Jupyter Notebook
```bash
jupyter notebook
```

### 3. Run the project
- Open:  
  code/final_code.ipynb
- Run all cells sequentially (top to bottom)
OR
- see this colab link for direct execution: https://colab.research.google.com/drive/1BL29GylIlpcMi-zaalU0YrCAmCdTju81?usp=sharing

⚠️ Important:
- Do NOT skip cells
- Dataset generation happens inside the notebook
- Training will take some time depending on hardware

---

## 🧠 Method

- Multi-view images as input
- CNN-based encoder extracts features
- Features aggregated across views
- 3D decoder predicts voxel grid (32×32×32)
- Loss: Binary Cross Entropy

---

## 📊 Results

The model successfully reconstructs basic 3D shapes from multiple views.

Examples:
- Ground truth vs predicted voxel grids
- Training loss curve
- Exported 3D mesh (.ply) for visualization

Check:
images/
results/

---

## 📁 Directory Structure

.
├── code/
│   └── final_code.ipynb
├── images/
├── results/
├── README.md
├── requirements.txt

---

## 🎥 Demo

A short demo video (2–3 minutes) is included in:
video/demo.mp4

---

## 🤖 ChatGPT Usage

We used ChatGPT for:
- Debugging PyTorch and visualization issues
- Understanding voxel-based reconstruction
- Structuring and organizing the code

All final implementation and understanding were verified by us.

---

## 📜 Honor Code

see readme.txt

---

## 📄 Reports

- Project Report: report/ProjectReport.pdf
- Reflection Essay: report/ReflectionEssay.pdf

---

## ⚠️ Notes

- Open3D visualization may not work in Google Colab (headless environment)
- Mesh outputs can be downloaded from:
  results/model.ply
  and viewed locally using MeshLab or Open3D
- see this colab link for direct execution: https://colab.research.google.com/drive/1BL29GylIlpcMi-zaalU0YrCAmCdTju81?usp=sharing

