# Multimodal Pneumonia Detection & Representation Analysis

This repository implements a three-part deep learning pipeline for pneumonia detection and representation analysis using the PneumoniaMNIST dataset.

## Pretrained Model

The pretrained model is hosted on Google Drive.

Download it here:
👉 [Download Pretrained Model for task 2](https://drive.google.com/file/d/1_gZhVlnP7vBOj66Fw6X5WUvYS4SaToZ5/view?usp=sharing)

After downloading, place the file inside the `models/` directory.

The project consists of:

1. **Task 1 – CNN Classification (Supervised Learning)**
2. **Task 2 – Vision-Language Model (MedGemma) Analysis**
3. **Task 3 – FAISS Retrieval & Representation Evaluation**

The goal is not only classification performance, but also:

* Cross-modal agreement analysis
* Hallucination detection
* Embedding space separability analysis
* Retrieval-based representation evaluation

---

# 📂 Repository Structure

```
.
├── task1_classification/
│   ├── train.py
│   ├── evaluate.py
│
├── task2_report_generation/
│   ├── images/
│   ├── pneumonia_data_gen.py
│   └── task2_report_gen.py
│
├── task3_retrieval/
│   ├── extract_embeddings_resnet.py
│   ├── build_index_resnet.py
│   ├── evaluate_retrieval_resnet.py
│   ├── visualize_retrieval_resnet.py
│
├── notebooks/
│   ├── task1_classification.ipynb
│   ├── task2_report_generation.ipynb
│   └── task3_retrieval.ipynb
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation (Local Setup – .py Execution)

## 1️⃣ Clone the repository

```bash
git clone https://github.com/Alamnoor/AI_Medical_Imaging_VLM_SR.git
cd AI_Medical_Imaging_VLM_SR
```

## 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv ## create vertual envrnment for the AI_Medical_Imaging_VLM_SR.
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate      # Windows
```

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

All required packages (PyTorch, transformers, FAISS, medmnist, scikit-learn, etc.) are included in `requirements.txt`.

---

# ☁️ Running in Google Colab

You can run each task using the provided Colab notebooks.

## 🔹 Option 1 – Upload Repository ZIP

1. Upload repository ZIP to Colab
2. Extract it

```python
!unzip your_repo.zip
%cd your_repo
!pip install -r requirements.txt
```

---
OR
Use each colab notebook with dependancy mentioned in it. 

## 🔹 Option 2 – Clone from GitHub

```python
!git clone https://github.com/Alamnoor/AI_Medical_Imaging_VLM_SR.git
%cd AI_Medical_Imaging_VLM_SR
!pip install -r requirements.txt
```

For **Task 2 (MedGemma)**:

Enable GPU:

Runtime → Change runtime type → GPU

---

# 🔹 TASK 1 – CNN Classification

## Objective

Train a ResNet18 model for binary pneumonia classification.

---

## ▶ Run Locally (.py)

```bash
cd task1_classification
python train.py
```

Model checkpoint will be saved to:

```
models/
```

Evaluate:

```bash
python evaluate.py
```

Outputs:

* Accuracy
* AUC
* Confusion matrix

---

## ▶ Run in Colab

Open:

```
notebooks/task1_classification.ipynb
```

Or run via terminal cell:

```python
%cd task1_classification
!python train.py
```

---

# 🔹 TASK 2 – Vision-Language Analysis (MedGemma)

## Objective

Use MedGemma to generate structured radiology reports from X-ray images and compare with CNN predictions.

Pipeline includes:

* Report generation
* Structured pneumonia extraction
* CNN vs VLM agreement analysis
* Hallucination flagging

---

## ▶ Run Locally (.py)

```bash
cd task2_report_generation
python pneumonia_data_gen.py ### for data to generate from pneumonia_MNIST dataset into folder images/
python task2_report_gen.py
```

Outputs:

* Structured VLM predictions
* Agreement metrics
* CSV analysis file

---

## ▶ Run in Colab

Open:

```
notebooks/task2_report_generation.ipynb
```

Steps:

1. Install requirements
2. Enable GPU
3. Run cells sequentially
4. Results saved in `/content/`

⚠ Recommended: GPU with ≥6GB VRAM.

---

# 🔹 TASK 3 – FAISS Retrieval & Representation Analysis

## Objective

Evaluate embedding space quality using similarity search and clustering analysis.

Metrics computed:

* Precision@K
* Recall@K
* Intra vs Inter class similarity
* Statistical separability (t-test)
* t-SNE visualization

---

## ▶ Run Locally (.py)

```bash
cd task3_retrieval
python extract_embeddings_resnet.py
python visualize_retrieval_resnet.py
python evaluate_retrieval_resnet.py
python task3_search_resnet.py --query_index 25 --k 5
```
build_index_resnet.py
│   ├── evaluate_retrieval_resnet.py
│   ├── visualize_retrieval_resnet.py
Optional visualization:

```bash
python visualize_retrieval.py
```

---

## ▶ Run in Colab

Open:

```
notebooks/task3_retrieval.ipynb
```

Run cells in order:

1. Extract embeddings
2. Normalize embeddings
3. Build FAISS index
4. Evaluate Precision@K
5. Visualize retrieval

---

# 📊 Summary of Tasks

| Task   | Focus                               |
| ------ | ----------------------------------- |
| Task 1 | Supervised classification           |
| Task 2 | Cross-modal reasoning & alignment   |
| Task 3 | Representation & retrieval analysis |

---

# 💻 Hardware Recommendations

* Task 1 & 3: CPU or GPU
* Task 2: GPU strongly recommended

---

# 📜 License

For research and educational purposes.

---




