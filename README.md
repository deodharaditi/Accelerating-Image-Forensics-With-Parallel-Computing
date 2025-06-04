#  Accelerating Image Forensics With Parallel Computing

##  Overview

This project explores the application of advanced deep learning and parallel computing strategies to accelerate **image forensics**, focusing on detecting **AI-generated (deepfake) images**. Two core architectures — **ResNet18** and **Vision Transformers (ViT)** — were used to classify real vs. AI-generated content, while various parallelism techniques were benchmarked for performance and scalability.

---

##  Key Highlights

-  **Models**: ResNet18 (lightweight CNN) & Vision Transformer (ViT)
-  **Dataset**: 90,000+ real and AI-generated images from [Kaggle](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images)
-  **Parallelism Explored**:
  - Distributed Data Parallel (DDP) across 1, 2, and 4 GPUs
  - Model Parallelism (manual split across GPUs)
  - Mixed Precision Training (AMP with FP16)
  - CPU Thread Parallelism (1, 2, 4 workers)
-  **Benchmarking**:
  - Measured speedup, efficiency, memory usage, training time, accuracy
  - Comparative visualizations across CPU/GPU setups

---

##  Methodology

1. **Preprocessing**:
   - Resized images to 224×224
   - Normalized using ImageNet statistics
   - Loaded via PyTorch `DataLoader` with multiprocessing
2. **Training**:
   - ResNet18 and ViT trained using `torchrun` or model-split logic
   - All benchmarks tested with and without AMP
   - CPU parallelism used for baseline speed comparison
3. **Evaluation**:
   - Accuracy measured on test/validation sets
   - GPU memory tracked via `torch.cuda.max_memory_allocated()`
   - Speedup = Baseline Time / Current Time
   - Efficiency = Speedup / Number of Devices

---

##  Results Summary

| Method                | Model     | GPUs | Speedup | Accuracy | Training Time (s) | Max GPU Mem (GB) |
|----------------------|-----------|------|---------|----------|--------------------|------------------|
| Single GPU (Baseline)| ResNet18  | 1    | 1×      | 98.58%   | 743.45             | 0.50             |
| Model Parallel        | ResNet18  | 2    | ~1×     | 98.75%   | 741.73             | 0.46             |
| DDP + AMP             | ViT       | 4    | 2.25×   | 99.90%   | 254.92             | ~0.40            |
| CPU Parallel (2 Core) | ResNet18  | CPU  | 1.42×   | 96.42%   | (Scaled from 25%)  | —                |

>  ViT benefited greatly from GPU scaling. ResNet18 showed limited GPU scaling due to its small model size but benefited from memory-aware training.

---

##  Streamlit App

Experience our model live through a user-friendly UI:

 [Launch App](https://ai-vs-real-image-detection-hpc.streamlit.app/)

**Features**:
- Upload real/AI images for classification
- Compare ResNet18 vs ViT predictions
- Download logs and CSV outputs
- Batch processing + confidence scores

---

##  Full Report & Code

This repository includes:
-  Performance plots (training time, speedup, efficiency)
-  Model training/evaluation code (CPU & GPU)
-  Full project report with graphs and conclusions

---

##  Acknowledgements

- **CSYE7105 Instructor (High Performance ML & AI)** – Professor Handan Liu
- **Teammate** – Kaushik Malikireddy
- Built with: PyTorch, Streamlit, OpenCV, NCCL, DDP

---

##  Future Scope

- Integrate **Fully Sharded Data Parallel (FSDP)**
- Extend to **video-based deepfake detection**
