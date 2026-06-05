# Hate Speech Detection and Visualization using AmpleHate

## Overview

This repository contains my implementation of the **AmpleHate** framework proposed in:

**AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection (EMNLP 2025)**

The project is designed to detect both explicit and implicit hate speech from textual content using transformer-based deep learning models and attention-driven target-context relationship modeling.

In addition to reproducing the original framework, this implementation provides:

- Single sentence hate speech detection
- Batch prediction from text files
- Hate / Non-Hate classification
- Prediction result visualization
- Research and educational use

---

## Author

**Avinash Dixit**  
Ph.D. Research Scholar  
Department of Computer Science and Engineering  
Indian Institute of Technology Indore

### Research Interests

- Livestock Biometrics
- Computer Vision
- Deep Learning
- Artificial Intelligence
- Pattern Recognition

---

## Features

### Single Text Prediction

Analyze an individual sentence and determine whether it contains hateful content.

Example:

```text
Input:
Immigrants are ruining our country.

Output:
Prediction: Hate Speech
Confidence: 0.94
```

### Batch File Prediction

Analyze an entire text file containing multiple sentences.

```bash
python predict_file.py --input sample.txt
```

### Visualization Module

Generate visual explanations and prediction summaries:

```bash
python visualization.py
```

The visualization module can be used to:

- Display prediction distributions
- Compare hate vs non-hate samples
- Visualize model confidence scores
- Generate research-friendly plots
- Support result interpretation and analysis

---

## Model Architecture

The implementation follows the AmpleHate framework:

1. Target Identification
2. Context Representation
3. Attention-Based Relation Modeling
4. Relation Amplification
5. Hate Speech Classification

---

## Installation

```bash
git clone https://github.com/Avinashdixitt/dlproject.git
cd dlproject

pip install -r requirements.txt
```

---

## Usage

### Predict a Single Sentence

```bash
python predict.py
```

### Predict from a Text File

```bash
python predict_file.py --input data.txt
```

### Generate Visualizations

```bash
python visualization.py
```

---

## Research Paper

This implementation is based on:

**Lee, Y., Hahn, J., Ahn, H., & Han, Y. (2025)**

*AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection*

Paper: https://arxiv.org/abs/2505.19528

---

## Disclaimer

This repository is an independent implementation for research and educational purposes. All credit for the original AmpleHate methodology belongs to the authors of the paper. The repository may contain offensive language samples used solely for hate speech detection research.

---

## Citation

```bibtex
@misc{lee2025amplehateamplifyingattentionversatile,
  title={AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection},
  author={Yejin Lee and Joonghyuk Hahn and Hyeseon Ahn and Yo-Sub Han},
  year={2025},
  eprint={2505.19528},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.19528}
}
```