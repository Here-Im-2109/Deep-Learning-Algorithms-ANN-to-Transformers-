# 🧠 Deep Learning Algorithms: ANN to Transformers

A comprehensive collection of deep learning algorithms, implementations, and learning resources — covering the fundamentals of **Artificial Neural Networks (ANNs)** to modern **Transformer architectures**.

This repository is designed to build a strong understanding of deep learning by combining theoretical concepts, mathematical foundations, Python implementations, and practical experiments.

Explore how neural networks learn, how sequential data is processed, how attention mechanisms work, and how Transformers power modern AI applications.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Learning Roadmap](#-learning-roadmap)
- [Algorithms Covered](#-algorithms-covered)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Key Learning Objectives](#-key-learning-objectives)
- [Getting Started](#-getting-started)
- [Applications](#-applications)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Overview

Deep learning is a subset of machine learning that uses multi-layered neural networks to learn complex patterns from data.

This repository follows a structured learning path, starting with the fundamentals of neural networks and progressing toward advanced architectures used in computer vision, natural language processing, and generative AI.

### What you will find here

- Implementations of fundamental and advanced deep learning algorithms.
- Explanations of important neural network architectures.
- Hands-on Python experiments and model training examples.
- Exploration of activation functions, loss functions, optimizers, and backpropagation.
- Practical implementations of attention mechanisms and Transformers.
- A foundation for exploring modern deep learning research.

---

## 🗺️ Learning Roadmap

```text
Deep Learning
     |
     v
Artificial Neural Networks (ANN)
     |
     v
Convolutional Neural Networks (CNN)
     |
     v
Recurrent Neural Networks (RNN)
     |
     +----> Long Short-Term Memory (LSTM)
     |
     +----> Gated Recurrent Unit (GRU)
     |
     v
Attention Mechanisms
     |
     v
Transformer Architecture
     |
     +----> Encoder-Decoder Transformers
     |
     +----> Vision Transformers (ViT)
     |
     +----> Large Language Models (LLMs)
```

---

## 🔬 Algorithms Covered

### 1. Artificial Neural Networks (ANN)

Explore the fundamental building blocks of deep learning.

**Topics:**
- Perceptrons and multilayer perceptrons (MLPs)
- Forward propagation
- Backpropagation and gradient descent
- Activation functions: ReLU, Sigmoid, Tanh, and Softmax
- Loss functions and optimization algorithms
- Weight initialization and regularization

**Applications:** Classification, regression, and structured-data prediction.

### 2. Convolutional Neural Networks (CNN)

Learn how convolutional networks extract spatial features from images.

**Topics:**
- Convolution and cross-correlation
- Filters, kernels, and feature maps
- Padding, stride, and pooling
- Convolutional and fully connected layers
- Batch normalization and dropout
- CNN architectures and transfer learning

**Applications:** Image classification, object detection, medical image analysis, and image segmentation.

### 3. Recurrent Neural Networks (RNN)

Understand neural networks designed for sequential and time-dependent data.

**Topics:**
- Recurrent connections and hidden states
- Sequential processing
- Backpropagation Through Time (BPTT)
- Vanishing and exploding gradients
- Sequence prediction

**Applications:** Time-series forecasting, sequence classification, and language modeling.

### 4. Long Short-Term Memory (LSTM)

Study a specialized recurrent architecture designed to capture longer-term dependencies.

**Topics:**
- Cell state and hidden state
- Forget gate
- Input gate
- Output gate
- LSTM forward propagation and training

**Applications:** Time-series analysis, speech processing, and sequential prediction.

### 5. Gated Recurrent Unit (GRU)

Explore a computationally simpler alternative to LSTM.

**Topics:**
- Update gate
- Reset gate
- Hidden-state updates
- GRU versus LSTM
- Computational efficiency and sequence modeling

**Applications:** Text classification, time-series forecasting, and sequence modeling.

### 6. Attention Mechanisms

Learn how neural networks dynamically focus on the most relevant parts of an input.

**Topics:**
- Attention scores and alignment
- Query, Key, and Value representations
- Scaled dot-product attention
- Self-attention
- Multi-head attention
- Additive and multiplicative attention
- Attention masking

**Applications:** Machine translation, document understanding, and Transformer-based architectures.

### 7. Transformer Architecture

Understand the architecture that has become foundational to modern language models and many vision models.

**Topics:**
- Positional encoding and positional embeddings
- Scaled dot-product attention
- Multi-head self-attention
- Feed-forward networks
- Residual connections and layer normalization
- Encoder and decoder blocks
- Causal and padding masks
- Training objectives and token prediction

**Applications:** Natural language processing, machine translation, text generation, and multimodal AI.

### 8. Advanced Deep Learning Architectures

Build on the core concepts to explore modern architectures.

Potential extensions include:

- Vision Transformers (ViT)
- U-Net and Attention U-Net
- Residual Networks (ResNet)
- Autoencoders and Variational Autoencoders (VAE)
- Generative Adversarial Networks (GANs)
- GPT-style decoder-only language models
- Large Language Models (LLMs)

---

## 🛠️ Tech Stack

The repository uses Python and popular machine learning and scientific computing tools.

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computation |
| Pandas | Data manipulation and analysis |
| Matplotlib | Visualization and plotting |
| Scikit-learn | Data preprocessing and evaluation |
| TensorFlow / Keras | Deep learning model development |
| PyTorch | Neural network implementation and training |
| Jupyter Notebook | Interactive experimentation |

*The exact dependencies depend on the implementations included in each directory.*

---

## 📁 Repository Structure

A suggested organization for the repository is:

```text
Deep-Learning-Algorithms-ANN-to-Transformers/
│
├── ANN/
│   ├── perceptron.ipynb
│   ├── forward_propagation.ipynb
│   └── backpropagation.ipynb
│
├── CNN/
│   ├── convolution_operations.ipynb
│   └── image_classification.ipynb
│
├── RNN/
│   └── rnn_implementation.ipynb
│
├── LSTM/
│   └── lstm_implementation.ipynb
│
├── GRU/
│   └── gru_implementation.ipynb
│
├── Attention/
│   ├── additive_attention.ipynb
│   └── self_attention.ipynb
│
├── Transformers/
│   ├── positional_encoding.ipynb
│   ├── multi_head_attention.ipynb
│   └── transformer_from_scratch.ipynb
│
├── Advanced_Architectures/
│
├── datasets/
│
├── requirements.txt
├── LICENSE
└── README.md
```

*This structure is illustrative. Adapt the folder and notebook names to match the actual contents of your repository.*

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- Basic knowledge of Python and linear algebra
- Familiarity with machine learning fundamentals
- Jupyter Notebook or JupyterLab
- Optional NVIDIA GPU with compatible drivers and deep learning libraries

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/Deep-Learning-Algorithms-ANN-to-Transformers.git
```

**2. Navigate to the project directory**

```bash
cd Deep-Learning-Algorithms-ANN-to-Transformers
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

**4. Install dependencies**

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the libraries required by your chosen notebooks.

**5. Launch Jupyter Notebook**

```bash
pip install notebook
jupyter notebook
```

Open the notebook for the algorithm you want to explore and execute the cells in order.

---

## 🎓 Key Learning Objectives

By working through this repository, you can develop an understanding of:

- How neural networks learn through forward propagation and backpropagation.
- How gradients optimize model parameters.
- How CNNs extract spatial features from images.
- How RNNs, LSTMs, and GRUs model sequential information.
- How attention mechanisms identify relevant contextual information.
- How Transformers process sequences using self-attention.
- How to implement, train, evaluate, and compare deep learning models.
- How foundational architectures connect to modern AI systems.

---

## 🌍 Applications

The concepts covered in this repository form the foundation for several real-world applications:

- **Computer Vision:** Image classification, object detection, and segmentation.
- **Natural Language Processing:** Text classification, translation, and summarization.
- **Time-Series Analysis:** Forecasting, anomaly detection, and trend analysis.
- **Healthcare AI:** Medical image analysis and computer-aided diagnosis.
- **Generative AI:** Text generation, language models, and multimodal systems.
- **Research and Development:** Experimenting with novel architectures and optimization techniques.

---

## 🔮 Future Enhancements

Planned improvements may include:

- [ ] Implement core algorithms from scratch using NumPy.
- [ ] Add mathematical derivations and visual explanations.
- [ ] Compare ANN, CNN, RNN, LSTM, and GRU architectures experimentally.
- [ ] Implement attention mechanisms step by step.
- [ ] Build a Transformer from scratch using PyTorch.
- [ ] Add model evaluation metrics and training visualizations.
- [ ] Include benchmark datasets and reproducible experiments.
- [ ] Explore Vision Transformers and GPT-style language models.
- [ ] Document computational complexity and parameter counts.
- [ ] Add practical projects and deployment examples.

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

To contribute:

1. Fork the repository.
2. Create a new branch for your changes.
3. Add an implementation, explanation, or improvement.
4. Test your code and update the relevant documentation.
5. Submit a pull request.

Please keep implementations readable, reproducible, and well documented.

---

## 👨‍💻 Author

**Arpan Chandra**

B.Tech in Electrical Engineering | National Institute of Technology Durgapur

**Areas of Interest:**
- Deep Learning and Neural Network Architectures
- Computer Vision and Medical Image Analysis
- Natural Language Processing
- Attention Mechanisms and Transformers
- Generative AI and Large Language Models

GitHub: [YOUR_GITHUB_PROFILE](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project can be distributed under the MIT License. If you choose MIT, add a `LICENSE` file containing the appropriate license text and update this section accordingly.

---

## ⭐ Support

If you find this repository useful for learning deep learning, understanding neural network architectures, or exploring modern AI, consider giving it a **star ⭐** on GitHub.

Keep learning, keep experimenting, and keep building!
