# HyperDiffRec: Diffusion-Guided Multi-Type Hypergraph for Quality-Aware Multimodal Recommendation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.11+](https://img.shields.io/badge/PyTorch-1.11+-red.svg)](https://pytorch.org/)
[![DGL](https://img.shields.io/badge/DGL-1.0+-green.svg)](https://www.dgl.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **HyperDiffRec** is an advanced multimodal recommendation system that integrates hypergraph architecture, modality completion, and contrastive learning. It achieves exceptional recommendation performance through innovative hypergraph neural networks and advanced modality diffusion completion techniques.

## 🚀 Project Overview

### 🌟 Technical Innovations
1. **Adaptive Quality-Driven Hypergraph Enhancement Framework**: Multi-modal hyperedges, interest aggregation hyperedges, completion quality hyperedges
2. **Advanced Multi-layered Conditional Diffusion Completion Framework**: Intelligent missing detection + high-quality diffusion completion
3. **Adaptive Feature Fusion**: Three fusion strategies - attention, weighted, and concatenation
4. **Fully Independent Implementation**: All dependency code inlined, single-file deployment

## 🏗️ Technical Architecture

### Overall Architecture Diagram
![HyperDiffRec](./framework.svg)

### Core Components

#### 1. 🔄 Advanced Multi-layered Conditional Diffusion Completion Framework (Core Innovation)
- **Intelligent Detection**: Adaptive threshold missing detection
- **Diffusion Completion**: Lightweight diffusion model + high-quality sampling
- **Quality Assessment**: Four quality evaluation algorithms (norm rationality, modal consistency, feature stability, cross-modal correlation)

#### 2. 🌐 Adaptive Quality-Driven Hypergraph Enhancement Framework (Core Innovation)
- **Multi-modal Hyperedges**: Connect item nodes with similar visual/textual features
- **Interest Aggregation Hyperedges**: Interest aggregation connections based on user historical behavior
- **Completion Quality Hyperedges**: Dynamic hyperedge construction based on completion quality assessment

#### 3. 🔧 Adaptive Feature Fusion Layer
- **Attention Fusion**: Dynamically learn weights for contrastive learning features and hypergraph features
- **Weighted Fusion**: Configuration-based fixed weight fusion strategy
- **Concatenation Fusion**: Feature concatenation followed by linear transformation for dimensionality reduction

## 🛠️ Installation & Configuration

### Environment Requirements
- **Python**: 3.10+
- **PyTorch**: 1.11.0 + CUDA 11.8
- **DGL**: 1.0+ (Hypergraph neural network support)


### Environment Configuration
The project provides complete environment configuration:
```bash
# Use the provided environment configuration file
conda env create -f environment.yml
conda activate HyperDiffRec

# Or manually install key dependencies
conda install pytorch==1.11.0 torchvision torchaudio cudatoolkit=11.8 -c pytorch
conda install dgl-cuda11.8 -c dglteam
pip install scipy scikit-learn pyyaml
```

### Dataset Preparation
```bash
# Download dataset
# Dataset link: https://drive.google.com/file/d/16UJ19l4zSIb0oxG2NIYpw3jM8_pc9xSo/view?usp=drive_link

# Extract to data directory
mkdir -p data
# Place the downloaded dataset in the data/ directory
# Directory structure:
# data/
# ├── baby/
# ├── clothing/
# └── sports/
```

## 🚀 Quick Start

python main.py -d baby -m HyperDiffRec


## 📁 Project Structure

```
Eenie-Meenie-master/
├── 📁 common/                  # Core abstract classes and trainer
│   ├── abstract_recommender.py # Recommender model base class
│   ├── loss.py                # Loss function definitions
│   ├── trainer.py             # Trainer
│   └── init.py                # Initialization file
├── 📁 configs/                # Configuration files
│   ├── dataset/               # Dataset configurations
│   │   ├── baby.yaml          # Baby dataset configuration
│   │   ├── clothing.yaml      # Clothing dataset configuration
│   │   └── sports.yaml        # Sports dataset configuration
│   ├── model/                 # Model configurations
│   │   └── HyperDiffRec.yaml  # 🌟 HyperDiffRec model configuration
│   └── overall.yaml           # Global configuration
├── 📁 data/                   # Datasets
│   ├── baby/                  # Baby dataset
│   ├── clothing/              # Clothing dataset
│   └── sports/                # Sports dataset
├── 📁 docs/                   # Project documentation
├── 📁 log/                    # Training logs
├── 📁 models/                 # 🌟 Model implementations
│   └── hyperdiffrec.py        # 🔥 HyperDiffRec fully independent model file
├── 📁 utils/                  # Utility modules
│   ├── configurator.py        # Configuration manager
│   ├── data_utils.py          # Data processing utilities
│   ├── dataloader.py          # Data loader
│   ├── dataset.py             # Dataset class
│   ├── hypergraph_builder.py  # Hypergraph builder
│   ├── logger.py              # Logging utilities
│   ├── metrics.py             # Evaluation metrics
│   ├── misc.py                # Miscellaneous utilities
│   ├── modality_completion.py # Modality completion module
│   ├── multi_metric_evaluator.py # Multi-metric evaluator
│   ├── quick_start.py         # Quick start utilities
│   ├── risk_mitigation.py     # Risk control module
│   ├── topk_evaluator.py      # TopK evaluator
│   └── utils.py               # General utility functions
├── 📄 main.py                 # Main training script
├── 📄 environment.yml         # Environment configuration
├── 📄 framework.svg           # Architecture diagram
└── 📄 README.md              # Project documentation
```


## 🙏 Acknowledgments

- **MMRec**: Provided excellent multimodal recommendation framework foundation
- **DGL**: Deep graph learning library with hypergraph neural network support
- **PyTorch**: Deep learning framework support

---

**🎉 HyperDiffRec: Next-Generation Multimodal Recommendation System!**

**⭐ If this project helps you, please give us a Star!**
