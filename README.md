# AryaXAI: DLBacktrace

This repository contains the source code for the paper: [DLBacktrace: A Model Agnostic Explainability for any Deep Learning Models](https://arxiv.org/abs/2411.12643v1).
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Introducing DLBacktrace, a method for analyzing neural networks by tracing the relevance of each component from output to input, to understand how each part contributes to the final prediction. Developed by [AryaXAI](https://www.aryaxai.com/) Team. It offers two modes: Default and Contrast, and is compatible with TensorFlow and PyTorch. 



## Overview

The DlBacktrace Module, a patent-pending algorithm developed by AryaXAI, is designed to significantly enhance the explainability of Deep Learning AI models.

## Features

- **Explainability:** Gain deep insights into your AI models by using the Backtrace algorithm, providing multiple explanations for their decisions.

- **Consistency:** Ensure consistent and accurate explanations across different scenarios and use cases.

- **Mission-Critical Support:** Tailored for mission-critical AI use cases where transparency is paramount.

## Installation

To integrate the Backtrace Module into your project, follow these simple steps:

```bash
pip install dl-backtrace
```

## Usage 

### Tensoflow-Keras based models

```python
from dl_backtrace.tf_backtrace import Backtrace as B
```

### Pytorch based models

```python
from dl_backtrace.pytorch_backtrace import Backtrace as B
```

### Evalauting using Backtrace:

1. Step - 1: Initialize a Backtrace Object using your Model
```python
backtrace = B(model=model)
```

2. Step - 2: Calculate layer-wise output using a data instance

```python
layer_outputs = backtrace.predict(test_data[0])
```

3. Step - 3: Calculate layer-wise Relevance using Evaluation 
```python
relevance = backtrace.eval(layer_outputs,mode='default',scaler=1,thresholding=0.5,task="binary-classification")
```

#### Depending on Task we have several attributes for Relevance Calculation in Evalaution:

| Attribute    | Description | Values |
|--------------|-------------|--------|
| mode         | evaluation mode of algorithm | { default, contrastive}|
| scaler       | Total / Starting Relevance at the Last Layer | Integer ( Default: None, Preferred: 1)|
| thresholding | Thresholding Model Prediction in Segemntation Task to select Pixels predicting the actual class. (Only works in Segmentation Tasks) |  Default:0.5      |
| task         | The task of the Model | { binary-classification, multi-class classification, bbox-regression, binary-segmentation} |
| model-type   | Type of the Model | {Encoder} |

## Example Notebooks : 

### Tensorflow-Keras : 

| Name        | Task        | Link                          |
|-------------|-------------|-------------------------------|
| Backtrace Loan Classification Tabular Dataset | Binary Classification | [Colab Link](https://colab.research.google.com/drive/1H5jaryVPEAQuqk9XPP71UIL4cemli98K?usp=sharing) |
| Backtrace Image FMNIST Dataset | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1BZsdo7IWYGhdy0Pg_m8r7c3COczuW_tG?usp=sharing)  |
| Backtrace CUB Bounding Box Regression Image Dataset | Single Object Detection | [Colab Link](https://colab.research.google.com/drive/15mmJ2aGt-_Ho7RdPWjNEEoFXE9mu9HLV?usp=sharing) |
| Backtrace Next Word Generation Textual Dataset | Next Word Generation | [Colab Link](https://colab.research.google.com/drive/1iRMMcEm4iMVuk236vDtEB6rKsirny8cB?usp=sharing) |
| Backtrace ImDB Sentiment Classification Textual Dataset | Sentiment Classification | [Colab Link](https://colab.research.google.com/drive/1L5nEMO6H8pbGo1Opd9S4mbYPMmq5Vl-M?usp=sharing)|
| Backtrace Binary Classification Textual Dataset | Binary Classification | [Colab Link](https://colab.research.google.com/drive/1PxFY4hEhcIr4nTVyfwLE_29CzQTCD3dA?usp=sharing) |
| Backtrace Multi-Class NewsGroup20 Classification Textual Dataset | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1u3B18TZwfTdYJeYBGcHQ0T3fzfM2USGT?usp=sharing) |
| Backtrace CVC-ClinicDB Colonoscopy Binary Segmentation | Organ Segmentation | [Colab Link](https://colab.research.google.com/drive/1cUNUao7fahDgndVI-cpn2iSByTiWaB4j?usp=sharing) | 
| Backtrace CamVid Road Car Binary Segmentation | Binary Segmentation | [Colab Link](https://colab.research.google.com/drive/1OAY7aAraKq_ucyVt5AYPBD8LkQOIuy1C?usp=sharing) |
| Backtrace Transformer Encoder for Sentiment Analysis | Binary Classification | [Colab Link](https://colab.research.google.com/drive/1H7-4ox3YWMtoH0vptYGXaN63PRJFbTrX?usp=sharing) |

### Pytorch : 
| Name        | Task        | Link                          |
|-------------|-------------|-------------------------------|
| Backtrace Tabular Dataset | Binary Classification | [Colab Link](https://colab.research.google.com/drive/1_r-IS7aIuATSvGNRLk8VDVVLkDSaKCpD?usp=sharing)|
| Backtrace Image Dataset | Multi-Class Classification | [Colab Link](https://colab.research.google.com/drive/1v2XajWtIbf7Vt31Z1fnKnAjyiDzPxwnU?usp=sharing) |

## Supported Layers and Future Work :

### Tensorflow-Keras:

- [x] Dense (Fully Connected) Layer
- [x] Convolutional Layer (Conv2D,Conv1D)
- [x] Transpose Convolutional Layer (Conv2DTranspose,Conv1DTranspose)
- [x] Reshape Layer
- [x] Flatten Layer
- [x] Global Max Pooling (2D & 1D) Layer
- [x] Global Average Pooling (2D & 1D) Layer
- [x] Max Pooling (2D & 1D) Layer
- [x] Average Pooling (2D & 1D) Layer
- [x] Concatenate Layer
- [x] Add Layer
- [x] Long Short-Term Memory (LSTM) Layer
- [x] Dropout Layer
- [x] Embedding Layer
- [x] TextVectorization Layer
- [x] Self-Attention Layer
- [x] Cross-Attention Layer
- [x] Feed-Forward Layer
- [x] Pooler Layer
- [x] Decoder LM (Language Model) Head
- [ ] Other Custom Layers 

### Pytorch :

(Note: Currently we only Support Binary and Multi-Class Classification in Pytorch, Segmentation and Single Object Detection will be supported in the next release.)

- [x] Linear (Fully Connected) Layer
- [x] Convolutional Layer (Conv2D)
- [x] Reshape Layer
- [x] Flatten Layer
- [x] Global Average Pooling 2D Layer (AdaptiveAvgPool2d)
- [x] Max Pooling 2D Layer (MaxPool2d)
- [x] Average Pooling 2D Layer (AvgPool2d)
- [x] Concatenate Layer
- [x] Add Layer
- [x] Long Short-Term Memory (LSTM) Layer
- [X] 1d Convolution Layer (Conv1d)
- [X] 1d Pooling Layers (AvgPool1d,MaxPool1d,AdaptiveAvgPool1d,AdaptiveMaxPool1d)
- [X] Transpose Convolution Layers (ConvTranspose2d,ConvTranspose1d)
- [ ] Global Max Pooling 2D Layer (AdaptiveMaxPool2d)
- [ ] Dropout Layer
- [X] Embedding Layer
- [ ] EmbeddingBag Layer
- [ ] Other Custom Layers

## Limitations 
### Limitations for both TensorFlow and PyTorch

1. **Custom Layers Not Supported**: The implementation of custom layers is currently not supported in the framework.  
2. **Model Requirements**: All models must be developed entirely from scratch, adhering to the framework's specifications.  
3. **Layer Abstraction**: Custom layers using class-based abstractions are not supported. Instead, layers should be directly utilized within `tf.Sequential` for TensorFlow and `nn.Sequential` for PyTorch.

### Additional Limitations Torch

1. **Torch Functional Not Supported**: The use of `torch.functional` for activation functions is not supported. Instead, define them as attributes in the constructor.  
2. **Single Function Per Line**: Avoid using multiple functions in a single line within the forward pass.  
3. **Tensor Operations in Forward Pass**: Operations like `transpose` and other tensor manipulations should not be used directly in the forward pass.

We hope to address these limitations soon in the future versions.

## Getting Started
If you're new to Backtrace, check out our Example Notebooks for a quick and comprehensive guide covering a variety of tasks.

## Contributing
We welcome contributions from the community.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citations
This code is free. So, if you use this code anywhere, please cite us:
```
@misc{sankarapu2024dlbacktracemodelagnosticexplainability,
      title={DLBacktrace: A Model Agnostic Explainability for any Deep Learning Models}, 
      author={Vinay Kumar Sankarapu and Chintan Chitroda and Yashwardhan Rathore and Neeraj Kumar Singh and Pratinav Seth},
      year={2024},
      eprint={2411.12643},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.12643}, 
}
```

## Get in touch
Contanct us at [AryaXAI](https://www.aryaxai.com/).
