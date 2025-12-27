# 🐾 Cat vs. Dog Classifier: ResNet18 + Streamlit + Docker

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

![demo](/Demo_Output.jpg?raw=true "demo")

## Project Live Demo

Click here for [Live Demo](https://cats-n-dogs-classification.streamlit.app)

## Project Overview

This project is an end-to-end Computer Vision application that classifies images of cats and dogs. It utilizes a **Fine-Tuned ResNet18** architecture to achieve high accuracy, wrapped in a user-friendly **Streamlit** web interface, and containerized with **Docker** for seamless deployment.

This project demonstrates the transition from a research-based model to a production-ready AI application.

## Features

- **Deep Learning Model:** Fine-tuned ResNet18 (Transfer Learning) on a custom dataset.
- **Interactive UI:** A sleek web interface built with Streamlit for real-time image uploads and predictions.
- **Containerization:** Fully Dockerized to ensure "it works on my machine" translates to "it works everywhere."
- **Scalable Design:** Modular Python code structure.

## Tech Stack

- **Framework:** PyTorch
- **Architecture:** ResNet18 (Pre-trained on ImageNet)
- **Frontend:** Streamlit
- **Deployment:** Docker
- **Libraries:** Torchvision, PIL, NumPy

## Installation & Usage

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Setup

Clone the repository:

```bash
git clone https://github.com/Frizz-0/cat-dog-classifier.git.git
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Training

To train the model with default parameters:

```bash
python models\CatsnDogs.pt --train_nn
```

## Model Architecture

```python
def resnetModel(num_classes):

    model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)

    for params in model.parameters():
        params.requires_grad = False


    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,num_classes)

    return model
```

## Results

The model achieves:

- 99% accuracy on the test set
- 0.03 cross-entropy loss
- Training time of ~20 minutes on GPU T4

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->
