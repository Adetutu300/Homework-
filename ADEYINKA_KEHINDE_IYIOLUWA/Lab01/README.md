# Lab 01 - Backpropagation and MNIST Classification
# Lab 01 - Backpropagation and MNIST Classification

Course: Introduction of Intelligent Computing Chip and System

## Objective
Implement and train a three-linear-layer neural network using PyTorch for handwritten digit classification on the MNIST dataset.

## Network Architecture
784 → 256 → 128 → 10

## Requirements
- Python 3.10 or newer
- PyTorch
- torchvision
- NumPy
- Matplotlib

## How to Run

1. Download or clone this repository.
2. Open a terminal in the Lab01 directory.
3. Install the required packages:

   pip install -r requirements.txt

4. Run the experiment:

   python lab01.py

The program automatically downloads MNIST, trains the neural network for 10 epochs, evaluates the model on 10,000 test images, and generates training and validation graphs.

## Submitted Files
- lab01.py — Python source code
- Lab_1_Final.pdf — Laboratory report
- README.md — Experiment description and running instructions
- requirements.txt — Required Python packages

## Experimental Result
The Google Colab experiment achieved 97.95% overall test accuracy. Results may vary slightly depending on the execution environment.
